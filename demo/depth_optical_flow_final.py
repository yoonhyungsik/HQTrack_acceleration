import os
import glob
import torch
import cv2
import numpy as np
import random
import sys
import torch.nn.functional as F
from PIL import Image
import importlib
from torchvision import transforms
from os.path import join
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from transformers import pipeline
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.image import flip_tensor
AOT_PATH = os.path.join(os.path.dirname(__file__), '..')
import dataloaders.video_transforms as tr
from networks.engines import build_engine
from utils.checkpoint import load_network
from networks.models import build_vos_model
from utils.metric import pytorch_iou
from pathlib import Path
from collections import deque  # 뎁스 변화량 저장을 위한 추가
from scipy.stats import norm

base_path = os.path.dirname(os.path.abspath(__file__))
demo_video = 'p_09'
img_files = sorted(glob.glob(join(base_path, demo_video, '*.jp*'))) 
point_box_prompts=[]

# 랜덤 시드 설정
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)

cur_colors = [(0, 255, 255), # yellow b g r
              (255, 0, 0), # blue
              (0, 255, 0), # green
              (0, 0, 255), # red
              (255, 255, 255), # white
              (0, 0, 0), # black
              (255, 255, 0), # Cyan
              (225, 228, 255), # MistyRose
              (180, 105, 255), # HotPink
              (255, 0, 255), # Magenta
              ]*100

class MotionDepthAnalyzer:
    def __init__(self, use_ci=True, use_iqr=True, use_ewma=True):
        self.depth_history = {}  # {obj_id: deque([(frame_idx, depth), ...])}
        self.motion_history = {}  # {obj_id: deque([...])}
        self.bayesian_stats = {}  # {obj_id: (mean, std)}
        self.accel_history = {}
        self.use_ci = use_ci
        self.use_iqr = use_iqr
        self.use_ewma = use_ewma
        self.window_size = 10
        self.min_N = 5
        self.max_N = 20
        self.base_N = 10
        self.removal_threshold = 0.02
        self.max_frame_diff = 10
        self.alpha_base_k = 0.1
        self.prev_motion_magnitude = {}  # For computing acceleration
        self.total_skipped = 0

    def compute_optical_flow(self, prev_frame, curr_frame, obj_mask):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        if np.any(obj_mask):
            mean_x = np.mean(flow_x[obj_mask])
            mean_y = np.mean(flow_y[obj_mask])
        else:
            mean_x, mean_y = 0.0, 0.0
        return (mean_x, mean_y)

    def compute_motion_magnitude(self, motion_vector):
        return np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)

    def compute_accel_threshold(self, obj_id, current_accel, frame_idx):
        return self.compute_final_threshold(obj_id, current_accel, frame_idx, mode='accel')

    def compute_motion_threshold(self, obj_id, current_motion, frame_idx):
        return self.compute_final_threshold(obj_id, current_motion, frame_idx, mode='motion')
       
    def update_depth_history(self, obj_id, depth_value, frame_idx):
        if obj_id not in self.depth_history:
            self.depth_history[obj_id] = deque(maxlen=self.base_N)
        history = self.depth_history[obj_id]
        history.append((frame_idx, depth_value))
        if len(history) < 3:
            return history
        depth_values = np.array([d for _, d in history])
        if np.any(np.isnan(depth_values)) or len(depth_values) < 3:
            return history
        differences = np.abs(depth_values - depth_value)
        std_depth = np.std(np.diff(depth_values)) if len(depth_values) > 1 else 0
        new_N = min(self.max_N, len(history) + 2) if std_depth > 0.05 else max(self.min_N, len(history) - 2)
        self.depth_history[obj_id] = deque(history, maxlen=new_N)
        if len(history) > self.min_N:
            idx_to_remove = np.argmin(differences)
            hist_list = list(history)
            hist_list.pop(idx_to_remove)
            self.depth_history[obj_id] = deque(hist_list, maxlen=new_N)
        filtered_history = [(f, d) for f, d in self.depth_history[obj_id] if frame_idx - f <= self.max_frame_diff]
        filtered_history.sort(reverse=True, key=lambda x: x[0])
        self.depth_history[obj_id] = deque(filtered_history[:new_N], maxlen=new_N)
        return self.depth_history[obj_id]

    def update_motion_history(self, obj_id, motion_mag):
        if obj_id not in self.motion_history:
            self.motion_history[obj_id] = deque(maxlen=self.window_size)
        self.motion_history[obj_id].append(motion_mag)

    def compute_dynamic_alpha(self, values):
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(values) < 10:
            return 0.9
        std_val = np.std(values)
        median_val = np.median(values)
        k = self.alpha_base_k * median_val if median_val > 0 else self.alpha_base_k
        alpha = 0.95 - (std_val / (std_val + k))
        return np.clip(alpha, 0.7, 0.95)

    def compute_dynamic_confidence(self, values):
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(values) < 5:
            return 0.95
        Q1, Q3 = np.percentile(values, [25, 75])
        IQR = Q3 - Q1
        outliers_iqr = np.sum((values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR)))
        median = np.median(values)
        MAD = np.median(np.abs(values - median)) * 1.4826
        outliers_mad = np.sum((values < (median - 3 * MAD)) | (values > (median + 3 * MAD)))
        outliers = (outliers_iqr + outliers_mad) / 2
        outlier_ratio = outliers / len(values)
        confidence = 0.95 * np.exp(-3 * outlier_ratio)
        return np.clip(confidence, 0.9, 0.95)

    def compute_accel_threshold(self, obj_id, current_accel, frame_idx):
        return self.compute_final_threshold(obj_id, current_accel, frame_idx, mode='accel')

    def compute_final_threshold(self, obj_id, new_value, frame_idx, mode='depth'):
        if mode == 'motion':
            history = self.motion_history.get(obj_id, [])
            min_val, max_val = 0.1, 10.0
        elif mode == 'accel':
            history = self.accel_history.get(obj_id, [])
            min_val, max_val = 0.01, 5.0
        else:
            history = self.update_depth_history(obj_id, new_value, frame_idx)
            min_val, max_val = 1.0, 1000.0

        values = [v for v in history if not np.isnan(v)] if mode != 'depth' else [d for _, d in history if not np.isnan(d)]
        if len(values) < 3:
            print(f"[Threshold:{mode}] Obj {obj_id} - Not enough valid history, using current value: {new_value:.4f}")
            return new_value

        std_dev = np.std(values)
        mean_val = np.mean(values)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1

        percentile_threshold = q3 + 1.5 * iqr
        variance_threshold = mean_val + 1.5 * std_dev * (1 / (1 + std_dev))

        # EWMA 기반 confidence interval
        confidence = self.compute_dynamic_confidence(values)
        z_score = norm.ppf(1 - (1 - confidence) / 2)
        alpha_ewma = 0.2
        ewma_mean = values[0]
        for val in values[1:]:
            ewma_mean = alpha_ewma * val + (1 - alpha_ewma) * ewma_mean
        ci_threshold = ewma_mean + z_score * std_dev * min(1.0, 0.1 + std_dev / max(np.max(values), 1e-6))

        # ✅ 구성에 따라 포함할 threshold만 선택
        thresholds = []
        labels = []

        if getattr(self, 'use_ewma', True):
            thresholds.append(max(variance_threshold, 1e-3))
            labels.append(f"Variance: {variance_threshold:.4f}")
        if getattr(self, 'use_iqr', True):
            thresholds.append(max(percentile_threshold, 1e-3))
            labels.append(f"Percentile: {percentile_threshold:.4f}")
        if getattr(self, 'use_ci', True):
            thresholds.append(max(ci_threshold, 1e-3))
            labels.append(f"CI: {ci_threshold:.4f}")

        if not thresholds:
            print(f"[Threshold:{mode}] Obj {obj_id} - No thresholds selected, using mean: {mean_val:.4f}")
            return mean_val

        thresholds = np.array(thresholds)
        total = np.sum(thresholds) + 1e-6
        weights = thresholds / total
        final_threshold = np.sum(weights * thresholds)
        final_threshold = np.clip(final_threshold, min_val, max_val)

        print(f"[Threshold:{mode}] Obj {obj_id} | Std: {std_dev:.4f}, Mean: {mean_val:.4f}, "
            + ", ".join(labels) + f", Final: {final_threshold:.4f}")
        return final_threshold

    def should_skip_tracking(self, obj_id, depth_value, motion_mag, frame_idx=0, total_frames=None):
        if obj_id not in self.depth_history or len(self.depth_history[obj_id]) < 3:
            return False
        valid_depths = [d for _, d in self.depth_history[obj_id] if not np.isnan(d)]
        if not valid_depths:
            return False

        if total_frames is not None and (frame_idx < 3 or total_frames - frame_idx <= 5):
            print(f"[Decision] Obj {obj_id} → ✅ Forced tracking in first/last frames")
            return False

        # Compute acceleration
        if not hasattr(self, 'prev_motion_magnitude'):
            self.prev_motion_magnitude = {}
        prev_mag = self.prev_motion_magnitude.get(obj_id, motion_mag)
        acceleration = abs(motion_mag - prev_mag)
        self.prev_motion_magnitude[obj_id] = motion_mag

        # Store motion and accel values in history
        if obj_id not in self.motion_history:
            self.motion_history[obj_id] = deque(maxlen=self.window_size)
        if obj_id not in self.accel_history:
            self.accel_history[obj_id] = deque(maxlen=self.window_size)
        self.motion_history[obj_id].append(motion_mag)
        self.accel_history[obj_id].append(acceleration)

        # Ensure historical values are properly registered before thresholding
        if len(self.motion_history[obj_id]) < 3:
            print(f"[Threshold:motion] Obj {obj_id} - Not enough valid history, using current value: {motion_mag:.4f}")
            motion_thresh = motion_mag + 0.1
        else:
            motion_thresh = self.compute_final_threshold(obj_id, motion_mag, frame_idx, mode='motion') + 0.05

        if len(self.accel_history[obj_id]) < 3:
            print(f"[Threshold:accel] Obj {obj_id} - Not enough valid history, using current value: {acceleration:.4f}")
            accel_thresh = acceleration + 0.05
        else:
            accel_thresh = self.compute_final_threshold(obj_id, acceleration, frame_idx, mode='accel') + 0.02

        depth_thresh = self.compute_final_threshold(obj_id, depth_value, frame_idx, mode='depth') + 1.0

        # Determine if tracking should be skipped based on violations
        violations = sum([
            depth_value < depth_thresh,
            motion_mag < motion_thresh,
            acceleration < accel_thresh
        ])
        skip = violations >= 3

        print(f"[Analyzer] Obj {obj_id} | Depth: {depth_value:.4f} (th: {depth_thresh:.4f}), "
            f"Motion: {motion_mag:.4f} (th: {motion_thresh:.2f}), Accel: {acceleration:.4f} (th: {accel_thresh:.2f}) → Skip: {skip} violations: {violations}")
        if skip:
            print(f"[Decision] Obj {obj_id} → ❌ Tracking Skipped")
            self.total_skipped += 1
        else:
            print(f"[Decision] Obj {obj_id} → ✅ Tracking Proceeded")    
        return skip


    def compute_fusion_weights(self, vx, vy, z_change, motion_mag=None):
        # 기본 XY 변화량
        xy_mag = np.sqrt(vx**2 + vy**2)

        # NaN 방어 처리
        if np.isnan(xy_mag) or np.isnan(z_change) or (motion_mag is not None and np.isnan(motion_mag)):
            print("[WARNING] NaN detected in fusion weight computation. Falling back to default weights.")
            return 0.5, 0.5, "NaN"

        total_mag = xy_mag + z_change + 1e-6

        # 기본 가중치 (normalized)
        base_w_motion = xy_mag / total_mag
        base_w_depth = z_change / total_mag

        # 동적 조정
        if motion_mag is not None:
            # Soft-normalized 변화량
            motion_norm = np.log1p(motion_mag)
            depth_norm = np.log1p(z_change)
            total_norm = motion_norm + depth_norm + 1e-6

            dyn_w_motion = motion_norm / total_norm
            dyn_w_depth = depth_norm / total_norm

            # 기본과 동적 가중치의 평균
            w_motion = 0.5 * base_w_motion + 0.5 * dyn_w_motion
            w_depth = 1.0 - w_motion
        else:
            w_motion = base_w_motion
            w_depth = base_w_depth

        dominant = "XY" if w_motion > w_depth else "Z"
        print(f"[Fusion Weights] vx: {vx:.3f}, vy: {vy:.3f}, z: {z_change:.3f} → "
            f"w_xy: {w_motion:.2f}, w_z: {w_depth:.2f}, Dominant: {dominant}")
        return w_motion, w_depth, dominant

         
analyzer = MotionDepthAnalyzer(use_ci=False, use_iqr=False, use_ewma=False)  

class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.with_crop = False
        self.EXPAND_SCALE = None
        self.small_ratio = 12
        self.mid_ratio = 100
        self.large_ratio = 0.5
        self.AOT_INPUT_SIZE = (465, 465)
        self.cnt = 2
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        self.model.cuda(gpu_id)
        self.model.eval()
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.aug_nums = len(cfg.TEST_MULTISCALE)
        if cfg.TEST_FLIP:
            self.aug_nums *= 2
        self.engine = []
        for aug_idx in range(self.aug_nums):
            self.engine.append(build_engine(cfg.MODEL_ENGINE,
                                            phase='eval',
                                            aot_model=self.model,
                                            gpu_id=gpu_id,
                                            short_term_mem_skip=cfg.TEST_SHORT_TERM_MEM_SKIP,
                                            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
                                            ))
            self.engine[-1].eval()
        self.transform = transforms.Compose([
            tr.MultiRestrictSize_(cfg.TEST_MAX_SHORT_EDGE,
                                  cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
                                  cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])
    
    def add_first_frame(self, frame, mask):
        sample = {'current_img': frame, 'current_label': mask}
        sample = self.transform(sample)
        frame = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id)
        mask = sample[0]['current_label'].unsqueeze(0).float().cuda(self.gpu_id)
        mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")
        self.engine[0].add_reference_frame(frame, mask, frame_step=0, obj_nums=int(mask.max()))

    def track(self, image):
        height = image.shape[0]
        width = image.shape[1]
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
            'flip': False
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
            torch.cuda.empty_cache()
        all_preds = []
        for aug_idx in range(self.aug_nums):
            output_height = sample[aug_idx]['meta']['height']
            output_width = sample[aug_idx]['meta']['width']
            image = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            image = image.cuda(self.gpu_id, non_blocking=True)
            self.engine[aug_idx].match_propogate_one_frame(image) #network->engine->aot_engine.py->aotengine
            is_flipped = sample[aug_idx]['meta']['flip']
            pred_logit = self.engine[aug_idx].decode_current_logits((output_height, output_width))
            if is_flipped:
                pred_logit = flip_tensor(pred_logit, 3)
            pred_prob = torch.softmax(pred_logit, dim=1)
            all_preds.append(pred_prob)
            cat_all_preds = torch.cat(all_preds, dim=0)
            pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)
            pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()
            _pred_label = F.interpolate(pred_label,
                                        size=self.engine[aug_idx].input_size_2d,
                                        mode="nearest")
            self.engine[aug_idx].update_memory(_pred_label)
            mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)
        conf = 0

        return mask, conf


# 깊이 예측 모델 로드
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=0)

class HQTrack(object):
    def __init__(self, cfg, config, local_track=False,sam_refine=False,sam_refine_iou=0):
        self.mask_size = None
        self.local_track = local_track
        self.aot_tracker = AOTTracker(cfg, config['gpu_id'])
        self.sam_refine=sam_refine
        if self.sam_refine:
            model_type = 'vit_h'
            sam_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'segment_anything_hq/pretrained_model/sam_hq_vit_h.pth')
            output_mode = "binary_mask"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=torch.device('cuda'))
            self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
            self.mask_prompt = SamPredictor(sam)
        self.sam_refine_iou=sam_refine_iou

    def initialize(self, image, mask):
        self.tracker = self.aot_tracker
        self.tracker.add_first_frame(image, mask)
        self.mask_size = mask.shape
        self.obj_num = int(mask.max())

    def track(self, image):
        depth_result = pipe(Image.fromarray(image))
        depth_map = np.array(depth_result["depth"])

        all_masks = []
        for obj_id in range(self.obj_num):
            obj_mask = (self.tracker.track(image) == obj_id+1)
            obj_depth = np.mean(depth_map[obj_mask])

            # 뎁스 변화량이 작으면 트래킹 스킵
            if analyzer.should_skip_tracking(obj_id, obj_depth, motion_mag, frame_idx):
                print(f"Skipping tracking for object {obj_id}")
                continue

            m, confidence = self.tracker.track(image)
            analyzer.update_depth_history(obj_id, obj_depth, frame_idx)
            all_masks.append(m)

        if not all_masks:
            return None, 0

        return np.max(all_masks, axis=0), confidence

def OnMouse_box(event,x,y,flags,param):
    global x0, y0, img4show, img
    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 =x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        x_temp, y_temp = x, y
        img4show=img.copy()
        cv2.rectangle(img4show, (x0, y0), (x_temp, y_temp), (255, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        x1, y1 = x, y
        cv2.rectangle(img4show, (x0, y0), (x, y), (255, 255, 0), 2)
        img=img4show
        point_box_prompts.append([x0, y0, x1, y1])

def OnMouse_point(event,x,y,flags,param):
    global x0, y0, img4show, img
    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 =x,y
        point_box_prompts.append([x0,y0])
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img4show, (x0, y0), 4, (0, 255, 0), 6)
        img=img4show
        
def count_total_frames(img_files):
    return len(img_files)
    

print("SAM init ...")
model_type = 'vit_h'
sam_checkpoint = os.path.join(base_path, '..', 'segment_anything_hq/pretrained_model/sam_hq_vit_h.pth')
output_mode = "binary_mask"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=torch.device('cuda'))
mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
mask_prompt = SamPredictor(sam)

SAM_prompt = 'Box'
set_Tracker = 'HQTrack'
sam_refine = True
sam_refine_iou = 0.1
muti_object = True
epoch_num=42000
config = {
        'exp_name': 'default',
        'model': 'internT_msdeaotl_v2',
        'pretrain_model_path': 'result/default_InternT_MSDeAOTL_V2/YTB_DAV_VIP/ckpt/save_step_{}.pth'.format(epoch_num),
        'gpu_id': 0,}

if set_Tracker in ['HQTrack']:
    engine_config = importlib.import_module('configs.' + 'ytb_vip_dav_deaot_internT')
cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
cfg.TEST_CKPT_PATH = os.path.join(AOT_PATH, config['pretrain_model_path'])
palette_template = Image.open(os.path.join(os.path.dirname(__file__), '..', 'my_tools/mask_palette.png')).getpalette()
tracker = HQTrack(cfg, config, True, sam_refine, sam_refine_iou)
save_dir = 'C:/Users/ye761/HQTrack/demo/output'
prev_frame = None
last_mask = None
frame_idx = 0
total_frames = count_total_frames(img_files)

for idx, img_file in enumerate(img_files):
    #print(f"Processing image: {img_file}")

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] Failed to read image: {img_file}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ori = img.copy()

    if idx == 0:
        if img_ori is None or img_ori.size == 0:
            print(f"[ERROR] Frame {idx} is empty. Skipping...")
            continue

        img4show = img.copy()
        while (1):
            cv2.namedWindow("demo")
            cv2.imshow('demo', cv2.cvtColor(img4show, cv2.COLOR_RGB2BGR))
            if SAM_prompt == 'Box':
                OnMouse = OnMouse_box
            elif SAM_prompt == 'Point':
                OnMouse = OnMouse_point

            cv2.setMouseCallback('demo', OnMouse)
            k = cv2.waitKey(1)
            if k == ord('r'):
                break

        start_time = time.time()
        print("ROI selected. Processing image...")

        masks_ls = []
        mask_2 = np.zeros_like(img[:,:,0])
        masks_ls.append(mask_2)
        
        for obj_idx, prompt in enumerate(point_box_prompts):
            mask_prompt.set_image(img_ori)
            if SAM_prompt == 'Box':
                masks_, iou_predictions, _ = mask_prompt.predict(box=np.array(prompt).astype(float))
            elif SAM_prompt == 'Point':
                masks_, iou_predictions, _ = mask_prompt.predict(point_labels=np.asarray([1]), point_coords=np.asarray([prompt]))
            select_index = list(iou_predictions).index(max(iou_predictions))
            init_mask = masks_[select_index].astype(np.uint8)
            masks_ls.append(init_mask)
            mask_2 = mask_2 + init_mask * (obj_idx+1)

        masks_ls = np.stack(masks_ls)
        masks_ls_ = masks_ls.sum(0)
        masks_ls_argmax = np.argmax(masks_ls, axis=0)
        rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)
        rs = np.array(rs).astype(np.uint8)

        init_masks = []
        for i in range(len(masks_ls)):
            m_temp = rs.copy()
            m_temp[m_temp!=i+1]=0
            m_temp[m_temp!=0]=1
            init_masks.append(m_temp)

        img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_RGB2BGR)
        for idx, m in enumerate(init_masks):
            img[:, :, 1] += 127.0 * m
            img[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
        im_m = im_m.clip(0, 255).astype(np.uint8)
        cv2.putText(im_m, 'Init', (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
        cv2.imshow('demo', im_m)
        k = cv2.waitKey(1)

        print('init target objects ...')
        tracker.initialize(img_ori, rs)
        obj_num = len(init_masks)
        print('HQTrack running ...')

        # 🔹 초기 프레임에서 객체별 뎁스 저장
        depth_result = pipe(Image.fromarray(img_ori))
        depth_map = np.array(depth_result["depth"])
        for obj_id in range(obj_num):
            analyzer.update_depth_history(obj_id, np.mean(depth_map[rs == (obj_id+1)]), frame_idx)
        last_mask = rs.copy()
        prev_frame = img_ori.copy()
        frame_idx = 1
            
    else:
            depth_result = pipe(Image.fromarray(img_ori))
            depth_map = np.array(depth_result["depth"])

            updated_mask = last_mask.copy()
            skip_flags = []

            for obj_id in range(obj_num):
                obj_mask = (last_mask == (obj_id + 1))
                if not np.any(obj_mask):
                    skip_flags.append(True)
                    continue

                obj_depth = np.mean(depth_map[obj_mask])
                motion_vector = analyzer.compute_optical_flow(prev_frame, img_ori, obj_mask)
                motion_mag = analyzer.compute_motion_magnitude(motion_vector)
                should_skip = analyzer.should_skip_tracking(obj_id, obj_depth, motion_mag, frame_idx, total_frames)
                skip_flags.append(should_skip)

                if not should_skip:
                    m, confidence = tracker.track(img_ori)
                    if m is not None:
                        updated_mask = m.copy()
                        print(f"[TRACK] Frame {idx} - Object {obj_id} tracked")
                    else:
                        print(f"[ERROR] Frame {idx} - Object {obj_id} tracking failed, m is None")
                        continue  
                else:
                    print(f"[SKIP] Frame {idx} - Object {obj_id} skipped, reusing last mask")

                analyzer.update_depth_history(obj_id, obj_depth, frame_idx)
                analyzer.update_motion_history(obj_id, motion_mag)

            # Mask fusion and visualization
            pred_logit = tracker.tracker.engine[0].decode_current_logits((img_ori.shape[0], img_ori.shape[1]))
            fused_tensor = torch.zeros_like(pred_logit[:, :1, :, :])

            for obj_id in range(obj_num):
                if obj_id != 0:
                    continue
                obj_mask = (updated_mask == (obj_id + 1))
                if not np.any(obj_mask):
                    continue

                motion_vector = analyzer.compute_optical_flow(prev_frame, img_ori, obj_mask)
                motion_mag = analyzer.compute_motion_magnitude(motion_vector)

                vx, vy = motion_vector
                depth_region = depth_map[obj_mask]
                if np.any(np.isnan(depth_region)):
                    depth_region = np.nan_to_num(depth_region, nan=np.nanmean(depth_region))
                depth_mean = np.mean(depth_region)
                prev_depth = analyzer.depth_history[obj_id][-1][1] if obj_id in analyzer.depth_history else depth_mean
                z_change = abs(depth_mean - prev_depth)

                depth_tensor = torch.from_numpy(depth_region.astype(np.float32)).cuda()
                motion_tensor = torch.full_like(depth_tensor, motion_mag)

                norm_depth = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-6)
                norm_motion = (motion_tensor - motion_tensor.min()) / (motion_tensor.max() - motion_tensor.min() + 1e-6)

                w_motion, w_depth, dominant = analyzer.compute_fusion_weights(vx, vy, z_change, motion_mag)
                fusion_val = w_depth * norm_depth + w_motion * norm_motion

                temp_tensor = torch.zeros_like(fused_tensor[0, 0])
                temp_tensor[obj_mask] = fusion_val.cuda() if temp_tensor.is_cuda else fusion_val
                fused_tensor[0, 0] += temp_tensor

                print(f"[Fusion Weights] Obj {obj_id} | vx: {vx:.3f}, vy: {vy:.3f}, z: {z_change:.3f} → w_xy: {w_motion:.2f}, w_z: {w_depth:.2f} (dominant: {dominant})")

            pred_logit += fused_tensor

            pred_masks = []
            for i in range(obj_num):
                m_temp = updated_mask.copy()
                m_temp[m_temp != i + 1] = 0
                m_temp[m_temp != 0] = 1
                pred_masks.append(m_temp)

            img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_RGB2BGR)
            for i, mask in enumerate(pred_masks):
                img[:, :, 1] += 127.0 * mask
                img[:, :, 2] += 127.0 * mask
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img = cv2.drawContours(img, contours, -1, cur_colors[i], 2)
            img = img.clip(0, 255).astype(np.uint8)

            save_path = os.path.join(save_dir, os.path.basename(img_file))
            cv2.imwrite(save_path, img)

            last_mask = updated_mask.copy()
            prev_frame = img_ori.copy()
            frame_idx += 1

            if all(skip_flags):
                print(f"[SKIP] Skipping entire frame {idx} due to all objects being stable")
                continue
            
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tracking completed in {elapsed_time:.2f} seconds.")
# ✅ 실험 결과 요약 출력
total_processed = frame_idx
tracked_frames = frame_idx - analyzer.total_skipped if hasattr(analyzer, 'total_skipped') else 'N/A'
skipped_frames = analyzer.total_skipped if hasattr(analyzer, 'total_skipped') else 'N/A'
fps = frame_idx / max(elapsed_time, 1.0)
success_rate = (frame_idx - analyzer.total_skipped) / max(frame_idx, 1)

print(f"\n[수동 실험 실행] CI={analyzer.use_ci}, IQR={analyzer.use_iqr}, EWMA={analyzer.use_ewma}")
print("\n[실험 요약 결과]")
print(f"총 프레임 수: {total_processed}")
print(f"트래킹된 프레임 수: {tracked_frames}")
print(f"스킵된 프레임 수: {skipped_frames}")
print(f"총 수행 시간: {elapsed_time:.2f}초")
print(f"Success Rate (Tracking Precision): {success_rate:.4f}")
print(f"FPS: {fps:.2f}")