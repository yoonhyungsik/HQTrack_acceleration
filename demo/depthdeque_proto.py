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

base_path = os.path.dirname(os.path.abspath(__file__))
demo_video = 'p_34'
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


# 뎁스 변화량 저장 구조
depth_history = {}  # {object_id: deque([d1, d2, ..., dN])}
depth_diff_history = []
N = 10  # 최근 프레임 개수
adaptive_threshold = None

# 깊이 예측 모델 로드
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=0)

def compute_dynamic_alpha(depth_diff_history, k=0.1):
    """깊이 변화량에 따라 a 값을 자동으로 조정"""
    depth_diff_history = np.array(depth_diff_history)
    depth_diff_history = depth_diff_history[~np.isnan(depth_diff_history)]  # NaN 제거

    if len(depth_diff_history) < 5:
        return 0.9  # 기본값 반환

    std_depth = np.std(depth_diff_history)
    alpha = 1 - (std_depth / (std_depth + k))

    return np.clip(alpha, 0.7, 0.95)  # 안정성을 위해 값 범위 제한


def compute_dynamic_confidence(depth_diff_history):
    """데이터에서 이상치 비율을 기반으로 confidence 값을 자동 조정"""
    depth_diff_history = np.array(depth_diff_history)
    depth_diff_history = depth_diff_history[~np.isnan(depth_diff_history)]  # NaN 제거

    if len(depth_diff_history) < 5:
        return 0.95  # 기본값 반환

    Q1 = np.percentile(depth_diff_history, 25)
    Q3 = np.percentile(depth_diff_history, 75)
    IQR = Q3 - Q1
    outliers = np.sum((depth_diff_history < (Q1 - 1.5 * IQR)) | (depth_diff_history > (Q3 + 1.5 * IQR)))

    confidence = 0.95 - (outliers / len(depth_diff_history)) * 0.05
    return np.clip(confidence, 0.9, 0.95)  # 신뢰 구간 범위 제한


class AdaptiveThreshold:
    def __init__(self):
        """데이터 적응형 신뢰 구간 + 베이지안 업데이트 기반 스레쉬홀드"""
        self.mean = 0.1
        self.std = 0.05

    def compute_confidence_interval_threshold(self, depth_diff_history):
        """적응형 신뢰 구간 기반 임계값 계산"""
        confidence = compute_dynamic_confidence(depth_diff_history)  # 신뢰 구간 자동 조정
        depth_diff_history = np.array(depth_diff_history)
        depth_diff_history = depth_diff_history[~np.isnan(depth_diff_history)]

        if len(depth_diff_history) < 5:
            return 0

        mean_diff = np.mean(depth_diff_history)
        std_diff = np.std(depth_diff_history)

        z_score = 1.96 if confidence == 0.95 else 1.645  # 90% or 95% CI
        return mean_diff + z_score * std_diff

    def update_bayesian(self, new_value, depth_diff_history):
        """베이지안 업데이트 방식으로 적응형 a 적용"""
        alpha = compute_dynamic_alpha(depth_diff_history)  # α 값 자동 조정

        if np.isnan(new_value):
            return self.mean + 1.5 * self.std

        self.mean = alpha * self.mean + (1 - alpha) * new_value
        self.std = alpha * self.std + (1 - alpha) * abs(new_value - self.mean)

        return self.mean + 1.5 * self.std

    def compute_final_threshold(self, depth_diff_history, new_value):
        """신뢰 구간 방식과 베이지안 업데이트 방식 중 더 안정적인 값 선택"""
        ci_threshold = self.compute_confidence_interval_threshold(depth_diff_history)
        bayesian_threshold = self.update_bayesian(new_value, depth_diff_history)
        
        return min(ci_threshold, bayesian_threshold)
    
adaptive_threshold = AdaptiveThreshold()  # 클래스 인스턴스 생성

def should_skip_tracking(obj_id, obj_depth):
    """적응형 스레쉬홀드를 활용하여 트래킹 여부 결정"""
    if obj_id not in depth_history or len(depth_history[obj_id]) < 3:  # 최소 3개 이상의 데이터가 있어야 함
        return False  

    depth_diff_history = depth_history[obj_id]
    final_threshold = adaptive_threshold.compute_final_threshold(depth_diff_history, obj_depth)

    last_depth = depth_history[obj_id][-1] if depth_history[obj_id] else obj_depth
    depth_diff = abs(obj_depth - last_depth)

    print(f"Object {obj_id} | Depth: {obj_depth:.4f} | Last Depth: {last_depth:.4f} | Depth Diff: {depth_diff:.4f} | Threshold: {final_threshold:.4f}")

    return depth_diff < final_threshold  # 변화량이 threshold보다 작으면 스킵


def update_depth_history(obj_id, depth_value):
    """객체별 뎁스 기록 갱신"""
    if np.isnan(depth_value) or depth_value == 0:
        return  # NaN 값은 저장하지 않음

    if obj_id not in depth_history:
        depth_history[obj_id] = deque(maxlen=10)  # 최근 10개 값 저장

    depth_history[obj_id].append(depth_value)


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
            if should_skip_tracking(obj_id, obj_depth):
                print(f"Skipping tracking for object {obj_id}")
                continue

            m, confidence = self.tracker.track(image)
            update_depth_history(obj_id, obj_depth)
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
tracker = HQTrack(cfg, config, True, sam_refine,sam_refine_iou)
save_dir = 'C:/Users/ye761/HQTrack/demo/output'

for idx, img_file in enumerate(img_files):
    print(f"Processing image: {img_file}")

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error loading image: {img_file}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ori = img.copy()

    if idx == 0:
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
            update_depth_history(obj_id, np.mean(depth_map[rs == (obj_id+1)]))

    else:
        # 🔹 깊이 맵 추정
        depth_result = pipe(Image.fromarray(img_ori))
        depth_map = np.array(depth_result["depth"])
        depth_map_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).float().cuda()

        # 🔹 뎁스 변화량 확인 후 트래킹 수행
        skip_tracking = False
        for obj_id in range(obj_num):
            obj_depth = np.mean(depth_map[tracker.track(img_ori) == obj_id+1])
            if should_skip_tracking(obj_id, obj_depth):
                print(f"Skipping tracking for object {obj_id} in frame {idx}")
                skip_tracking = True
                break  # 하나라도 변화량이 작으면 프레임 전체를 스킵

        if all(should_skip_tracking(obj_id, np.mean(depth_map[tracker.track(img_ori) == obj_id+1])) for obj_id in range(obj_num)):
            print(f"Skipping entire frame {idx} due to low depth variation.")
            continue

        # 🔹 트래킹 수행
        m, confidence = tracker.track(img_ori)
        print(f'Running frame {idx}')

        pred_masks = []
        for i in range(obj_num):
            m_temp = m.copy()
            m_temp[m_temp != i + 1] = 0
            m_temp[m_temp != 0] = 1
            pred_masks.append(m_temp)

        # 🔹 깊이 맵을 pred_logit에 추가
        pred_logit = tracker.tracker.engine[0].decode_current_logits((img_ori.shape[0], img_ori.shape[1]))
        pred_logit = pred_logit + depth_map_tensor

        img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_RGB2BGR)
        for idx, m in enumerate(pred_masks):
            img[:, :, 1] += 127.0 * m
            img[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
        im_m = im_m.clip(0, 255).astype(np.uint8)

        save_path = os.path.join(save_dir, img_file.split('/')[-1])
        cv2.imwrite(save_path, im_m)

        # 🔹 새로운 프레임의 뎁스 정보 업데이트
        for obj_id in range(obj_num):
            obj_depth = np.mean(depth_map[m == (obj_id+1)])
            update_depth_history(obj_id, obj_depth)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tracking completed in {elapsed_time:.2f} seconds.")

