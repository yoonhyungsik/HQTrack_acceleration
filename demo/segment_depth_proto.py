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


# 뎁스 변화량 저장 구조  -> 동적 큐로 변환가능? 고정된 크기보다 큐의 크기도 동적으로 바뀌었으면 좋겠음음
depth_history = {}  # {object_id: deque([d1, d2, ..., dN])}
depth_diff_history = []
N = 10  # 최근 프레임 개수
adaptive_threshold = None

class DepthThresholdManager:
    def __init__(self, min_N=5, max_N=20, base_N=10, threshold_std=0.05, removal_threshold=0.02, max_frame_diff=10):
        self.depth_history = {}  # {object_id: deque([(frame_idx, depth), ...])}
        self.min_N = min_N  # 최소 저장 프레임 개수
        self.max_N = max_N  # 최대 저장 프레임 개수
        self.base_N = base_N  # 기본 프레임 개수
        self.threshold_std = threshold_std  # 변동성 기준
        self.removal_threshold = removal_threshold  # 제거 기준 (변화량이 너무 작을 때)  -> depth_diff 같은 값으로 바꾸는 방법 생각하기
        self.max_frame_diff = max_frame_diff  # 오래된 프레임 삭제 기준

    def update_depth(self, obj_id, depth_value, frame_idx):
        """객체별 뎁스 변화량 저장 (동적 길이 조정 + 오래된 프레임 삭제)"""
        if isinstance(obj_id, deque):
            obj_id = obj_id[0][0] 
    
        if obj_id not in self.depth_history:
            self.depth_history[obj_id] = deque(maxlen=self.base_N)

        history = self.depth_history[obj_id]
         # 📌 (1) 초기 deque 상태 출력
        print(f"[DEBUG] Initial history for obj {obj_id}: {list(history)}")

        # 새로운 값 추가
        frame_idx = int(frame_idx) if isinstance(frame_idx, (int, float)) else 0  # 예외 처리
        history.append((frame_idx, depth_value))
        
        # 📌 (2) 값 추가 후 deque 상태 출력
        print(f"[DEBUG] After append for obj {obj_id}: {list(history)}")

        if len(history) < 3:
            return history  # 데이터가 부족하면 유지
    
        # 현재 뎁스값과의 차이 계산
        depth_values = np.array([depth for _, depth in history])
        differences = np.abs(depth_values - depth_value)

        # (1) 변동성 기반 길이 조정
        std_depth = np.std(np.diff(depth_values)) if len(depth_values) > 1 else 0
        if std_depth > self.threshold_std:
            new_N = min(self.max_N, len(history) + 2)  # 변동성이 크면 증가
        else:
            new_N = max(self.min_N, len(history) - 2)  # 변동성이 작으면 감소
    
        self.depth_history[obj_id] = deque(history, maxlen=new_N)  # ✅ 길이 업데이트 반영
        
        # 📌 (3) 길이 조정 후 deque 상태 출력
        print(f"[DEBUG] After resizing for obj {obj_id} (new maxlen={new_N}): {list(self.depth_history[obj_id])}")

        # (2) 변화량이 너무 작은 데이터 삭제 
        if len(history) > self.min_N:
            idx_to_remove = np.argmin(differences)  # 현재값과 가장 차이가 적은 값 선택
            history_list = list(history)
            history_list.pop(idx_to_remove)  # 리스트 변환 후 삭제
            self.depth_history[obj_id] = deque(history_list, maxlen=new_N)  # 다시 deque 변환
            print(f"[REMOVE] Low variance detected, removing index {idx_to_remove} for object {obj_id}")
            
             # 📌 (4) 값 제거 후 deque 상태 출력
            print(f"[DEBUG] After removing low variance data for obj {obj_id}: {list(self.depth_history[obj_id])}")

        # (3) 오래된 프레임 삭제 (frame_idx 기준 필터링 + 최신순 정렬)
        filtered_history = [(f, d) for f, d in self.depth_history[obj_id] if frame_idx - f <= self.max_frame_diff]
        filtered_history.sort(reverse=True, key=lambda x: x[0])  # 최신 프레임 우선 정렬
        self.depth_history[obj_id] = deque(filtered_history[:new_N], maxlen=new_N)  # ✅ 길이 유지
        
         # 📌 (5) 최종 deque 상태 출력
        print(f"[DEBUG] Final history for obj {obj_id}: {list(self.depth_history[obj_id])}")

        return self.depth_history[obj_id]

        
manager = DepthThresholdManager() # 클래스 인스턴스 생성(Queuing)

# 깊이 예측 모델 로드
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=0)

def compute_dynamic_alpha(depth_diff_history, base_k=0.1):
    """깊이 변화량(표준편차)에 따라 α 값을 동적으로 조정"""
    depth_diff_history = np.array(depth_diff_history)
    depth_diff_history = depth_diff_history[~np.isnan(depth_diff_history)]  # NaN 제거

    if len(depth_diff_history) < 10:
        return 0.9  # 최소 샘플 개수가 부족하면 기본값 유지

    std_depth = np.std(depth_diff_history)
    
    # k 값을 데이터의 중앙값(median) 기반으로 동적 조정
    median_value = np.median(depth_diff_history)
    k = base_k * median_value if median_value > 0 else base_k

    # α 계산 (최대값 0.95로 제한)
    alpha = 0.95 - (std_depth / (std_depth + k))

    return np.clip(alpha, 0.7, 0.95)  # 값 범위 제한


def compute_dynamic_confidence(depth_diff_history):
    """데이터에서 이상치 비율을 기반으로 confidence 값을 자동 조정"""
    depth_diff_history = np.array(depth_diff_history)
    depth_diff_history = depth_diff_history[~np.isnan(depth_diff_history)]  # NaN 제거

    if len(depth_diff_history) < 5:
        return 0.95  # 기본값 반환

    Q1, Q3 = np.percentile(depth_diff_history, [25, 75])
    IQR = Q3 - Q1
    outliers_iqr = np.sum((depth_diff_history < (Q1 - 1.5 * IQR)) | (depth_diff_history > (Q3 + 1.5 * IQR)))

     # MAD 방식 이상치 검출
    median = np.median(depth_diff_history)
    MAD = np.median(np.abs(depth_diff_history - median)) * 1.4826
    outliers_mad = np.sum((depth_diff_history < (median - 3 * MAD)) | (depth_diff_history > (median + 3 * MAD)))

    # 이상치 개수 평균화
    outliers = (outliers_iqr + outliers_mad) / 2

    # 이상치 비율 계산
    outlier_ratio = outliers / len(depth_diff_history)

    # 지수 감쇄 방식 적용 (λ=3)
    lambda_factor = 3
    confidence = 0.95 * np.exp(-lambda_factor * outlier_ratio)

    return np.clip(confidence, 0.9, 0.95)  # 신뢰구간 범위 제한

def compute_percentile_threshold(depth_diff_history, k=1.5):
    """퍼센타일 기반 임계값 설정 (Percentile-Based Thresholding)"""
    depth_diff_history = np.array(depth_diff_history)
    depth_diff_history = depth_diff_history[~np.isnan(depth_diff_history)]

    if len(depth_diff_history) < 5:
        return np.mean(depth_diff_history) if len(depth_diff_history) > 0 else 0

    Q1 = np.percentile(depth_diff_history, 25)
    Q3 = np.percentile(depth_diff_history, 75)
    IQR = Q3 - Q1

    return Q3 + k * IQR

def compute_variance_adaptive_threshold(depth_diff_history, k_base=1.5):
    """변동성 기반 임계값 설정 (Variance-Adaptive Thresholding)"""
    depth_diff_history = np.array(depth_diff_history)
    depth_diff_history = depth_diff_history[~np.isnan(depth_diff_history)]

    if len(depth_diff_history) < 5:
        return np.mean(depth_diff_history) if len(depth_diff_history) > 0 else 0

    mean_diff = np.mean(depth_diff_history)
    std_diff = np.std(depth_diff_history)

    # 변동성 기반으로 k 값을 조정 (std가 클수록 가중치 증가)
    k_dynamic = k_base * (1 + (std_diff / (mean_diff + 1e-6)))

    return mean_diff + k_dynamic * std_diff

class AdaptiveThreshold:
    def __init__(self):
        """데이터 적응형 신뢰 구간 + 베이지안 업데이트 기반 스레쉬홀드"""
        self.mean = None
        self.std = None
    
    def compute_confidence_interval_threshold(self, depth_diff_history):
        """고도화된 적응형 신뢰 구간 기반 임계값 계산"""
        confidence = compute_dynamic_confidence(depth_diff_history)
    
        depth_diff_history = np.array(depth_diff_history)
        depth_diff_history = depth_diff_history[~np.isnan(depth_diff_history)]  # NaN 제거

        if len(depth_diff_history) < 5:
            return np.mean(depth_diff_history) + np.std(depth_diff_history) * 0.1  # 기본값 반환

        # 이상치 제거 (IQR 방법)
        q1, q3 = np.percentile(depth_diff_history, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_depth_diff = depth_diff_history[
        (depth_diff_history >= lower_bound) & (depth_diff_history <= upper_bound)
    ]

        if len(filtered_depth_diff) < 5:
            filtered_depth_diff = depth_diff_history  # 이상치 제거 후 샘플 부족하면 원본 유지

        mean_diff = np.mean(filtered_depth_diff)
        std_diff = np.std(filtered_depth_diff)

        # 지수 가중 이동 평균(EWMA) 적용
        alpha = 0.2
        ewma_mean = filtered_depth_diff[0]
        for val in filtered_depth_diff[1:]:
            ewma_mean = alpha * val + (1 - alpha) * ewma_mean

        # 신뢰수준에 따른 Z-score 계산
        z_score = norm.ppf(1 - (1 - confidence) / 2)

        # 동적 스케일링 팩터 적용
        scaling_factor = min(1.0, 0.1 + std_diff / np.max(filtered_depth_diff))

        return ewma_mean + z_score * std_diff * scaling_factor    
    
    def update_bayesian(self, new_value, depth_diff_history):
        """베이지안 업데이트 방식으로 적응형 a 적용"""
        alpha = compute_dynamic_alpha(depth_diff_history)  # α 값 자동 조정

        if np.isnan(new_value):
            return self.mean if self.mean is not None else 5.0 # 기존 평균 유지
        
        #초기값 설정
        if self.mean is None:
            self.mean = new_value
            self.std = 1e-6 # 작은 값으로 초기화하여 안정성 확보보
        else:
            # 베이지안 업데이트 방식 적용용
            self.mean = alpha * self.mean + (1-alpha) * new_value
            self.std = np.sqrt(alpha * self.std**2 + (1-alpha) * (new_value - self.mean)**2)

        # 표준편차 기반 동적 스케일링 팩터
        scaling_factor = 1 / (1+self.std)
        
        return self.mean+ 1.5 * self.std * scaling_factor

    def compute_final_threshold(self, obj_id, new_value, frame_idx):
        """신뢰 구간 방식, 베이지안 업데이트, variance adaptive, percentile 방식을 조합하여 동적 임계값 결정"""

        # ✅ 동적 큐를 활용하여 최근 depth 변화량 업데이트
        depth_diff_history = manager.update_depth(obj_id, new_value, frame_idx)

        if len(depth_diff_history) < 3:
            return obj_depth  # 데이터 부족 시 현재 뎁스값값 반환

        # ✅ 개별 임계값 계산
        std_dev = np.std([depth for _, depth in depth_diff_history])  # 표준편차 계산
        variance_threshold = compute_variance_adaptive_threshold(std_dev)
        percentile_threshold = compute_percentile_threshold([depth for _, depth in depth_diff_history])
        ci_threshold = self.compute_confidence_interval_threshold([depth for _, depth in depth_diff_history])
        bayesian_threshold = self.update_bayesian(new_value, [depth for _, depth in depth_diff_history])

        # ✅ Variance와 Percentile을 바탕으로 동적 가중치 계산
        variance_weight = 0.6 * np.clip(1 - std_dev, 0, 1)  # 표준편차 작을수록 variance 반영
        percentile_weight = 1 - variance_weight  # 나머지 가중치는 Percentile에 부여

        # ✅ CI와 Bayesian 방식의 가중치 계산
        ci_weight = 0.5 + (0.5 * np.clip(std_dev, 0, 1))  # 표준편차 클수록 CI 중요
        bayesian_weight = 1 - ci_weight

        # ✅ 임계값 상한 조정 (변동성이 크면 여유있게 조정)
        max_threshold = obj_depth * (1.1 if std_dev > 0.2 else 1.0)

        # ✅ 최종 임계값 계산 (가중치 조합)
        final_threshold = (
        variance_weight * variance_threshold +
        percentile_weight * percentile_threshold + 
        ci_weight * ci_threshold +
        bayesian_weight * bayesian_threshold
    )

        # ✅ 최댓값 제한 적용
        final_threshold = min(final_threshold, max_threshold)

        #print(f"Variance Threshold: {variance_threshold:.4f}, Percentile Threshold: {percentile_threshold:.4f}, CI Threshold: {ci_threshold:.4f}, Bayesian Threshold: {bayesian_threshold:.4f}, Final Threshold: {final_threshold:.4f}")
    
        return final_threshold

adaptive_threshold = AdaptiveThreshold()  # 클래스 인스턴스 생성(thresholding)

def should_skip_tracking(obj_id, obj_depth, frame_idx):
    # NaN 검사 추가
    if np.isnan(obj_depth):
        return False  # NaN인 경우 추적 계속
    
    if obj_id not in depth_history or len(depth_history[obj_id]) < 3:
        return False
    
    # ✅ 동적 큐 활용하여 `depth_diff_history` 업데이트
    depth_diff_history = manager.update_depth(obj_id, obj_depth, frame_idx)
    final_threshold = adaptive_threshold.compute_final_threshold(depth_diff_history, obj_depth, frame_idx)
    
    # ✅ 최신 Depth 가져오기
    last_depth = depth_diff_history[-1][1] if depth_diff_history else obj_depth  # (frame_idx, depth) 튜플에서 depth 값 추출
    # NaN 검사 추가
    if np.isnan(last_depth):
        return False
    
    depth_diff = abs(obj_depth - last_depth)
    
    print(f"Object {obj_id} | Depth: {obj_depth:.4f} | Last Depth: {last_depth:.4f} | Depth Diff: {depth_diff:.4f} | Threshold: {final_threshold:.4f}")
    
    return obj_depth < final_threshold  #obj_depth가 final_threshold보다 작으면 True, 아니면 False리턴


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
    def __init__(self, cfg, config, local_track=False, sam_refine=False, sam_refine_iou=0):
        self.mask_size = None
        self.local_track = local_track
        self.aot_tracker = AOTTracker(cfg, config['gpu_id'])
        self.sam_refine = sam_refine
        if self.sam_refine:
            model_type = 'vit_h'
            sam_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'segment_anything_hq/pretrained_model/sam_hq_vit_h.pth')
            output_mode = "binary_mask"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=torch.device('cuda'))
            self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
            self.mask_prompt = SamPredictor(sam)
        self.sam_refine_iou = sam_refine_iou

        self.frame_idx = 0

    def initialize(self, image, mask):
        self.tracker = self.aot_tracker
        self.tracker.add_first_frame(image, mask)
        self.mask_size = mask.shape
        self.obj_num = int(mask.max())
    
    def track(self, image):
        # Depth 맵 생성
        depth_result = pipe(Image.fromarray(image))
        depth_map = np.array(depth_result["depth"])

        all_masks = []
        last_masks = getattr(self, 'last_masks', [None] * self.obj_num)  # 이전 마스크 저장 속성 추가
        confidence = 0  # 기본 confidence 값 설정
    
        for obj_id in range(self.obj_num):
            # 깊이 변화량이 작으면 트래킹 스킵하고 이전 마스크 사용
            obj_depth = np.mean(depth_map[last_masks[obj_id] if last_masks[obj_id] is not None else np.zeros_like(depth_map, dtype=bool)])

            # should_skip_tracking() 호출 시 frame_idx 전달
            if should_skip_tracking(obj_id, obj_depth, self.frame_idx):
                print(f"Skipping tracking for object {obj_id}")   # obj_depth가 final_threshold보다 작으면 True 반환
                if last_masks[obj_id] is not None:
                    # 이전 프레임의 마스크를 사용하여 객체 ID로 채운 마스크 생성
                    obj_mask = np.zeros_like(last_masks[obj_id], dtype=np.uint8)
                    obj_mask[last_masks[obj_id]] = obj_id + 1
                    all_masks.append(obj_mask)
                    # 이전 마스크에서 깊이값 업데이트
                    update_depth_history(obj_id, obj_depth)
                continue

            # 현재 프레임에 대한 마스크 트래킹
            m, curr_confidence = self.tracker.track(image)
            confidence = max(confidence, curr_confidence)  # 최대 confidence 값 유지
            obj_mask = (m == obj_id + 1)  # 객체 ID 맵에서 특정 ID만 선택
            last_masks[obj_id] = obj_mask  # 현재 마스크 저장
        
            obj_depth = np.mean(depth_map[obj_mask])
            update_depth_history(obj_id, obj_depth)
            all_masks.append(m)
    
        # 클래스 속성으로 마스크 저장하여 다음 호출에서 사용
        self.last_masks = last_masks

        # 프레임 인덱스 증가
        self.frame_idx += 1

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
    print("Image load success")

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
        print("Initial mask shape:", rs.shape)
        print("Unique values in initial mask:", np.unique(rs))
        
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

        depth_result = pipe(Image.fromarray(img_ori))
        depth_map = np.array(depth_result["depth"])
        segmented_depth_map = np.zeros_like(depth_map)
        for obj_id in range(obj_num):
            segmented_mask = (rs == (obj_id+1))
            if np.any(segmented_mask):
                obj_depth = np.mean(depth_map[segmented_mask])
                update_depth_history(obj_id, obj_depth)
                segmented_depth_map[segmented_mask] = depth_map[segmented_mask]
    
    else:
        depth_result = pipe(Image.fromarray(img_ori))
        depth_map = np.array(depth_result["depth"])
        m, confidence = tracker.track(img_ori)
        print('Running frame ', idx)
        
        pred_masks = []
        depth_values = {}
        segmented_depth_map = np.zeros_like(depth_map)
        for obj_id in range(obj_num):
            segmented_mask = (m == obj_id+1)
            if np.any(segmented_mask):
                obj_depth = np.mean(depth_map[segmented_mask])
                depth_values[obj_id] = obj_depth
                segmented_depth_map[segmented_mask] = depth_map[segmented_mask]
            else:
                depth_values[obj_id] = 0
        
        print("\n🔹 Object Depth Values 🔹")
        for obj_id, depth in depth_values.items():
            print(f"Object {obj_id}: Depth = {depth:.4f}")
        
        for obj_id in range(obj_num):
            update_depth_history(obj_id, depth_values[obj_id])
        
        for i in range(obj_num):
            m_temp = m.copy()
            m_temp[m_temp != i + 1] = 0
            m_temp[m_temp != 0] = 1
            pred_masks.append(m_temp)
        
        img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_RGB2BGR)
        for idx, m in enumerate(pred_masks):
            img[:, :, 1] += 127.0 * m
            img[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
        im_m = im_m.clip(0, 255).astype(np.uint8)
        save_path = os.path.join(save_dir, img_file.split('/')[-1])
        cv2.imwrite(save_path, im_m)
        
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tracking completed in {elapsed_time:.2f} seconds.")
