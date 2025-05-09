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
        #print(f"[DEBUG] Initial history for obj {obj_id}: {list(history)}")

        # 새로운 값 추가
        frame_idx = int(frame_idx) if isinstance(frame_idx, (int, float)) else 0  # 예외 처리
        history.append((frame_idx, depth_value))
        
        # 📌 (2) 값 추가 후 deque 상태 출력
        #print(f"[DEBUG] After append for obj {obj_id}: {list(history)}")

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
        #print(f"[DEBUG] After resizing for obj {obj_id} (new maxlen={new_N}): {list(self.depth_history[obj_id])}")

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
        #print(f"[DEBUG] Final history for obj {obj_id}: {list(self.depth_history[obj_id])}")

        return self.depth_history[obj_id]
        
manager = DepthThresholdManager() # 클래스 인스턴스 생성(Queuing)

############################################################################################################## Thresholding  ##########################################################################

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

adaptive_threshold = AdaptiveThreshold()

####################################################################################################################### Motion & Depth ################################################################################################################

class AdaptiveMotionDepthTracker:
    def __init__(
        self, 
        depth_threshold_ratio=0.05,      # 깊이 변화 임계값 비율
        motion_threshold=0.3,            # 모션 변화량 임계값
        tracking_sensitivity=0.7,        # 트래킹 민감도
        adaptation_rate=0.3,             # 가중치 적응 속도 (0~1)
        min_weight=0.2,                  # 최소 가중치 (0~1)
        max_weight=0.8                   # 최대 가중치 (0~1)
    ):
        self.depth_history = {}          # 깊이 히스토리: {obj_id: [(frame_idx, depth_value), ...]}
        self.motion_history = {}         # 모션 히스토리: {obj_id: [(frame_idx, motion_vector), ...]}
        self.depth_threshold_ratio = depth_threshold_ratio
        self.motion_threshold = motion_threshold
        self.tracking_sensitivity = tracking_sensitivity
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_history = {}         # 객체별 가중치 히스토리: {obj_id: current_weight}
        self.depth_threshold_manager = DepthThresholdManager()
        self.adaptive_threshold = AdaptiveThreshold()

    def compute_motion_magnitude(self, motion_vector):
        """모션 벡터의 크기 계산"""
        return np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)

    def update_object_history(self, obj_id, depth_value, motion_vector, frame_idx):
        """객체의 깊이와 모션 히스토리 업데이트"""
        # 깊이 히스토리 초기화 및 업데이트
        if obj_id not in self.depth_history:
            self.depth_history[obj_id] = []
        self.depth_history[obj_id].append((frame_idx, depth_value))
        self.depth_history[obj_id] = self.depth_history[obj_id][-10:]  # 최근 10개만 유지
        
        # 모션 히스토리 초기화 및 업데이트
        if obj_id not in self.motion_history:
            self.motion_history[obj_id] = []
        self.motion_history[obj_id].append((frame_idx, motion_vector))
        self.motion_history[obj_id] = self.motion_history[obj_id][-10:]  # 최근 10개만 유지
        
        # 가중치 히스토리 초기화
        if obj_id not in self.weight_history:
            self.weight_history[obj_id] = 0.5  # 초기 가중치는 균등하게 설정

    def calculate_depth_change(self, obj_id):
        """깊이 변화량 계산"""
        if len(self.depth_history[obj_id]) < 2:
            return 0.0
        
        depth_values = [d for _, d in self.depth_history[obj_id]]
        # 최근 5개 프레임 또는 모든 프레임의 깊이 값을 사용하여 변화량 계산
        recent_depth = depth_values[-min(5, len(depth_values)):]
        if len(recent_depth) < 2:
            return 0.0
            
        # 표준편차 계산 (변화량의 지표)
        depth_std = np.std(recent_depth)
        # 최대-최소 변화량 계산
        depth_range = np.max(recent_depth) - np.min(recent_depth)
        # 평균 대비 변화 비율 계산
        depth_mean = np.mean(recent_depth)
        depth_change_ratio = (depth_range / depth_mean) if depth_mean > 0 else 0
        
        # 변화량 점수 계산 (표준편차와 변화 비율의 조합)
        depth_change_score = depth_std * 0.5 + depth_change_ratio * 0.5
        
        return depth_change_score

    def calculate_motion_change(self, obj_id):
        """모션 변화량 계산"""
        if len(self.motion_history[obj_id]) < 2:
            return 0.0
        
        # 최근 5개 프레임 또는 모든 프레임의 모션 벡터 사용
        recent_motion = self.motion_history[obj_id][-min(5, len(self.motion_history[obj_id])):]
        
        # 각 모션 벡터의 크기 계산
        motion_magnitudes = [self.compute_motion_magnitude(mv) for _, mv in recent_motion]
        
        if len(motion_magnitudes) < 2:
            return 0.0
            
        # 모션 변화량 계산 (표준편차 사용)
        motion_std = np.std(motion_magnitudes)
        # 최대 모션 크기 계산
        max_motion = np.max(motion_magnitudes)
        
        # 변화량 점수 계산 (표준편차와 최대 모션의 조합)
        motion_change_score = motion_std * 0.3 + max_motion * 0.7
        
        return motion_change_score

    def update_adaptive_weight(self, obj_id):
        """뎁스와 모션 변화량에 기반한 적응형 가중치 업데이트"""
        depth_change = self.calculate_depth_change(obj_id)
        motion_change = self.calculate_motion_change(obj_id)
        
        # 변화량이 너무 작으면 가중치 유지
        if depth_change < 0.01 and motion_change < 0.01:
            return self.weight_history[obj_id]
        
        # 뎁스와 모션 변화량의 상대적 비율 계산
        total_change = depth_change + motion_change
        
        if total_change > 0:
            # 뎁스 변화 비율 계산 (a 값)
            depth_ratio = depth_change / total_change
            # 현재 가중치와 새로운 비율을 혼합하여 가중치 업데이트
            new_weight = (1 - self.adaptation_rate) * self.weight_history[obj_id] + self.adaptation_rate * depth_ratio
            # 가중치 범위 제한
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            self.weight_history[obj_id] = new_weight
            
            print(f"Object {obj_id} | Depth Change: {depth_change:.4f} | Motion Change: {motion_change:.4f} | Weight: {new_weight:.4f}")
        
        return self.weight_history[obj_id]

    def should_track(self, obj_id, depth_value, motion_vector, frame_idx):
        """객체 추적 여부 결정"""
        # 히스토리 업데이트
        self.update_object_history(obj_id, depth_value, motion_vector, frame_idx)
        
        # 객체 가중치 업데이트 (a 값)
        weight = self.update_adaptive_weight(obj_id)
        
        # 뎁스 기반 스레스홀드 계산
        depth_queue = self.depth_threshold_manager.update_depth(obj_id, depth_value, frame_idx)
        depth_threshold = self.adaptive_threshold.compute_final_threshold(obj_id, depth_value, frame_idx)
        
        # 모션 크기 계산
        motion_magnitude = self.compute_motion_magnitude(motion_vector)
        
        # 뎁스 변화량 계산
        depth_values = [d for _, d in depth_queue]
        last_depth = depth_values[-1] if depth_values else depth_value
        depth_diff = abs(depth_value - last_depth)
        
        # 뎁스 기반 결정 (뎁스 변화량이 임계값보다 작으면 추적 중단)
        depth_decision = depth_diff < depth_threshold
        
        # 모션 기반 결정 (모션 크기가 임계값보다 작으면 추적 중단)
        motion_decision = motion_magnitude < self.motion_threshold
        
        # 가중치 적용된 최종 결정 (가중합 계산)
        # 가중치(weight)는 뎁스의 영향력을 의미하고, (1-weight)는 모션의 영향력을 의미함
        # 두 결정을 가중합으로 계산하여 최종 결정
        weighted_decision = weight * depth_decision + (1 - weight) * motion_decision
        
        # 최종 결정 값이 0.5 이상이면 추적 중단 (True는 추적 중단, False는 추적 계속)
        track_decision = weighted_decision >= 0.5
        
        print(f"Object {obj_id} | Depth Decision: {depth_decision} | Motion Decision: {motion_decision} | Weighted: {weighted_decision:.4f} | Track: {not track_decision}")
        
        return not track_decision  # 추적 여부 반환 (True: 추적, False: 추적 중단)

    def should_skip_tracking(self, obj_id, depth_value, motion_vector, frame_idx):
        """추적 건너뛰기 여부 결정 (should_track과 반대 의미)"""
        return not self.should_track(obj_id, depth_value, motion_vector, frame_idx)

    def adjust_tracking_parameters(self, depth_threshold_ratio=None, motion_threshold=None, 
                                 tracking_sensitivity=None, adaptation_rate=None, 
                                 min_weight=None, max_weight=None):
        """추적 파라미터 조정"""
        if depth_threshold_ratio is not None:
            self.depth_threshold_ratio = depth_threshold_ratio
        if motion_threshold is not None:
            self.motion_threshold = motion_threshold
        if tracking_sensitivity is not None:
            self.tracking_sensitivity = tracking_sensitivity
        if adaptation_rate is not None:
            self.adaptation_rate = adaptation_rate
        if min_weight is not None:
            self.min_weight = min_weight
        if max_weight is not None:
            self.max_weight = max_weight



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

