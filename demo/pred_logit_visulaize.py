import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# PT 파일 불러오기
pt_file_path = r"C:\Users\ye761\HQTrack\demo\p_36_pred_logit.path\pred_logit_frame1_aug0.pt"
data = torch.load(pt_file_path)

# 데이터가 4D 텐서일 경우, 채널 0과 1만 시각화
if isinstance(data, torch.Tensor):
    channels = data.shape[1]  # 채널 수
    # 최대 5개씩 한 행에 배치, 채널 0과 1만 표시
    rows = 1  # 두 개의 채널만 시각화
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6))  # 2개의 서브플롯 설정
    axes = axes.flatten()  # 2D 배열을 1D 배열로 변환

    # 채널 0과 1에 대해서만 히스토그램 그리기
    for c in range(2):  # 채널 0과 1만 처리
        tensor_data = data[0, c].cpu().numpy()  # 첫 번째 배치, c번째 채널 선택
        
        # 히스토그램 표시
        axes[c].hist(tensor_data.flatten(), bins=50, color='gray', edgecolor='black')
        axes[c].set_title(f'Histogram of Channel {c}')
        axes[c].set_xlim([tensor_data.min(), tensor_data.max()])  # 값 범위에 맞게 x축 설정

    plt.tight_layout()  # 서브플롯 간격 자동 조정
    plt.show()
else:
    print('Data is not a tensor')
