# HQTrack Acceleration with Depth-Anything-V2

이 프로젝트는 HQTrack을 기반으로 하여 **모션-깊이 기반 프레임 스킵 최적화**를 적용한 고정밀 객체 추적 시스템입니다.  
`Depth-Anything-V2`를 사용해 고정밀 깊이 정보를 추출하고, 이를 활용해 추적 수행 여부를 동적으로 제어합니다.

---

## 📁 프로젝트 구성

- `networks/`: HQTrack 백엔드 및 모델 정의
- `dataloaders/`: 영상 데이터 처리 로직
- `my_tools/`: 커스텀 추적/분석 유틸리티
- `demo/`: 데모 실행용 코드
- `configs/`: 추적 설정 파일
- `utils/`: 공용 유틸리티 함수
- `pretrain_models/`: (GitHub에는 포함되지 않음, 직접 다운로드 필요)

---

## 🔧 설치 방법

### 1. 이 리포지토리 클론

```bash
git clone https://github.com/yoonhyungsik/HQTrack_acceleration.git
cd HQTrack_acceleration
```

### 2. 서브모듈 또는 외부 리포지토리 클론

이 프로젝트는 다음 두 리포지토리를 **별도로 클론**해야 합니다:

#### ✅ Depth-Anything-V2 (깊이 추정)

```bash
git submodule add https://github.com/isl-org/Depth-Anything-V2.git
```

#### ✅ HQTrack (백엔드 추적기)

```bash
git submodule add https://github.com/hkchengrex/HQTrack.git
```

> 또는 수동으로 아래 경로에 클론:
> - `Depth-Anything-V2/`
> - `segment_anything_hq/` 또는 `HQTrack/`

---

### 3. 의존성 설치

Python 환경을 구성한 후 필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

> ⚠️ `requirements.txt`가 없는 경우 다음 패키지들을 수동 설치해야 합니다:
> - `torch`, `opencv-python`, `tqdm`, `einops`, `matplotlib` 등
> - HQTrack과 depth-anything requirements 설치하시면 됩니다.


---

## 📌 주의사항

- 본 프로젝트는 일부 대용량 모델 파일(`.pt`, `.onnx`)이 GitHub에 포함되어 있지 않습니다. `pretrain_models/` 디렉토리에 수동으로 추가해야 합니다.
- 일부 demo 영상은 `.gitignore`로 제외되어 있습니다. 별도로 제공되는 영상 데이터를 `/demo/` 디렉토리에 넣어주세요.

---

## 📄 참고 논문 및 프로젝트

- [HQTrack (ECCV 2022)](https://github.com/hkchengrex/HQTrack)
- [Depth-Anything-V2 (2024)](https://github.com/isl-org/Depth-Anything-V2)

---

## 💻 라이센스

본 프로젝트는 연구 및 학술 목적에 한해 사용할 수 있습니다.  
각 서브모듈 및 외부 프로젝트는 해당 리포지토리의 라이센스를 따릅니다.
