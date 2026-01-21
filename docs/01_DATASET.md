# 01. 데이터셋 설명서

## 개요

**데이터셋명**: 대한민국 배경영상 상세묘사 데이터  
**출처**: AI-Hub (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71849)  
**구축 목적**: 대한민국 자연, 도시, 건축물 등 다양한 배경영상에 대한 한국어 캡셔닝 데이터 제공

---

## 1. 데이터셋 구성

### 1.1 전체 규모

| 항목 | 수량 |
|------|------|
| 총 클립 수 | 10,701개 |
| 총 캡션 수 | 10,701개 (클립당 1개) |
| 영상 형식 | MP4 |
| 라벨 형식 | JSON |

### 1.2 데이터 분할 (aihub_splitted)

```
aihub_splitted/
├── train/
│   ├── videos/     # 학습용 비디오 (~8,561개, 80%)
│   └── labels/     # 학습용 라벨 (JSON)
└── val/
    ├── videos/     # 검증용 비디오 (~2,140개, 20%)
    └── labels/     # 검증용 라벨 (JSON)
```

---

## 2. 카테고리 분류

### 2.1 대분류 (3개)

| 카테고리 | 영문명 | 설명 | 비율 |
|----------|--------|------|------|
| 자연 | Nature | 산, 바다, 강, 하늘, 숲 등 자연 경관 | ~35% |
| 도시 | Urban | 거리, 교통, 상업지구, 주거지역 등 | ~40% |
| 건축물 | Architecture | 전통 건축, 현대 건축, 랜드마크 등 | ~25% |

### 2.2 연도별 분류

- **2020년 이전**: 기존 영상 자료
- **2020-2023년**: 최근 촬영 영상
- **2024년**: 최신 데이터 (데이터설명서 기준)

### 2.3 상세 카테고리 예시

```
자연 (Nature)
├── 산 (Mountain)
├── 바다 (Sea/Ocean)
├── 강/호수 (River/Lake)
├── 하늘/구름 (Sky/Cloud)
├── 숲/나무 (Forest/Tree)
└── 계절별 풍경 (Seasonal Landscape)

도시 (Urban)
├── 거리/도로 (Street/Road)
├── 교통 (Traffic)
├── 상업지구 (Commercial Area)
├── 주거지역 (Residential Area)
├── 공원/광장 (Park/Square)
└── 야경 (Night View)

건축물 (Architecture)
├── 전통 건축 (Traditional)
│   ├── 한옥 (Hanok)
│   ├── 궁궐 (Palace)
│   └── 사찰 (Temple)
├── 현대 건축 (Modern)
│   ├── 고층 빌딩 (Skyscraper)
│   ├── 주거 건물 (Residential)
│   └── 상업 건물 (Commercial)
└── 랜드마크 (Landmark)
```

---

## 3. 데이터 형식

### 3.1 비디오 파일 (MP4)

| 속성 | 값 |
|------|-----|
| 형식 | MP4 (H.264) |
| 해상도 | 1920x1080 (FHD) 또는 1280x720 (HD) |
| 프레임레이트 | 30fps |
| 길이 | 5-30초 |
| 코덱 | H.264/AVC |

### 3.2 라벨 파일 (JSON)

```json
{
  "video_id": "nature_001_2024",
  "filename": "nature_001_2024.mp4",
  "category": {
    "main": "자연",
    "sub": "바다"
  },
  "metadata": {
    "duration": 15.5,
    "resolution": "1920x1080",
    "fps": 30,
    "year": 2024,
    "location": "부산 해운대"
  },
  "caption": {
    "korean": "푸른 바다 위로 하얀 파도가 부서지며 해변으로 밀려오고 있다. 멀리 수평선 위로 몇 척의 배가 지나가고, 맑은 하늘에는 갈매기들이 날아다니고 있다.",
    "english": null
  },
  "annotations": {
    "objects": ["바다", "파도", "해변", "배", "갈매기", "하늘"],
    "actions": ["부서지다", "밀려오다", "지나가다", "날아다니다"],
    "attributes": ["푸른", "하얀", "맑은"]
  }
}
```

### 3.3 캡션 특징

- **언어**: 한국어
- **길이**: 50-200자 (평균 약 100자)
- **스타일**: 상세 묘사 (객체, 동작, 속성, 배경 포함)
- **시제**: 현재 진행형 위주

---

## 4. 데이터 품질

### 4.1 품질 기준

| 항목 | 기준 |
|------|------|
| 영상 품질 | HD 이상, 흔들림 최소화 |
| 캡션 정확도 | 영상 내용과 일치 |
| 문법 정확성 | 맞춤법, 문법 오류 없음 |
| 묘사 상세도 | 객체, 동작, 속성 포함 |

### 4.2 검수 과정

1. **1차 검수**: 자동화 도구로 형식 검증
2. **2차 검수**: 전문가 영상-캡션 일치 확인
3. **3차 검수**: 언어 전문가 문법/표현 검토

---

## 5. 베이스라인 성능

### 5.1 AI-Hub 공식 베이스라인

| 메트릭 | 한국어 | 영어 |
|--------|--------|------|
| **METEOR** | 0.3052 | 0.33 |

### 5.2 성능 분석

- **한국어 METEOR 0.3052**: 영상 캡셔닝 태스크에서 준수한 성능
- **영어 대비 낮은 이유**: 
  - 한국어 형태소 분석 복잡성
  - 어순 차이 (SOV vs SVO)
  - 조사/어미 다양성

### 5.3 개선 목표

| 목표 | METEOR |
|------|--------|
| 베이스라인 | 0.3052 |
| 목표 1 (10% 향상) | 0.3357 |
| 목표 2 (20% 향상) | 0.3662 |
| 최종 목표 | 0.40+ |

---

## 6. 데이터 활용 가이드

### 6.1 학습 데이터 로딩

```python
import os
import json
from pathlib import Path

class KoreanVideoCaptionDataset:
    def __init__(self, data_root, split="train"):
        self.data_root = Path(data_root)
        self.split = split
        self.video_dir = self.data_root / split / "videos"
        self.label_dir = self.data_root / split / "labels"
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        for label_file in self.label_dir.glob("*.json"):
            with open(label_file, "r", encoding="utf-8") as f:
                label = json.load(f)
            video_path = self.video_dir / label["filename"]
            if video_path.exists():
                samples.append({
                    "video_path": str(video_path),
                    "caption": label["caption"]["korean"],
                    "category": label["category"]["main"],
                    "metadata": label["metadata"]
                })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# 사용 예시
DATA_ROOT = "/content/drive/MyDrive/mutsa-02/aihub_splitted"
train_dataset = KoreanVideoCaptionDataset(DATA_ROOT, split="train")
val_dataset = KoreanVideoCaptionDataset(DATA_ROOT, split="val")

print(f"학습 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(val_dataset)}개")
```

### 6.2 카테고리별 분포 확인

```python
from collections import Counter

def analyze_category_distribution(dataset):
    categories = [sample["category"] for sample in dataset.samples]
    distribution = Counter(categories)
    
    print("카테고리 분포:")
    for category, count in distribution.most_common():
        percentage = count / len(categories) * 100
        print(f"  {category}: {count}개 ({percentage:.1f}%)")
    
    return distribution

train_dist = analyze_category_distribution(train_dataset)
```

### 6.3 캡션 길이 분석

```python
import numpy as np

def analyze_caption_length(dataset):
    lengths = [len(sample["caption"]) for sample in dataset.samples]
    
    print("캡션 길이 통계:")
    print(f"  최소: {min(lengths)}자")
    print(f"  최대: {max(lengths)}자")
    print(f"  평균: {np.mean(lengths):.1f}자")
    print(f"  중앙값: {np.median(lengths):.1f}자")
    print(f"  표준편차: {np.std(lengths):.1f}자")
    
    return lengths

train_lengths = analyze_caption_length(train_dataset)
```

---

## 7. 데이터 전처리

### 7.1 비디오 프레임 추출

```python
import cv2
import numpy as np

def extract_frames(video_path, num_frames=8, sampling="uniform"):
    """비디오에서 프레임 추출"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if sampling == "uniform":
        # 균등 간격 샘플링
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif sampling == "keyframe":
        # 키프레임 기반 샘플링 (장면 변화 감지)
        indices = detect_keyframes(video_path, num_frames)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def detect_keyframes(video_path, num_frames):
    """장면 변화 기반 키프레임 감지"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 프레임 간 차이 계산
    diffs = []
    prev_frame = None
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
            diffs.append((i, diff))
        prev_frame = gray
    
    cap.release()
    
    # 차이가 큰 프레임 선택
    diffs.sort(key=lambda x: x[1], reverse=True)
    keyframe_indices = sorted([d[0] for d in diffs[:num_frames]])
    
    return keyframe_indices
```

### 7.2 텍스트 전처리

```python
import re

def preprocess_korean_caption(caption):
    """한국어 캡션 전처리"""
    # 불필요한 공백 제거
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    # 특수문자 정규화
    caption = re.sub(r'["""]', '"', caption)
    caption = re.sub(r"[''']", "'", caption)
    
    # 문장 끝 마침표 확인
    if not caption.endswith(('.', '!', '?')):
        caption += '.'
    
    return caption
```

---

## 8. 주의사항

### 8.1 데이터 사용 시 주의점

1. **저작권**: AI-Hub 이용약관 준수 필수
2. **비상업적 목적**: 연구/학습 목적으로만 사용
3. **출처 표기**: 논문/발표 시 AI-Hub 출처 명시

### 8.2 알려진 제한사항

- 일부 영상에 워터마크 포함 가능
- 카테고리 간 불균형 존재 (도시 > 자연 > 건축물)
- 캡션 스타일 일부 불일치 (작성자 간 차이)

### 8.3 데이터 확장 제안

- **데이터 증강**: 영상 augmentation (crop, flip, color jitter)
- **역번역**: 한→영→한 역번역으로 캡션 다양화
- **합성 데이터**: 유사 영상에 대한 캡션 생성

---

## 참고 자료

- [AI-Hub 데이터셋 페이지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71849)
- [METEOR 논문](https://aclanthology.org/W05-0909.pdf)
- [데이터설명서 PDF](./reference/데이터설명서.pdf)
