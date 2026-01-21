# 🎬 Korean Video Captioning

> 대한민국 배경영상 한국어 캡셔닝 모델 개발

LLaVA 기반 Vision-Language Model을 활용하여 한국어 비디오 캡션을 생성하는 프로젝트입니다.

[![Presentation](https://img.shields.io/badge/📊_Presentation-Slides-00D4AA?style=for-the-badge)](https://korean-video-captioning-html.vercel.app/)
[![YouTube](https://img.shields.io/badge/🎥_YouTube-Demo-FF0000?style=for-the-badge)](https://youtu.be/M1heDotZoRY)

---

## 📊 발표자료

**[👉 발표자료 바로가기](https://korean-video-captioning-html.vercel.app/)**

> 키보드 방향키(←→) 또는 스와이프로 슬라이드 이동

---

## 🎥 데모 영상

[![Demo Video](https://img.youtube.com/vi/M1heDotZoRY/maxresdefault.jpg)](https://youtu.be/M1heDotZoRY)

**[▶️ YouTube에서 보기](https://youtu.be/M1heDotZoRY)**

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 대한민국 배경영상에 대한 한국어 캡션 생성 |
| **데이터** | AI-Hub 대한민국 배경영상 데이터셋 (9,631 샘플) |
| **모델** | CLIP-ViT + Projector + Qwen3-8B |
| **평가** | METEOR + SigLIP2 + Diversity |

---

## 🏗️ 모델 아키텍처

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Video     │────▶│ Vision Encoder  │────▶│   Projector     │────▶│             │
│  (8 frames) │     │ CLIP-ViT-L/14   │     │  4종 비교 실험   │     │     LLM     │────▶ 한국어 캡션
└─────────────┘     │ (304M, Frozen)  │     │  (4M ~ 206M)    │     │ Qwen3-8B    │
                    └─────────────────┘     └─────────────────┘     │ (4-bit+LoRA)│
┌─────────────┐                                                     │             │
│    Text     │────────────────────────────────────────────────────▶└─────────────┘
│ Instruction │                  (Text Tokens)
└─────────────┘
```

### 컴포넌트

| 컴포넌트 | 모델 | 파라미터 | 역할 |
|----------|------|----------|------|
| Vision Encoder | CLIP-ViT-L/14 | 304M | 프레임 → 시각 특징 추출 (Frozen) |
| Projector | Linear / MLP / Perceiver / C-Abstractor | 4M ~ 206M | 시각 특징 → LLM 공간 변환 |
| LLM | Qwen3-8B | 8.2B | 한국어 캡션 생성 (4-bit + LoRA) |

---

## 📊 데이터셋

### AI-Hub 대한민국 배경영상

| 항목 | 값 |
|------|-----|
| 총 샘플 | 9,631 |
| Train | 8,563 (88.9%) |
| Validation | 1,068 (11.1%) |
| 평균 캡션 길이 | 575자 |

### 카테고리 분포

- **시가지** (40.3%): 도심 거리, 상업지역, 교통시설
- **자연** (34.3%): 산, 바다, 강, 공원
- **건축물** (25.3%): 전통/현대 건축, 랜드마크

---

## ⚙️ 학습 전략

### 2-Stage Training

| Stage | 학습 대상 | Learning Rate | Epochs | 목적 |
|-------|----------|---------------|--------|------|
| **Stage 1** | Projector only | 1e-3 | 2 | Vision→Language 공간 정렬 |
| **Stage 2** | Projector + LoRA | 5e-5 | 3 | 한국어 캡셔닝 최적화 |

### 적용된 최적화

- **Vision Feature 캐싱**: 30-40% 속도 향상
- **Mixed Precision (BF16)**: 연산 가속
- **4-bit 양자화**: LLM 메모리 75% 절약
- **Gradient Checkpointing**: Activation 메모리 60% 절약

---

## 📈 평가 지표

| 지표 | 역할 | 목표 |
|------|------|------|
| **METEOR** | 텍스트 품질 (단어 수준 유사도) | > 0.40 |
| **SigLIP2** | Vision-Text 정렬도 | > 0.10 |
| **Diversity** | Mode Collapse 탐지 | > 0.50 |

---

## 🔬 Projector 비교 실험

| Projector | 파라미터 | 출력 토큰 | 특징 |
|-----------|----------|-----------|------|
| **Linear** | 4M | 4,608 | 단순 선형 변환 ✅ 권장 |
| **MLP-2L** | 8M | 4,608 | 비선형 변환 ✅ 권장 |
| **Perceiver** | 134M | 64 | 토큰 압축 (대규모 데이터용) |
| **C-Abstractor** | 206M | 64 | Cross-Attention 기반 (대규모 데이터용) |

### 핵심 발견

> 작은 데이터셋(~1K)에서는 **단순한 Projector**(Linear/MLP)가 Mode Collapse 방지에 효과적

---

## 📁 프로젝트 구조

```
├── data/                   # 데이터셋 (추가 예정)
├── models/                 # 모델 체크포인트 (추가 예정)
├── src/                    # 소스 코드 (추가 예정)
│   ├── train.py
│   ├── evaluate.py
│   └── ...
├── notebooks/              # 실험 노트북 (추가 예정)
└── README.md
```

---

## 🚀 Quick Start

```bash
# 추가 예정
```

---

## 📝 핵심 교훈

1. **모델 크기 ≠ 성능**: 작은 데이터셋에서는 단순한 모델이 더 효과적
2. **평가 지표 다양화**: SigLIP 높아도 Diversity 낮으면 Mode Collapse
3. **한국어 = Qwen**: 토크나이저 효율 2.3배, 자연스러운 한국어 생성
4. **평가 모델 선택**: 한국어 캡셔닝에는 다국어 지원(SigLIP2) 필수

---

## 👥 팀원

| 이름 | 역할 |
|------|------|
| - | - |
| - | - |
| - | - |

---

## 📜 License

MIT License

---

## 🔗 References

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Qwen](https://github.com/QwenLM/Qwen)
- [AI-Hub 대한민국 배경영상](https://aihub.or.kr/)
