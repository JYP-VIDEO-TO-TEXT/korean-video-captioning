<div align="center">

# 🎬 Korean Video Captioning

### 대한민국 배경영상 한국어 캡셔닝 모델

LLaVA 기반 Vision-Language Model을 활용한 한국어 비디오 캡션 생성

[![Demo](https://img.shields.io/badge/🎯_Live_Demo-Presentation-00D4AA?style=for-the-badge)](https://korean-video-captioning-html.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br>

[📊 발표자료](#-발표자료) · [🎥 데모 영상](#-데모-영상) · [🏗️ 아키텍처](#️-모델-아키텍처) · [🚀 시작하기](#-quick-start)

</div>

---

## 📊 발표자료

<div align="center">

### 🖥️ Interactive Presentation

**[👉 발표자료 바로가기](https://korean-video-captioning-html.vercel.app/)**

<a href="https://korean-video-captioning-html.vercel.app/">
  <img src="https://img.shields.io/badge/📑_프레젠테이션_보기-Click_Here-00D4AA?style=for-the-badge&logoColor=white" alt="Presentation"/>
</a>

> 키보드 방향키(←→) 또는 스와이프로 슬라이드 이동

</div>

---

## 🎥 데모 영상

<div align="center">

### 프로젝트 소개 및 데모

<!-- 유튜브 영상 임베드 자리 -->
<!-- 아래 YOUR_VIDEO_ID를 실제 유튜브 영상 ID로 교체하세요 -->

[![Video Demo](https://img.shields.io/badge/▶️_YouTube-Demo_Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com)

<!--
[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)
-->

> 🎬 데모 영상 준비 중...

</div>

---

## 📌 프로젝트 개요

<table>
<tr>
<td width="50%">

### 🎯 목표
대한민국 배경영상에 대한 **자연스러운 한국어 캡션** 자동 생성

### 📊 데이터
**AI-Hub** 대한민국 배경영상 데이터셋
- 9,631 샘플 (Train 8,563 / Val 1,068)
- 평균 캡션 길이: 575자

### 🏆 성과 지표
| 지표 | 목표 |
|------|------|
| METEOR | > 0.40 |
| SigLIP2 | > 0.10 |
| Diversity | > 0.50 |

</td>
<td width="50%">

### 🛠️ 기술 스택

<p align="center">
<img src="https://img.shields.io/badge/CLIP-Vision_Encoder-blue?style=flat-square"/>
<img src="https://img.shields.io/badge/Qwen3--8B-LLM-green?style=flat-square"/>
<img src="https://img.shields.io/badge/LoRA-Fine--tuning-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/4--bit-Quantization-purple?style=flat-square"/>
</p>

### ⚡ 최적화
- Vision Feature 캐싱 (30-40% 속도↑)
- Mixed Precision (BF16)
- 4-bit 양자화 (메모리 75%↓)
- Gradient Checkpointing

</td>
</tr>
</table>

---

## 🏗️ 모델 아키텍처

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                   Korean Video Captioning VLM               │
                    └─────────────────────────────────────────────────────────────┘
                    
   ┌─────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────┐
   │   🎬 Video  │─────▶│  Vision Encoder │─────▶│   Projector     │─────▶│             │
   │  (8 frames) │      │  CLIP-ViT-L/14  │      │  ✨ 4종 비교     │      │    🤖 LLM   │────▶ 📝 한국어 캡션
   └─────────────┘      │  (304M, Frozen) │      │  (4M ~ 206M)    │      │  Qwen3-8B   │
                        └─────────────────┘      └─────────────────┘      │ (4-bit+LoRA)│
   ┌─────────────┐                                        │               │             │
   │  💬 Text    │────────────────────────────────────────┴──────────────▶└─────────────┘
   │ Instruction │                         (Text Tokens)
   └─────────────┘
```

### 🔧 컴포넌트

| 컴포넌트 | 모델 | 파라미터 | 역할 | 상태 |
|:--------:|:----:|:--------:|:----:|:----:|
| 🔍 Vision Encoder | CLIP-ViT-L/14 | 304M | 프레임 → 시각 특징 | ❄️ Frozen |
| 🔗 Projector | Linear / MLP / Perceiver / C-Abstractor | 4M ~ 206M | 시각 특징 → LLM 공간 | 🔥 Trainable |
| 🧠 LLM | Qwen3-8B | 8.2B | 한국어 캡션 생성 | 🔥 LoRA |

---

## 🔬 Projector 비교 실험

<div align="center">

| Projector | 파라미터 | 출력 토큰 | 특징 | 권장 |
|:---------:|:--------:|:---------:|:----:|:----:|
| **Linear** | 4M | 4,608 | 단순 선형 변환 | ✅ |
| **MLP-2L** | 8M | 4,608 | 비선형 변환 | ✅ |
| **Perceiver** | 134M | 64 | 토큰 압축 | ⚠️ 대규모용 |
| **C-Abstractor** | 206M | 64 | Cross-Attention | ⚠️ 대규모용 |

</div>

> 💡 **핵심 발견**: 작은 데이터셋(~1K)에서는 **단순한 Projector**(Linear/MLP)가 Mode Collapse 방지에 효과적

---

## ⚙️ 학습 전략

### 📚 2-Stage Training

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Projector Alignment                                           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                          │
│  • 학습: Projector만 (LLM Frozen)                                       │
│  • LR: 1e-3 (높음)  •  Epochs: 2                                        │
│  • 목표: Vision → Language 공간 정렬                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2: End-to-End Fine-tuning                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                       │
│  • 학습: Projector + LLM (LoRA)                                         │
│  • LR: 5e-5 (낮음)  •  Epochs: 3                                        │
│  • 목표: 한국어 캡셔닝 최적화                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 평가 체계

<div align="center">

| 지표 | 역할 | 목표 | 설명 |
|:----:|:----:|:----:|:----:|
| 📊 **METEOR** | 텍스트 품질 | > 0.40 | 단어 수준 유사도 |
| 🔗 **SigLIP2** | Vision-Text 정렬 | > 0.10 | 다국어 지원 |
| 🎯 **Diversity** | Mode Collapse 탐지 | > 0.50 | 캡션 다양성 |

</div>

---

## 📁 프로젝트 구조

```
📦 korean-video-captioning
├── 📂 data/                    # 데이터셋
│   ├── train/
│   └── val/
├── 📂 models/                  # 모델 체크포인트
│   └── checkpoints/
├── 📂 src/                     # 소스 코드
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   └── dataset.py
├── 📂 notebooks/               # 실험 노트북
├── 📂 configs/                 # 설정 파일
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🚀 Quick Start

```bash
# 1. 저장소 클론
git clone https://github.com/JYP-VIDEO-TO-TEXT/demo-repository.git
cd demo-repository

# 2. 환경 설정
pip install -r requirements.txt

# 3. 학습 실행
python src/train.py --config configs/linear.yaml

# 4. 평가
python src/evaluate.py --checkpoint models/best.pt
```

> ⚠️ 상세 설정 가이드 추가 예정

---

## 💡 핵심 교훈

<table>
<tr>
<td width="50%">

### 1️⃣ 모델 크기 ≠ 성능
> 작은 데이터셋에서는 **단순한 모델**이 더 효과적

```
C-Abstractor (206M) → Mode Collapse ❌
Linear (4M) → 안정적 학습 ✅
```

</td>
<td width="50%">

### 2️⃣ 평가 지표 다양화
> 단일 지표로 모델을 평가하지 말 것

```
SigLIP 높음 + Diversity 낮음 
= Mode Collapse 🚨
```

</td>
</tr>
<tr>
<td width="50%">

### 3️⃣ 한국어 = Qwen
> 한국어 토크나이저 효율 **2.3배**

```
LLaMA: 931 tokens
Qwen:  401 tokens ✨
```

</td>
<td width="50%">

### 4️⃣ 평가 모델도 다국어
> SigLIP v1 → SigLIP2로 전환

```
SigLIP v1 (영어): 0.003 ❌
SigLIP2 (다국어): 0.11+ ✅
```

</td>
</tr>
</table>

---

## 👥 Team

<div align="center">

| 이름 | 역할 | GitHub |
|:----:|:----:|:------:|
| - | - | - |
| - | - | - |
| - | - | - |

</div>

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 References

<div align="center">

[![LLaVA](https://img.shields.io/badge/LLaVA-GitHub-181717?style=flat-square&logo=github)](https://github.com/haotian-liu/LLaVA)
[![Qwen](https://img.shields.io/badge/Qwen-GitHub-181717?style=flat-square&logo=github)](https://github.com/QwenLM/Qwen)
[![AI-Hub](https://img.shields.io/badge/AI--Hub-Dataset-blue?style=flat-square)](https://aihub.or.kr/)

</div>

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by JYP-VIDEO-TO-TEXT Team

</div>
