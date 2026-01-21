# SigLIP Score 기반 Projector 비교 연구

> **연구 질문**: "어떤 Projector 아키텍처가 Vision 정보를 LLM에 가장 효과적으로 전달하는가?"

## 배경

### 기존 문제
- METEOR는 LLM의 한국어 능력을 측정 → Projector 효과 분리 불가
- AI-Hub 베이스라인은 영어 생성 → GPT 번역 파이프라인 사용

### 새로운 접근
- **SigLIP Score**: Vision-Language 정렬을 직접 측정
- Projector가 Vision 정보를 얼마나 잘 전달하는지 평가 가능

---

## 실험 설계

### 모델 구성

```
┌─────────────────────────────────────────────────────────────┐
│                     Custom VLM Architecture                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Vision    │    │  Projector  │    │      LLM        │ │
│  │   Encoder   │───▶│  (학습)     │───▶│   (LoRA)        │ │
│  │  (Frozen)   │    │             │    │                 │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│                                                             │
│  CLIP-ViT-L/14      4종 비교          Qwen3-8B (4-bit)     │
│  336px              - Linear                                │
│  1024-dim           - MLP-2L                                │
│                     - C-Abstractor                          │
│                     - Perceiver                             │
└─────────────────────────────────────────────────────────────┘
```

### 고정 변수
- **Vision Encoder**: CLIP-ViT-L/14-336 (frozen)
- **LLM**: Qwen3-8B (4-bit 양자화)
- **PEFT**: LoRA (r=16, alpha=32)
- **데이터**: AI-Hub 샘플 (865 train, 97 val) - aihub_splitted 전체 사용
- **프레임**: 8 frames

### 실험 변수

| Phase | ID | Projector | PEFT | 담당 | 노트북 |
|-------|-----|-----------|------|------|--------|
| 1 | E1 | Linear | LoRA | Person A | 01_person_a.ipynb |
| 1 | E3 | MLP-2L | LoRA | Person A | 01_person_a.ipynb |
| 1 | E5 | C-Abstractor | LoRA | Person B | 02_person_b.ipynb |
| 1 | E7 | Perceiver | LoRA | Person C | 03_person_c.ipynb |
| 2 | E? | Best | DoRA | 본인 | 선택적 |

---

## 평가 지표

### Primary: SigLIP Score

```python
from transformers import AutoModel, AutoProcessor

# SigLIP 모델 로드 (학습 불필요, 다운로드만)
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# 평가
def compute_siglip_score(frames, caption):
    inputs = processor(text=[caption], images=frames, return_tensors="pt")
    outputs = model(**inputs)
    score = torch.sigmoid(outputs.logits_per_image).mean().item()
    return score
```

- **범위**: 0-1 (높을수록 Vision-Language 정렬 좋음)
- **한국어 지원**: O (SigLIP은 100+ 언어 학습)

### Secondary (참고용)
- 생성 캡션 정성 평가

---

## 폴더 구조

```
siglip_study/
├── README.md                 # 이 파일
├── notebooks/
│   ├── 01_person_a.ipynb     # Linear, MLP-2L + LoRA
│   ├── 02_person_b.ipynb     # C-Abstractor + LoRA
│   └── 03_person_c.ipynb     # Perceiver + LoRA
└── results/                  # 실험 결과 (Colab Drive에 저장)
```

---

## 실행 방법

### 1. Google Colab 설정

1. Google Drive에 이 폴더 업로드
2. Colab에서 노트북 열기
3. 런타임 → 런타임 유형 변경 → **A100 GPU** 선택

### 2. 노트북 실행

각 노트북은 self-contained:
- 셀 순서대로 실행
- Drive 마운트 → 환경 설정 → 모델 학습 → 평가

### 3. 결과 확인

```
Google Drive/mutsa-02/siglip_study_results/
├── E1_linear_lora/
│   ├── checkpoints/
│   ├── logs/
│   └── final_metrics.json
├── E3_mlp2l_lora/
└── ...
```

---

## 담당 배분

| 담당자 | 노트북 | 실험 | 예상 시간 |
|--------|--------|------|----------|
| Person A | 01_person_a.ipynb | E1, E3 | ~6시간 |
| Person B | 02_person_b.ipynb | E5 | ~3시간 |
| Person C | 03_person_c.ipynb | E7 | ~3시간 |

---

## 예상 결과

| Projector | 예상 SigLIP Score | 특징 |
|-----------|------------------|------|
| Linear | 0.3-0.4 | Baseline |
| MLP-2L | 0.4-0.5 | 비선형 변환 |
| C-Abstractor | 0.35-0.45 | 토큰 압축 |
| Perceiver | 0.45-0.55 | 최고 표현력 |

### 성공 기준
- Projector 간 유의미한 차이 (0.05+)
- Best Projector 선정

---

## 참고 자료

- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [LLaVA-1.5 Paper](https://arxiv.org/abs/2310.03744) - MLP Projector
- [Honeybee Paper](https://arxiv.org/abs/2312.06742) - C-Abstractor
- [Flamingo Paper](https://arxiv.org/abs/2204.14198) - Perceiver Resampler
