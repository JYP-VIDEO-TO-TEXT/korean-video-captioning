# Vision-Language Model (VLM) 아키텍처

> **학습 목표**: VLM의 구조와 각 컴포넌트의 역할 이해

---

## 1. VLM이란?

**Vision-Language Model (VLM)**은 이미지/비디오를 이해하고 자연어로 설명할 수 있는 멀티모달 AI 모델입니다.

### 핵심 능력
- 이미지 캡셔닝 (Image Captioning)
- 비디오 캡셔닝 (Video Captioning)
- Visual Question Answering (VQA)
- 이미지 기반 대화 (Visual Chat)

---

## 2. VLM 발전 계보

```
CLIP (2021)           LLaVA (2023)           LLaVA-1.5 (2023)
    │                     │                       │
    ▼                     ▼                       ▼
Vision-Text 정렬      VLM 대중화              MLP Projector
Contrastive Learning  Linear Projector        더 나은 성능
```

### 주요 모델들

| 모델 | Vision Encoder | Projector | LLM | 특징 |
|------|---------------|-----------|-----|------|
| LLaVA | CLIP-ViT | Linear | Vicuna | 최초 오픈소스 VLM |
| LLaVA-1.5 | CLIP-ViT | MLP | Vicuna | MLP로 성능 향상 |
| Qwen-VL | ViT | Resampler | Qwen | 토큰 압축 |
| InternVL | InternViT | MLP | InternLM | 대규모 학습 |
| Honeybee | CLIP-ViT | C-Abstractor | Vicuna | 효율적 압축 |

---

## 3. VLM 아키텍처 구성

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Vision-Language Model                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │    Video     │     │   Vision     │     │      Projector       │ │
│  │   Frames     │────▶│   Encoder    │────▶│                      │ │
│  │  (8 frames)  │     │  (Frozen)    │     │  (Trainable)         │ │
│  └──────────────┘     └──────────────┘     └──────────┬───────────┘ │
│                                                        │             │
│                                                        ▼             │
│                       ┌──────────────┐     ┌──────────────────────┐ │
│                       │    Text      │     │        LLM           │ │
│                       │   Prompt     │────▶│                      │ │
│                       │              │     │  (LoRA Trainable)    │ │
│                       └──────────────┘     └──────────┬───────────┘ │
│                                                        │             │
│                                                        ▼             │
│                                            ┌──────────────────────┐ │
│                                            │   Generated Text     │ │
│                                            │   (Korean Caption)   │ │
│                                            └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 컴포넌트별 상세 설명

### 4.1 Vision Encoder

**역할**: 이미지/비디오를 고차원 특징 벡터로 변환

#### 우리 실험: CLIP-ViT-L/14-336

```python
from transformers import CLIPVisionModel

vision_encoder = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-large-patch14-336"
)
```

**입출력 차원**:
- 입력: 이미지 `(3, 336, 336)` - RGB, 336x336 픽셀
- 출력: `(577, 1024)` - 576 패치 + 1 CLS 토큰, 각 1024차원

#### 패치 분할 과정

```
336x336 이미지
    │
    ▼ (14x14 패치로 분할)
┌───┬───┬───┬─────┬───┐
│ 1 │ 2 │ 3 │ ... │24 │   24 x 24 = 576 패치
├───┼───┼───┼─────┼───┤
│25 │26 │27 │ ... │48 │
├───┼───┼───┼─────┼───┤
│...│...│...│ ... │...│
└───┴───┴───┴─────┴───┘
    │
    ▼ (Transformer 처리)
576 patches × 1024 dim
```

#### 비디오 처리 (8 프레임)

```
비디오 → 8개 프레임 샘플링 → 각 프레임 Vision Encoder 통과
       
Frame 1: (576, 1024)
Frame 2: (576, 1024)
   ...
Frame 8: (576, 1024)
       │
       ▼ (Flatten)
(8 × 576, 1024) = (4608, 1024) vision tokens
```

---

### 4.2 Projector

**역할**: Vision 특징을 LLM이 이해할 수 있는 공간으로 변환

#### 왜 Projector가 필요한가?

Vision Encoder와 LLM은 서로 다른 "언어"를 사용합니다:

| 구분 | Vision Encoder | LLM |
|------|---------------|-----|
| 차원 | 1024 | 4096 |
| 의미 | 시각적 패턴 | 언어적 의미 |
| 학습 데이터 | 이미지-텍스트 쌍 | 텍스트만 |

Projector는 이 둘 사이의 **"번역기"** 역할을 합니다.

> 자세한 내용은 [02_projector_types.md](02_projector_types.md) 참조

---

### 4.3 LLM (Large Language Model)

**역할**: Vision 정보와 텍스트 프롬프트를 기반으로 캡션 생성

#### 우리 실험: Qwen3-8B

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**왜 Qwen3인가?**
- 한국어 토큰화 효율 2.3배 (LLaMA 대비)
- 강력한 다국어 지원
- 최신 아키텍처

---

## 5. Forward Pass 수식

### 전체 프로세스

$$
\text{Caption} = \text{LLM}(\text{Projector}(\text{VisionEncoder}(\text{Video})) \oplus \text{Prompt})
$$

### 단계별 수식

#### Step 1: Vision Encoding

$$
\mathbf{V} = \text{ViT}(\mathbf{X}_{frames}) \in \mathbb{R}^{N_v \times d_v}
$$

- $\mathbf{X}_{frames}$: 입력 비디오 프레임 (8개)
- $N_v = 8 \times 576 = 4608$: 총 vision 토큰 수
- $d_v = 1024$: vision 특징 차원

#### Step 2: Projection

$$
\mathbf{H}_v = \text{Projector}(\mathbf{V}) \in \mathbb{R}^{N_p \times d_{llm}}
$$

- $N_p$: projection 후 토큰 수 (Projector 유형에 따라 다름)
- $d_{llm} = 4096$: LLM 임베딩 차원

#### Step 3: Text Embedding

$$
\mathbf{H}_t = \text{Embed}(\text{Prompt}) \in \mathbb{R}^{N_t \times d_{llm}}
$$

#### Step 4: Concatenation & Generation

$$
\mathbf{H}_{input} = [\mathbf{H}_v ; \mathbf{H}_t] \in \mathbb{R}^{(N_p + N_t) \times d_{llm}}
$$

$$
\text{Caption} = \text{LLM.generate}(\mathbf{H}_{input})
$$

---

## 6. 학습 시 Gradient Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Vision    │     │  Projector  │     │     LLM     │
│   Encoder   │────▶│             │────▶│   (LoRA)    │
│  (Frozen)   │     │ (Trainable) │     │ (Trainable) │
└─────────────┘     └─────────────┘     └─────────────┘
      ✗                   ✓                   ✓
   No Grad              Grad                Grad
```

**왜 Vision Encoder를 Freeze하는가?**
1. 이미 강력한 시각적 표현 학습됨 (CLIP 4억 쌍 학습)
2. 파라미터 수 절약 (~300M 파라미터)
3. 과적합 방지

---

## 7. 메모리 분석 (A100 80GB 기준)

| 컴포넌트 | 파라미터 | 메모리 (FP16) | 메모리 (4-bit) |
|----------|---------|--------------|----------------|
| Vision Encoder | 304M | ~0.6GB | N/A (Frozen) |
| Projector | 4-134M | ~0.03-0.27GB | N/A |
| Qwen3-8B | 8B | ~16GB | **~4GB** |
| Activations | - | ~10-20GB | ~10-20GB |
| **Total** | - | ~30GB | **~15-25GB** |

---

## 8. 핵심 개념 요약

| 컴포넌트 | 역할 | 상태 | 차원 |
|----------|------|------|------|
| Vision Encoder | 이미지 → 특징 벡터 | Frozen | 1024-d |
| Projector | 차원 변환 + 정보 압축 | Trainable | 1024→4096 |
| LLM | 캡션 생성 | LoRA Trainable | 4096-d |

---

## 참고 자료

- [LLaVA Paper](https://arxiv.org/abs/2304.08485) - Visual Instruction Tuning
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609) - Qwen Technical Report
