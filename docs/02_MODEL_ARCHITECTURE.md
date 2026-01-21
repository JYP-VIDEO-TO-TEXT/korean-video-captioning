# 02. 모델 아키텍처 상세

## 개요

본 문서에서는 대한민국 배경영상 캡셔닝 태스크에 사용되는 Vision-Language Model(VLM)의 아키텍처를 상세히 설명합니다. 기본 모델인 LLaVA-NeXT-Video부터 최신 Vision Encoder와 LLM 백본 옵션까지 다룹니다.

---

## 1. Vision-Language Model 기본 구조

### 1.1 VLM 아키텍처 개요

Vision-Language Model은 크게 세 가지 주요 컴포넌트로 구성됩니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vision-Language Model                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│   │    Vision    │    │   Projector  │    │       LLM        │  │
│   │   Encoder    │───▶│    (MLP)     │───▶│    Backbone      │  │
│   └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                    │                     │             │
│         ▼                    ▼                     ▼             │
│   이미지/비디오를        시각 특징을           자연어 생성       │
│   고차원 특징으로        언어 공간으로                           │
│   인코딩                 투영                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 컴포넌트별 역할

| 컴포넌트 | 역할 | 입력 | 출력 |
|----------|------|------|------|
| **Vision Encoder** | 시각 정보를 고차원 특징 벡터로 변환 | 이미지/프레임 (224×224×3) | 패치 토큰 (196×1024) |
| **Projector** | 시각 특징을 언어 모델 임베딩 공간으로 투영 | 시각 토큰 (196×1024) | 언어 토큰 (196×4096) |
| **LLM Backbone** | 시각 토큰과 텍스트 토큰을 결합하여 캡션 생성 | 멀티모달 토큰 | 텍스트 토큰 |

---

## 2. LLaVA-NeXT-Video 아키텍처 분석

### 2.1 모델 개요

**LLaVA-NeXT-Video-7B**는 LLaVA(Large Language and Vision Assistant) 시리즈의 비디오 확장 모델입니다.

| 속성 | 값 |
|------|-----|
| 모델명 | `llava-hf/LLaVA-NeXT-Video-7B-hf` |
| 파라미터 수 | ~7B |
| Vision Encoder | CLIP-ViT-L/14 (336px) |
| LLM Backbone | Vicuna-7B (LLaMA 기반) |
| Projector | 2-layer MLP |

### 2.2 Vision Encoder: CLIP-ViT-L/14

```python
# CLIP Vision Encoder 구조
class CLIPVisionEncoder:
    """
    CLIP (Contrastive Language-Image Pre-training) Vision Encoder
    
    구조:
    - Patch Embedding: 이미지를 14×14 패치로 분할
    - Positional Embedding: 각 패치에 위치 정보 추가
    - Transformer Encoder: 12개 레이어
    - [CLS] Token: 전체 이미지 표현
    """
    
    def __init__(self):
        self.patch_size = 14
        self.image_size = 336
        self.hidden_size = 1024
        self.num_layers = 24
        self.num_heads = 16
        
        # 패치 수 계산: (336/14)^2 = 576
        self.num_patches = (self.image_size // self.patch_size) ** 2
```

**CLIP 특징:**
- **Contrastive Learning**: 이미지-텍스트 쌍으로 사전학습
- **Zero-shot 능력**: 새로운 카테고리에 대한 일반화
- **한계**: 세밀한 공간 정보 손실 가능

### 2.3 Projector: Multi-Layer Perceptron

```python
class MultiModalProjector(nn.Module):
    """
    Vision-to-Language Projector
    
    역할: Vision Encoder의 출력을 LLM 임베딩 공간으로 변환
    
    구조:
    - Linear(vision_hidden, llm_hidden)
    - GELU activation
    - Linear(llm_hidden, llm_hidden)
    """
    
    def __init__(self, vision_hidden=1024, llm_hidden=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden)
        )
    
    def forward(self, vision_features):
        # vision_features: [batch, num_patches, vision_hidden]
        # output: [batch, num_patches, llm_hidden]
        return self.projector(vision_features)
```

**Projector 설계 고려사항:**
- **깊이**: 2-layer가 일반적 (너무 깊으면 정보 손실)
- **활성화 함수**: GELU (smooth, gradient-friendly)
- **Normalization**: LayerNorm 선택적 사용

### 2.4 LLM Backbone: Vicuna-7B

```
Vicuna-7B 구조:
├── Embedding Layer
│   └── vocab_size: 32000
│   └── hidden_size: 4096
├── Transformer Layers (×32)
│   ├── Self-Attention
│   │   ├── num_heads: 32
│   │   └── head_dim: 128
│   ├── RMSNorm
│   └── Feed-Forward Network
│       ├── intermediate_size: 11008
│       └── activation: SiLU (Swish)
└── LM Head
    └── Linear(4096, 32000)
```

### 2.5 비디오 처리 방식

```python
def process_video(video_frames, vision_encoder, projector):
    """
    비디오를 LLM 입력으로 변환하는 과정
    
    1. 프레임별 Vision Encoding
    2. 프레임별 Projection
    3. 시간축 결합 (Temporal Concatenation)
    """
    
    # video_frames: [num_frames, C, H, W]
    all_visual_tokens = []
    
    for frame in video_frames:
        # 각 프레임을 패치로 분할하고 인코딩
        # frame: [C, H, W] -> patches: [num_patches, hidden]
        vision_features = vision_encoder(frame)  # [576, 1024]
        
        # 언어 공간으로 투영
        projected = projector(vision_features)   # [576, 4096]
        all_visual_tokens.append(projected)
    
    # 시간축으로 결합
    # [num_frames, num_patches, hidden] -> [num_frames * num_patches, hidden]
    visual_tokens = torch.cat(all_visual_tokens, dim=0)
    
    return visual_tokens  # [4608, 4096] for 8 frames
```

**비디오 토큰 수 계산:**
- 프레임 수: 8
- 프레임당 패치: 576 (336/14)²
- 총 시각 토큰: 8 × 576 = 4,608 토큰

---

## 3. Vision Encoder 옵션 비교

### 3.1 옵션 개요

| Encoder | 학습 방식 | 특징 | 파라미터 | 장점 | 단점 |
|---------|----------|------|----------|------|------|
| **CLIP-ViT** | Contrastive | 이미지-텍스트 정렬 | 304M | 범용성, 안정성 | 세밀한 공간 정보 부족 |
| **SigLIP 2** | Contrastive | 시그모이드 손실 | 400M | 다국어, 효율성 | 상대적으로 새로운 모델 |
| **DINOv3** | Self-supervised | Dense feature | 1B+ | 세밀한 특징, Gram Anchoring | 메모리 사용량 높음 |
| **InternViT** | Hybrid | 동적 해상도 | 6B | 고해상도, 확장성 | 매우 큰 모델 |

### 3.2 CLIP-ViT (기본값)

```python
# CLIP-ViT 사용 예시
from transformers import CLIPVisionModel, CLIPImageProcessor

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# 이미지 처리
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 출력: last_hidden_state [1, 577, 1024]
# 577 = 1 (CLS token) + 576 (패치 토큰)
```

**선택 이유:**
- 검증된 안정성 (수많은 VLM에서 사용)
- T4/L4에서 메모리 효율적
- LLaVA-NeXT-Video 기본 구성

### 3.3 SigLIP 2

```python
# SigLIP 2 사용 예시
from transformers import SiglipVisionModel, SiglipImageProcessor

model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
```

**SigLIP 2의 핵심 개선:**

1. **시그모이드 손실 함수**
   ```
   기존 CLIP: Softmax Cross-Entropy
   L = -log(exp(sim(i,t)) / Σ exp(sim(i,t')))
   
   SigLIP: Sigmoid Binary Cross-Entropy
   L = -log(σ(sim(i,t))) - Σ log(1 - σ(sim(i,t')))
   ```
   - 배치 크기에 덜 민감
   - 더 안정적인 학습

2. **다국어 지원**
   - 109개 언어로 학습
   - 한국어 텍스트-이미지 정렬 향상

**선택 이유:**
- 한국어 캡셔닝에 유리한 다국어 지원
- CLIP 대비 향상된 fine-grained 이해

### 3.4 DINOv3

> **참고**: DINOv3 모델은 Meta 라이선스 승인이 필요합니다. [승인 절차](00_MODELS_APPROVAL.md) 참조

```python
# DINOv3 사용 예시 (승인 후)
from transformers import AutoModel, AutoImageProcessor

# 사용 가능한 DINOv3 모델:
# - facebook/dinov3-vits16-pretrain-lvd1689m (21M)
# - facebook/dinov3-vitb16-pretrain-lvd1689m (86M)
# - facebook/dinov3-vitl16-pretrain-lvd1689m (300M)

model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
```

**DINOv3의 핵심 기술: Gram Anchoring**

```
기존 DINO 문제: 특징 붕괴 (Feature Collapse)
- Self-supervised 학습 시 모든 이미지가 비슷한 특징으로 수렴

DINOv3 해결책: Gram Anchoring
1. 특징 행렬의 Gram Matrix 계산: G = F @ F.T
2. Gram Matrix의 고유값(eigenvalues) 모니터링
3. 고유값 분포가 균일하도록 정규화
4. 다양한 특징 유지
```

**DINOv3 장점:**
- **Dense Feature**: 픽셀 수준의 세밀한 특징
- **Self-supervised**: 라벨 없이 대규모 학습
- **Semantic Segmentation**: 자연스러운 객체 분리

**선택 이유:**
- 배경영상의 세밀한 시각적 특징 포착
- 자연, 도시, 건축물의 디테일 표현 향상

### 3.5 Vision Encoder 선택 가이드

```
GPU 메모리 기준 선택:

T4 (16GB):
  └─ CLIP-ViT-L/14 (권장)
     - 메모리 효율적
     - 검증된 성능

L4 (24GB):
  └─ SigLIP 2-So400M (권장)
     - 다국어 지원으로 한국어 향상
     - 적절한 메모리 사용

A100 (40GB):
  └─ SigLIP 2-So400M 또는 DINOv3-ViT-L
     - 더 나은 특징 추출
     - 실험적 조합 가능

H100 (80GB):
  └─ DINOv3-Giant (권장)
     - 최고 성능
     - 메모리 충분
```

---

## 4. LLM 백본 옵션 비교

### 4.1 옵션 개요

| LLM | 파라미터 | 한국어 성능 | 특징 | 권장 GPU |
|-----|----------|------------|------|----------|
| **Vicuna-7B** | 7B | 보통 | LLaVA 기본 | T4 |
| **Qwen3-4B-Instruct** | 4B | 우수 | 경량 모델 | T4 |
| **Qwen3-8B-Instruct** | 8B | 우수 | 다국어 특화 | L4 |
| **Qwen3-14B-Instruct** | 14B | 매우 우수 | 향상된 추론 | A100 |
| **Qwen3-32B-Instruct** | 32B | 최상 | 최고 품질 | H100 |
| **Qwen3-30B-A3B** | 30B (3B active) | 우수 | MoE 효율성 | A100/H100 |

### 4.2 Qwen 시리즈 선택 이유

**1. 한국어 성능**
```
다국어 벤치마크 (한국어 포함):
- Qwen3-8B: MMLU-Ko 68.5%
- Vicuna-7B: MMLU-Ko 52.1%
- 차이: +16.4% 향상
```

**2. 아키텍처 비교**

```python
# Vicuna (LLaMA 기반)
class VicunaConfig:
    hidden_size = 4096
    intermediate_size = 11008
    num_attention_heads = 32
    num_hidden_layers = 32
    vocab_size = 32000
    max_position_embeddings = 4096

# Qwen3
class Qwen3Config:
    hidden_size = 4096
    intermediate_size = 11008
    num_attention_heads = 32
    num_key_value_heads = 8  # GQA (Grouped Query Attention)
    num_hidden_layers = 32
    vocab_size = 151936  # 더 큰 vocab (다국어 지원)
    max_position_embeddings = 131072  # 128K 컨텍스트
```

**Qwen의 핵심 개선:**

1. **Grouped Query Attention (GQA)**
   ```
   기존 MHA (Multi-Head Attention):
   - Query, Key, Value 모두 동일한 head 수
   - 메모리: O(n_heads × d_head × seq_len)
   
   GQA:
   - Key/Value head를 그룹으로 공유
   - 메모리: O(n_kv_heads × d_head × seq_len)
   - 32 heads, 8 KV heads -> 4x 메모리 절약
   ```

2. **더 큰 어휘집 (Vocabulary)**
   - Vicuna: 32,000 토큰
   - Qwen: 151,936 토큰
   - 한국어, 중국어 등 비라틴 언어 지원 향상

3. **RoPE 확장 (Rotary Position Embedding)**
   - 최대 32K 토큰 컨텍스트
   - 긴 비디오 설명 생성 가능

### 4.3 Qwen3 MoE 모델

**Mixture of Experts (MoE) 개념:**

```
┌────────────────────────────────────────────────────────────┐
│                    MoE Layer                                │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──▶ [Router] ──▶ Top-K Expert Selection            │
│                 │                                           │
│         ┌──────┼──────┐──────┐──────┐──────┐               │
│         ▼      ▼      ▼      ▼      ▼      ▼               │
│      [E1]   [E2]   [E3]   [E4]   [E5]   [E6]  ... [E64]   │
│         │      │      │                                     │
│         └──────┴──────┘                                     │
│                │                                            │
│              [Sum] ──▶ Output                               │
│                                                             │
│   예: 64개 Expert 중 3개만 활성화 (Top-3)                   │
│   파라미터: 30B, 활성화: 3B                                 │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Qwen3-30B-A3B 특징:**
- **총 파라미터**: 30B
- **활성 파라미터**: 3B (토큰당)
- **Expert 수**: 64개
- **활성 Expert**: 3개 (Top-3)

```python
# MoE 라우팅 예시
class MoERouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        # x: [batch, seq_len, hidden]
        logits = self.gate(x)  # [batch, seq_len, num_experts]
        
        # Top-K expert 선택
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        return weights, indices
```

**MoE 장점:**
- 30B 모델 성능, 3B 모델 추론 비용
- 메모리 효율적 (활성 파라미터만 로드)
- 다양한 태스크에 특화된 expert 학습

### 4.4 LLM 교체 방법 (LLaVA 기반)

```python
from transformers import LlavaNextVideoForConditionalGeneration, AutoTokenizer

# 방법 1: 기존 LLaVA 모델에서 LLM만 교체 (복잡)
# - Vision Encoder와 Projector는 유지
# - LLM 백본만 Qwen으로 교체
# - Projector 재학습 필요

# 방법 2: Qwen-VL 계열 사용 (권장)
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Qwen2-VL은 이미 통합된 Vision-Language 모델
# Vision Encoder: ViT (자체 학습)
# LLM: Qwen3
# Projector: Cross-Attention 기반
```

### 4.5 LLM 선택 가이드

```
태스크 요구사항 기준:

한국어 품질 우선:
  └─ Qwen3 시리즈 권장
     - 우수한 한국어 생성 품질
     - 자연스러운 문장 구조

추론 속도 우선:
  └─ Qwen3-30B-A3B (MoE)
     - 30B 성능, 3B 비용
     - 배치 추론에 효율적

메모리 제약:
  └─ Qwen3-8B (4bit)
     - T4에서도 실행 가능
     - 품질-효율 균형
```

---

## 5. 모델 조합 권장

### 5.1 GPU별 권장 조합

| GPU | Vision Encoder | LLM | Projector | 예상 VRAM |
|-----|---------------|-----|-----------|-----------|
| **T4** | CLIP-ViT-L | Vicuna-7B (4bit) | MLP | ~14GB |
| **L4** | SigLIP 2-So400M | Qwen3-8B (4bit) | MLP | ~18GB |
| **A100** | DINOv3-ViT-L | Qwen3-14B (4bit) | MLP | ~28GB |
| **H100** | DINOv3-ViT-H | Qwen3-32B (4bit) | Cross-Attn | ~50GB |

### 5.2 품질 우선 조합

```python
# H100 최고 품질 설정
config = {
    "vision_encoder": {
        "type": "dinov3",
        "name": "facebook/dinov3-giant",  # 예상
        "image_size": 448,
    },
    "llm": {
        "name": "Qwen/Qwen3-32B-Instruct",
        "quantization": "4bit",
    },
    "projector": {
        "type": "cross_attention",  # MLP 대신 Cross-Attention
        "num_layers": 4,
    },
    "training": {
        "batch_size": 8,
        "num_frames": 16,
    }
}
```

### 5.3 효율 우선 조합

```python
# T4 효율 설정
config = {
    "vision_encoder": {
        "type": "clip",
        "name": "openai/clip-vit-large-patch14-336",
        "image_size": 336,
    },
    "llm": {
        "name": "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "quantization": "4bit",
    },
    "projector": {
        "type": "mlp",
        "num_layers": 2,
    },
    "training": {
        "batch_size": 1,
        "num_frames": 4,
        "gradient_checkpointing": True,
    }
}
```

---

## 6. 아키텍처 개선 방향

### 6.1 단기 개선 (현재 가능)

1. **Vision Encoder 업그레이드**
   - CLIP → SigLIP 2 (다국어 향상)
   - 한국어 캡션 품질 개선 기대

2. **LLM 백본 교체**
   - Vicuna → Qwen3 (한국어 특화)
   - 더 자연스러운 한국어 생성

3. **Projector 개선**
   - 2-layer MLP → 4-layer MLP
   - 더 풍부한 시각-언어 정렬

### 6.2 중기 개선 (연구 필요)

1. **Cross-Attention Projector**
   ```python
   class CrossAttentionProjector(nn.Module):
       def __init__(self, vision_hidden, llm_hidden, num_heads=8):
           super().__init__()
           self.cross_attn = nn.MultiheadAttention(
               embed_dim=llm_hidden,
               num_heads=num_heads,
               kdim=vision_hidden,
               vdim=vision_hidden
           )
           self.norm = nn.LayerNorm(llm_hidden)
       
       def forward(self, text_embeds, vision_features):
           # text_embeds: [seq_len, batch, llm_hidden]
           # vision_features: [num_patches, batch, vision_hidden]
           attn_output, _ = self.cross_attn(
               text_embeds, vision_features, vision_features
           )
           return self.norm(attn_output + text_embeds)
   ```

2. **Temporal Modeling 강화**
   - 프레임 간 관계 학습
   - Temporal Attention 추가

### 6.3 장기 개선 (최신 연구)

1. **동적 해상도 처리**
   - 고정 해상도 → 적응적 해상도
   - 중요 영역에 더 많은 토큰 할당

2. **Sparse Attention**
   - 긴 비디오를 위한 효율적 attention
   - Longformer 스타일 적용

---

## 7. 참고 자료

### 논문
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- [LLaVA-NeXT: Improved Reasoning](https://arxiv.org/abs/2310.03744)
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)

### 코드 레포지토리
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Qwen GitHub](https://github.com/QwenLM/Qwen)
