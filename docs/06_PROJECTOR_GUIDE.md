# 06. Projector 가이드: Vision-Language 연결의 핵심

> Custom VLM (CLIP + Qwen3-8B) 실험을 위한 Projector 동작 원리 및 구현 가이드

---

## 목차

1. [VLM 아키텍처 개요](#1-vlm-아키텍처-개요)
2. [Vision Encoder 출력 구조](#2-vision-encoder-출력-구조)
3. [Projector 종류별 구현](#3-projector-종류별-구현)
4. [LoRA vs DoRA 비교](#4-lora-vs-dora-비교)
5. [Forward Pass 전체 흐름](#5-forward-pass-전체-흐름)
6. [실험 설정](#6-실험-설정)

---

## 1. VLM 아키텍처 개요

### 1.1 기본 구조

Vision-Language Model은 세 가지 핵심 컴포넌트로 구성됩니다:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Custom VLM Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│   │  Vision Encoder  │    │    Projector     │    │   LLM Backbone   │  │
│   │   (CLIP-ViT-L)   │───▶│  (Linear/MLP/    │───▶│   (Qwen3-8B)     │  │
│   │                  │    │   C-Abstractor/  │    │                  │  │
│   │   [동결됨]       │    │   Perceiver)     │    │   [LoRA/DoRA]    │  │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│           │                       │                        │             │
│           ▼                       ▼                        ▼             │
│     비디오 프레임을          시각 특징을              한국어 캡션        │
│     고차원 특징으로          LLM 임베딩               생성               │
│     인코딩                   공간으로 변환                               │
│                                                                          │
│   Input: [8, 3, 336, 336]   [8×576, 1024]           "이 영상은..."      │
│          (8 frames)          → [8×576, 4096]                            │
│                              or [64, 4096]                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 컴포넌트별 역할

| 컴포넌트 | 모델 | 역할 | 학습 여부 |
|----------|------|------|-----------|
| **Vision Encoder** | CLIP-ViT-L/14-336 | 이미지/비디오 → 시각 특징 추출 | 동결 |
| **Projector** | Linear/MLP/C-Abstractor/Perceiver | 시각 특징 → LLM 임베딩 공간 변환 | **전체 학습** |
| **LLM Backbone** | Qwen3-8B | 시각+텍스트 토큰 → 캡션 생성 | LoRA/DoRA |

### 1.3 왜 Projector가 중요한가?

Vision Encoder와 LLM은 서로 다른 임베딩 공간에서 학습되었습니다:

- **CLIP**: 이미지-텍스트 대조 학습 (contrastive learning)
- **Qwen3**: 텍스트 생성 학습 (causal language modeling)

Projector는 이 두 공간을 **정렬(align)**하는 다리 역할을 합니다.

```python
# Vision Encoder 출력: [batch, 576, 1024]
# LLM 입력 필요:      [batch, ?, 4096]
#                              ↑
#                    Projector가 이 변환을 담당
```

---

## 2. Vision Encoder 출력 구조

### 2.1 CLIP-ViT-L/14-336 출력

```python
from transformers import CLIPVisionModel, CLIPImageProcessor

# 모델 로드
vision_encoder = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-large-patch14-336"
)
processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-large-patch14-336"
)

# 이미지 전처리 (336x336으로 리사이즈)
inputs = processor(images=image, return_tensors="pt")
# inputs["pixel_values"].shape: [1, 3, 336, 336]

# Forward pass
outputs = vision_encoder(**inputs)

# 출력 구조
print(outputs.last_hidden_state.shape)  # [1, 577, 1024]
print(outputs.pooler_output.shape)       # [1, 1024]
```

### 2.2 출력 텐서 해부

```
last_hidden_state: [batch, 577, 1024]
                          │     │
                          │     └── hidden_size (CLIP-ViT-L)
                          │
                          └── 577 = 1 (CLS token) + 576 (patch tokens)
                                     │
                                     └── 576 = (336/14)² = 24×24 패치
```

**패치 토큰 추출 (CLS 제외)**:

```python
# CLS 토큰 제외하고 패치 토큰만 사용
vision_features = outputs.last_hidden_state[:, 1:, :]  # [batch, 576, 1024]
```

### 2.3 비디오 멀티프레임 처리

```python
def encode_video_frames(vision_encoder, frames, device):
    """
    비디오 프레임들을 Vision Encoder로 인코딩
    
    Args:
        vision_encoder: CLIP Vision Encoder
        frames: [num_frames, 3, 336, 336] 텐서
        device: cuda device
        
    Returns:
        vision_features: [num_frames * 576, 1024] 텐서
    """
    num_frames = frames.shape[0]
    all_features = []
    
    with torch.no_grad():
        for i in range(num_frames):
            frame = frames[i:i+1]  # [1, 3, 336, 336]
            outputs = vision_encoder(pixel_values=frame.to(device))
            
            # CLS 토큰 제외
            patch_features = outputs.last_hidden_state[:, 1:, :]  # [1, 576, 1024]
            all_features.append(patch_features.squeeze(0))  # [576, 1024]
    
    # 시간축으로 결합 (Temporal Concatenation)
    vision_features = torch.cat(all_features, dim=0)  # [num_frames * 576, 1024]
    
    return vision_features

# 사용 예시
frames = load_video_frames("video.mp4", num_frames=8)  # [8, 3, 336, 336]
vision_features = encode_video_frames(vision_encoder, frames, device)
print(vision_features.shape)  # [4608, 1024]  (8 * 576 = 4608)
```

### 2.4 토큰 수 계산

| 설정 | 계산 | 토큰 수 |
|------|------|---------|
| 1 frame, 336px | (336/14)² | 576 |
| 4 frames, 336px | 4 × 576 | 2,304 |
| 8 frames, 336px | 8 × 576 | 4,608 |
| 16 frames, 336px | 16 × 576 | 9,216 |

> **주의**: 토큰 수가 많을수록 LLM 연산량이 증가합니다. C-Abstractor나 Perceiver는 이 토큰 수를 압축합니다.

---

## 3. Projector 종류별 구현

### 3.1 Linear Projector (가장 단순)

가장 기본적인 선형 변환입니다.

```python
class LinearProjector(nn.Module):
    """
    단순 선형 변환 Projector
    
    - LLaVA v1 초기 버전에서 사용
    - 파라미터 수: vision_dim × llm_dim = 1024 × 4096 ≈ 4M
    """
    
    def __init__(self, vision_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [num_tokens, vision_dim] 또는 [batch, num_tokens, vision_dim]
            
        Returns:
            projected: [num_tokens, llm_dim] 또는 [batch, num_tokens, llm_dim]
        """
        return self.proj(vision_features)

# 사용 예시
projector = LinearProjector(vision_dim=1024, llm_dim=4096)
vision_features = torch.randn(4608, 1024)  # 8 frames
projected = projector(vision_features)
print(projected.shape)  # [4608, 4096]
```

**특징**:
- 파라미터: ~4M
- 장점: 빠른 학습, 적은 메모리
- 단점: 표현력 제한, 비선형 패턴 학습 불가

---

### 3.2 MLP Projector (LLaVA 기본)

2-layer MLP로 비선형 변환을 추가합니다.

```python
class MLPProjector(nn.Module):
    """
    2-Layer MLP Projector
    
    - LLaVA v1.5 기본 구조
    - GELU 활성화 함수로 비선형성 추가
    - 파라미터 수: 1024×4096 + 4096×4096 ≈ 21M
    """
    
    def __init__(self, vision_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [num_tokens, vision_dim] 또는 [batch, num_tokens, vision_dim]
            
        Returns:
            projected: [num_tokens, llm_dim] 또는 [batch, num_tokens, llm_dim]
        """
        return self.proj(vision_features)

# 사용 예시
projector = MLPProjector(vision_dim=1024, llm_dim=4096)
vision_features = torch.randn(4608, 1024)
projected = projector(vision_features)
print(projected.shape)  # [4608, 4096]
```

**특징**:
- 파라미터: ~21M
- 장점: 비선형 패턴 학습, 표현력 향상
- 단점: 토큰 수 유지 (576 또는 4608 그대로)

---

### 3.3 C-Abstractor (Cross-Attention 기반)

Learnable query를 사용해 vision feature를 압축합니다.

```python
class CAbstractor(nn.Module):
    """
    Cross-Attention based Abstractor
    
    - Learnable query로 vision feature 압축
    - 576 또는 4608 토큰 → 64 토큰으로 압축
    - 파라미터 수: ~50M (query + cross-attention + FFN)
    """
    
    def __init__(
        self, 
        vision_dim: int = 1024, 
        llm_dim: int = 4096, 
        num_queries: int = 64,
        num_heads: int = 8
    ):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, llm_dim))
        
        # Vision feature를 llm_dim으로 변환
        self.vision_proj = nn.Linear(vision_dim, llm_dim)
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            kdim=llm_dim,
            vdim=llm_dim,
            batch_first=True
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)
        )
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, num_tokens, vision_dim] 또는 [num_tokens, vision_dim]
            
        Returns:
            output: [batch, num_queries, llm_dim] 또는 [num_queries, llm_dim]
        """
        # Handle unbatched input
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = vision_features.shape[0]
        
        # Project vision features
        v = self.vision_proj(vision_features)  # [batch, num_tokens, llm_dim]
        
        # Expand queries for batch
        q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_queries, llm_dim]
        
        # Cross-Attention: queries attend to vision features
        attn_output, _ = self.cross_attn(
            query=q,
            key=v,
            value=v
        )
        
        # Residual + LayerNorm
        x = self.norm1(q + attn_output)
        
        # FFN + Residual + LayerNorm
        x = self.norm2(x + self.ffn(x))
        
        if squeeze_output:
            x = x.squeeze(0)
        
        return x

# 사용 예시
projector = CAbstractor(vision_dim=1024, llm_dim=4096, num_queries=64)
vision_features = torch.randn(1, 4608, 1024)  # 8 frames, batched
projected = projector(vision_features)
print(projected.shape)  # [1, 64, 4096] - 4608 → 64 토큰으로 압축!
```

**특징**:
- 파라미터: ~50M
- 장점: 토큰 수 대폭 압축 (4608 → 64), LLM 연산 절약
- 단점: 압축 과정에서 정보 손실 가능

---

### 3.4 Perceiver Resampler (Flamingo 스타일)

여러 층의 Cross-Attention으로 반복적으로 정보를 추출합니다.

```python
class PerceiverAttentionBlock(nn.Module):
    """단일 Perceiver Attention Block"""
    
    def __init__(self, llm_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        self.norm3 = nn.LayerNorm(llm_dim)
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)
        )
    
    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-Attention: latents attend to context (vision features)
        attn_out, _ = self.cross_attn(latents, context, context)
        latents = self.norm1(latents + attn_out)
        
        # Self-Attention: latents attend to themselves
        self_attn_out, _ = self.self_attn(latents, latents, latents)
        latents = self.norm2(latents + self_attn_out)
        
        # FFN
        latents = self.norm3(latents + self.ffn(latents))
        
        return latents


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler (Flamingo 스타일)
    
    - 여러 층의 Cross-Attention으로 반복적 정제
    - Self-Attention으로 latent 간 상호작용
    - 파라미터 수: ~100M (depth에 따라 증가)
    """
    
    def __init__(
        self, 
        vision_dim: int = 1024, 
        llm_dim: int = 4096, 
        num_queries: int = 64,
        depth: int = 2,
        num_heads: int = 8
    ):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_queries, llm_dim))
        
        # Vision projection
        self.vision_proj = nn.Linear(vision_dim, llm_dim)
        
        # Perceiver layers
        self.layers = nn.ModuleList([
            PerceiverAttentionBlock(llm_dim, num_heads)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(llm_dim)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, num_tokens, vision_dim] 또는 [num_tokens, vision_dim]
            
        Returns:
            output: [batch, num_queries, llm_dim] 또는 [num_queries, llm_dim]
        """
        # Handle unbatched input
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = vision_features.shape[0]
        
        # Project vision features to llm_dim
        context = self.vision_proj(vision_features)  # [batch, num_tokens, llm_dim]
        
        # Initialize latents
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Pass through Perceiver layers
        for layer in self.layers:
            latents = layer(latents, context)
        
        # Final normalization
        output = self.final_norm(latents)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output

# 사용 예시
projector = PerceiverResampler(
    vision_dim=1024, 
    llm_dim=4096, 
    num_queries=64, 
    depth=2
)
vision_features = torch.randn(1, 4608, 1024)
projected = projector(vision_features)
print(projected.shape)  # [1, 64, 4096]
```

**특징**:
- 파라미터: ~100M
- 장점: 가장 풍부한 표현력, 반복적 정제로 정보 보존
- 단점: 학습 시간 증가, 메모리 사용량 높음

---

### 3.5 Projector 비교 요약

| Projector | 파라미터 | 입력 토큰 | 출력 토큰 | 특징 |
|-----------|----------|-----------|-----------|------|
| Linear | 4M | 4608 | 4608 | 가장 단순, baseline |
| MLP-2L | 21M | 4608 | 4608 | 비선형 변환, LLaVA 기본 |
| C-Abstractor | 50M | 4608 | 64 | 토큰 압축, 효율적 |
| Perceiver | 100M | 4608 | 64 | 최고 표현력, 반복 정제 |

---

## 4. LoRA vs DoRA 비교

### 4.1 LoRA (Low-Rank Adaptation)

```python
# LoRA 수식: W' = W + BA
# W: 원본 가중치 [out_features, in_features]
# B: [out_features, r]
# A: [r, in_features]
# r: rank (보통 8, 16, 32)

class LoRALayer(nn.Module):
    """LoRA 개념 설명용 구현"""
    
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
    
    def forward(self, x, original_linear):
        # 원본 출력
        original_output = original_linear(x)
        
        # LoRA 출력: x @ A.T @ B.T * scaling
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_output + lora_output
```

**LoRA 핵심 아이디어**:
- 원본 가중치 W는 동결
- 작은 rank의 행렬 A, B만 학습
- W' = W + BA로 가중치 업데이트 효과

### 4.2 DoRA (Weight-Decomposed Low-Rank Adaptation)

```python
# DoRA 수식: W' = m * (W + BA) / ||W + BA||
# m: learnable magnitude vector [out_features]
# 방향(direction)과 크기(magnitude)를 분리

class DoRALayer(nn.Module):
    """DoRA 개념 설명용 구현"""
    
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Low-rank matrices (LoRA와 동일)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Magnitude vector (DoRA 추가)
        self.magnitude = nn.Parameter(torch.ones(out_features))
    
    def forward(self, x, W):
        # LoRA delta
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        
        # Updated weight
        W_new = W + delta_W
        
        # Direction normalization
        W_norm = W_new.norm(dim=1, keepdim=True)
        W_direction = W_new / (W_norm + 1e-8)
        
        # Apply magnitude
        W_final = self.magnitude.unsqueeze(1) * W_direction
        
        return x @ W_final.T
```

**DoRA 핵심 아이디어**:
- LoRA의 확장: 방향과 크기를 분리
- magnitude 벡터로 각 출력 차원의 크기 학습
- full fine-tuning에 더 가까운 성능

### 4.3 PEFT 라이브러리 사용법

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

# DoRA 설정 (use_dora=True 추가)
dora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    use_dora=True,  # DoRA 활성화
)

# 모델에 적용
model = get_peft_model(base_model, lora_config)  # 또는 dora_config
model.print_trainable_parameters()
# 출력: trainable params: 26,738,688 || all params: 8,030,261,248 || trainable%: 0.33%
```

### 4.4 LoRA vs DoRA 비교

| 항목 | LoRA | DoRA |
|------|------|------|
| 파라미터 수 | ~26M | ~27M (magnitude 추가) |
| 학습 속도 | 빠름 | 약간 느림 |
| 성능 | 좋음 | 더 좋음 (full FT에 근접) |
| 안정성 | 안정적 | 더 안정적 |
| 사용 난이도 | 쉬움 | 쉬움 (use_dora=True) |

---

## 5. Forward Pass 전체 흐름

### 5.1 학습 시 Forward Pass

```python
class CustomVLM(nn.Module):
    """CLIP + Projector + Qwen3-8B Custom VLM"""
    
    def __init__(self, vision_encoder, projector, llm, tokenizer):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer
        
        # Vision encoder 동결
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        학습 시 forward pass
        
        Args:
            pixel_values: [batch, num_frames, 3, H, W] 비디오 프레임
            input_ids: [batch, seq_len] 텍스트 토큰
            attention_mask: [batch, seq_len] 어텐션 마스크
            labels: [batch, seq_len] 정답 레이블 (학습 시)
        
        Returns:
            loss 또는 logits
        """
        batch_size = pixel_values.shape[0]
        num_frames = pixel_values.shape[1]
        
        # 1. Vision Encoding (프레임별)
        vision_features_list = []
        for b in range(batch_size):
            frames = pixel_values[b]  # [num_frames, 3, H, W]
            with torch.no_grad():
                for f in range(num_frames):
                    frame = frames[f:f+1]  # [1, 3, H, W]
                    outputs = self.vision_encoder(pixel_values=frame)
                    patch_features = outputs.last_hidden_state[:, 1:, :]  # [1, 576, 1024]
                    vision_features_list.append(patch_features)
        
        # [batch * num_frames, 576, 1024] -> [batch, num_frames * 576, 1024]
        vision_features = torch.cat(vision_features_list, dim=1)
        vision_features = vision_features.view(batch_size, -1, vision_features.shape[-1])
        
        # 2. Projection
        projected = self.projector(vision_features)  # [batch, num_tokens, 4096]
        
        # 3. LLM Embedding
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [batch, seq_len, 4096]
        
        # 4. Vision + Text 결합
        # [batch, vision_tokens + text_tokens, 4096]
        combined_embeds = torch.cat([projected, text_embeds], dim=1)
        
        # 5. Attention mask 확장
        vision_attn = torch.ones(batch_size, projected.shape[1], device=attention_mask.device)
        combined_attn = torch.cat([vision_attn, attention_mask], dim=1)
        
        # 6. LLM Forward
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            labels=labels,  # Causal LM loss 계산
        )
        
        return outputs
```

### 5.2 데이터 흐름 시각화

```
입력:
  video: [8, 3, 336, 336]  (8 프레임)
  prompt: "이 영상을 설명해주세요."
  caption: "한 남자가 공원에서 강아지와 산책하고 있습니다."

Step 1: Vision Encoding (CLIP)
  [8, 3, 336, 336] → [8, 576, 1024]

Step 2: Flatten
  [8, 576, 1024] → [4608, 1024]

Step 3: Projection (예: MLP-2L)
  [4608, 1024] → [4608, 4096]
  
  또는 C-Abstractor:
  [4608, 1024] → [64, 4096]

Step 4: Text Embedding
  "이 영상을 설명해주세요." → [15, 4096]

Step 5: Concatenation
  [4608 + 15, 4096] = [4623, 4096]
  
  또는 C-Abstractor 사용 시:
  [64 + 15, 4096] = [79, 4096]

Step 6: LLM Forward (Qwen3-8B)
  [4623, 4096] → logits → loss

Step 7: Generation (추론 시)
  → "한 남자가 공원에서 강아지와 산책하고 있습니다."
```

---

## 6. 실험 설정

### 6.1 실험 매트릭스

| Exp | Projector | PEFT | Person | 우선순위 |
|-----|-----------|------|--------|----------|
| E1 | Linear | LoRA | A | 1 (Baseline) |
| E2 | Linear | DoRA | A | 5 |
| E3 | MLP-2L | LoRA | A | 2 (LLaVA 기준) |
| E4 | MLP-2L | DoRA | B | 3 |
| E5 | C-Abstractor | LoRA | B | 4 |
| E6 | C-Abstractor | DoRA | B | 6 |
| E7 | Perceiver | LoRA | C | 7 |
| E8 | Perceiver | DoRA | C | 8 |

### 6.2 공통 하이퍼파라미터

```yaml
# 모델
vision_encoder: openai/clip-vit-large-patch14-336
llm: Qwen/Qwen3-8B
load_in_4bit: true
torch_dtype: bfloat16

# Projector
vision_dim: 1024
llm_dim: 4096
num_queries: 64  # C-Abstractor, Perceiver용

# PEFT (LoRA/DoRA 공통)
r: 16
alpha: 32
dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Training
stage1_epochs: 2
stage1_lr: 1e-3
stage2_epochs: 5
stage2_lr: 5e-5
batch_size: 2
gradient_accumulation: 4
num_frames: 8
```

### 6.3 2-Stage Training

```
Stage 1: Projector Alignment
├── 목표: Vision 특징을 LLM 공간에 정렬
├── 학습: Projector만 (Vision Encoder, LLM 동결)
├── LR: 1e-3 (높음)
└── Epochs: 2

Stage 2: Joint Fine-tuning
├── 목표: 캡션 생성 태스크 최적화
├── 학습: Projector + LoRA/DoRA
├── LR: 5e-5 (낮음)
└── Epochs: 5
```

---

## 참고 자료

- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [LLaVA-1.5 Paper](https://arxiv.org/abs/2310.03744)
- [Flamingo Paper](https://arxiv.org/abs/2204.14198)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DoRA Paper](https://arxiv.org/abs/2402.09353)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/)
