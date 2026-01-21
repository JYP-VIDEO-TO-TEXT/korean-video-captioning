# 03. 학습 전략 상세

## 개요

본 문서에서는 대한민국 배경영상 캡셔닝 모델의 학습 전략을 상세히 설명합니다. LoRA 기반 효율적 학습, 2-Stage Training, 정규화 기법, 학습률 스케줄링, 적응적 프레임 샘플링 등을 다룹니다.

---

## 1. 학습 패러다임: Parameter-Efficient Fine-Tuning

### 1.1 Full Fine-Tuning vs PEFT

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fine-Tuning 방식 비교                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Full Fine-Tuning          PEFT (LoRA)                         │
│   ┌─────────────┐           ┌─────────────┐                     │
│   │ ██████████ │           │ ░░░░░░░░██ │                      │
│   │ ██████████ │           │ ░░░░░░░░░░ │                      │
│   │ ██████████ │           │ ░░░░░░░░██ │                      │
│   │ ██████████ │           │ ░░░░░░░░░░ │                      │
│   └─────────────┘           └─────────────┘                     │
│   모든 파라미터 학습         일부 파라미터만 학습                 │
│   ~7B params                ~10M params (0.14%)                 │
│   메모리: ~56GB             메모리: ~8GB (4bit)                 │
│                                                                  │
│   ██ = 학습 파라미터                                            │
│   ░░ = 동결 파라미터                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 PEFT 선택 이유

| 요소 | Full Fine-Tuning | LoRA (PEFT) |
|------|------------------|-------------|
| 메모리 사용량 | ~56GB (FP16) | ~8GB (4bit+LoRA) |
| 학습 시간 | 매우 김 | 빠름 |
| 과적합 위험 | 높음 (작은 데이터셋) | 낮음 |
| 원본 모델 보존 | 불가 | 가능 (adapter만 저장) |
| **Colab 적합성** | T4 불가 | T4 가능 |

---

## 2. LoRA (Low-Rank Adaptation) 상세

### 2.1 LoRA 개념

```
LoRA 핵심 아이디어:
- 사전학습된 가중치 W는 동결
- 저랭크(low-rank) 행렬 A, B만 학습
- 출력: h = Wx + BAx

수학적 표현:
W' = W + ΔW = W + BA

여기서:
- W: 원본 가중치 [d_out, d_in]
- B: 저랭크 행렬 [d_out, r]
- A: 저랭크 행렬 [r, d_in]
- r: rank (보통 8, 16, 32, 64)

파라미터 수 비교:
- 원본: d_out × d_in = 4096 × 4096 = 16.7M
- LoRA: d_out × r + r × d_in = 4096 × 32 + 32 × 4096 = 262K
- 감소율: 98.4%
```

### 2.2 LoRA 수식 및 구현

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    LoRA를 적용한 Linear 레이어
    
    출력 = W @ x + (B @ A) @ x * (alpha / r)
    
    초기화:
    - A: Kaiming Uniform (학습 시작부터 기여)
    - B: Zero (초기에는 원본 모델과 동일)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA 행렬
        self.lora_A = nn.Parameter(
            torch.zeros(r, original_layer.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(original_layer.out_features, r)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 원본 가중치 동결
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x):
        # 원본 출력
        original_output = self.original_layer(x)
        
        # LoRA 출력
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
```

### 2.3 LoRA Rank 선택 가이드

| Rank (r) | 파라미터 수 | 표현력 | 과적합 위험 | 권장 상황 |
|----------|------------|--------|------------|-----------|
| **8** | ~2.6M | 낮음 | 매우 낮음 | T4, 작은 데이터셋 |
| **16** | ~5.2M | 중간 | 낮음 | L4, 일반적 상황 |
| **32** | ~10.5M | 높음 | 중간 | A100, 충분한 데이터 |
| **64** | ~21M | 매우 높음 | 높음 | H100, 대규모 데이터 |

**선택 기준:**

```python
def select_lora_rank(dataset_size, gpu_vram):
    """
    데이터셋 크기와 GPU VRAM에 따른 LoRA rank 선택
    """
    if gpu_vram <= 16:  # T4
        return 8
    elif gpu_vram <= 24:  # L4
        return 16
    elif gpu_vram <= 40:  # A100 40GB
        if dataset_size < 5000:
            return 16
        else:
            return 32
    else:  # H100 80GB
        if dataset_size < 10000:
            return 32
        else:
            return 64
```

### 2.4 Target Modules 선택

```python
# 권장 target_modules 구성

# 기본 (메모리 효율):
basic_target_modules = [
    "q_proj",  # Query projection
    "k_proj",  # Key projection
    "v_proj",  # Value projection
    "o_proj",  # Output projection
]

# 확장 (성능 향상):
extended_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",  # MLP gate
    "up_proj",    # MLP up
    "down_proj",  # MLP down
]

# 선택 이유:
# - Attention 레이어: 시퀀스 간 관계 학습
# - MLP 레이어: 비선형 변환 학습
# - Attention만으로도 충분한 경우 많음
```

**모듈별 역할:**

```
Transformer Layer 구조:

Input
  │
  ▼
┌─────────────────────────────────────┐
│         Self-Attention              │
│  ┌─────┬─────┬─────┬─────┐         │
│  │Q_proj│K_proj│V_proj│O_proj│ ◀── LoRA 적용 │
│  └─────┴─────┴─────┴─────┘         │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│            MLP (FFN)                │
│  ┌──────────┬─────────┬──────────┐ │
│  │gate_proj │up_proj  │down_proj │ ◀── LoRA 적용 (선택적) │
│  └──────────┴─────────┴──────────┘ │
└─────────────────────────────────────┘
  │
  ▼
Output
```

### 2.5 Alpha 설정

```python
# Alpha 설정 가이드

# 일반적인 규칙: alpha = 2 * r
# 이유: scaling factor = alpha / r = 2 (적절한 학습 신호)

examples = {
    "r=8":  {"alpha": 16, "scaling": 2.0},
    "r=16": {"alpha": 32, "scaling": 2.0},
    "r=32": {"alpha": 64, "scaling": 2.0},
    "r=64": {"alpha": 128, "scaling": 2.0},
}

# 실험적 조정:
# - 학습이 느리면: alpha 증가 (scaling 증가)
# - 학습이 불안정하면: alpha 감소 (scaling 감소)
```

---

## 3. 2-Stage Training

### 3.1 2-Stage Training 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                    2-Stage Training                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Vision-Language Alignment (정렬)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  Vision    [FROZEN]                                        │  │
│  │  Encoder ─────────▶ Projector ──▶ LLM [FROZEN]           │  │
│  │                       [TRAIN]                              │  │
│  │                                                            │  │
│  │  목적: 시각 특징을 언어 공간으로 매핑 학습                  │  │
│  │  학습 대상: Projector only                                 │  │
│  │  Epoch: 1                                                  │  │
│  │  LR: 높음 (2e-4)                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  Stage 2: Task-Specific Fine-tuning (미세조정)                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  Vision    [FROZEN]                                        │  │
│  │  Encoder ─────────▶ Projector ──▶ LLM [LoRA]             │  │
│  │                       [TRAIN]     [TRAIN]                  │  │
│  │                                                            │  │
│  │  목적: 한국어 캡셔닝 태스크에 특화                          │  │
│  │  학습 대상: Projector + LLM LoRA                           │  │
│  │  Epoch: 4                                                  │  │
│  │  LR: 낮음 (1e-4)                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Stage별 설정

```python
# Stage 1 설정
stage1_config = {
    "name": "vision_language_alignment",
    "epochs": 1,
    "learning_rate": 2e-4,
    "freeze_vision_encoder": True,
    "freeze_llm": True,  # LLM 완전 동결
    "trainable_modules": ["multi_modal_projector"],
    "warmup_ratio": 0.1,  # 10% warmup
}

# Stage 2 설정
stage2_config = {
    "name": "task_specific_finetuning",
    "epochs": 4,
    "learning_rate": 1e-4,  # 더 낮은 LR
    "freeze_vision_encoder": True,
    "freeze_llm": False,  # LLM LoRA 학습
    "trainable_modules": ["multi_modal_projector", "lora"],
    "warmup_ratio": 0.03,  # 3% warmup
}
```

### 3.3 2-Stage Training 선택 이유

| 이점 | 설명 |
|------|------|
| **안정적 학습** | Projector 먼저 학습으로 급격한 변화 방지 |
| **효율적 수렴** | Stage 1에서 기본 정렬 완료 |
| **과적합 방지** | 단계별 학습으로 일반화 향상 |
| **디버깅 용이** | 문제 발생 시 단계별 분석 가능 |

### 3.4 구현 예시

```python
def train_two_stage(model, train_dataset, config):
    """2-Stage Training 구현"""
    
    # ========== Stage 1: Vision-Language Alignment ==========
    print("=" * 50)
    print("Stage 1: Vision-Language Alignment")
    print("=" * 50)
    
    # Projector만 학습 가능하게 설정
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Stage 1 학습
    stage1_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="outputs/stage1",
            num_train_epochs=1,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        ),
        train_dataset=train_dataset,
    )
    stage1_trainer.train()
    
    # ========== Stage 2: Task-Specific Fine-tuning ==========
    print("=" * 50)
    print("Stage 2: Task-Specific Fine-tuning")
    print("=" * 50)
    
    # LoRA 적용
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    
    # Projector도 계속 학습
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True
    
    # Stage 2 학습
    stage2_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="outputs/stage2",
            num_train_epochs=4,
            learning_rate=1e-4,
            warmup_ratio=0.03,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        ),
        train_dataset=train_dataset,
    )
    stage2_trainer.train()
    
    return model
```

---

## 4. 정규화 기법

### 4.1 정규화 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    정규화 기법 요약                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  입력                                                            │
│    │                                                             │
│    ▼                                                             │
│  [Dropout] ────────── 무작위 뉴런 비활성화                       │
│    │                                                             │
│    ▼                                                             │
│  [Model] ─────────── Weight Decay 적용                          │
│    │                                                             │
│    ▼                                                             │
│  [Gradient] ────────── Gradient Clipping                        │
│    │                                                             │
│    ▼                                                             │
│  [Loss] ─────────── Label Smoothing (선택적)                    │
│    │                                                             │
│    ▼                                                             │
│  출력                                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Dropout

```python
# LoRA Dropout 설정

# 개념:
# 학습 시 무작위로 뉴런을 비활성화하여 과적합 방지

# 권장값:
dropout_guide = {
    "small_dataset": 0.1,   # < 5K 샘플
    "medium_dataset": 0.05, # 5K-20K 샘플
    "large_dataset": 0.01,  # > 20K 샘플
}

# 우리 데이터셋 (10K):
lora_dropout = 0.05  # 권장

# 적용 위치:
# - LoRA 행렬 A의 출력에 적용
# - 원본 가중치에는 적용하지 않음
```

### 4.3 Weight Decay

```python
# Weight Decay (L2 정규화)

# 개념:
# 손실 함수에 가중치 크기 페널티 추가
# L = L_original + λ * ||W||²

# 권장값:
weight_decay = 0.01  # AdamW 기본값

# 효과:
# - 가중치가 너무 커지는 것 방지
# - 일반화 성능 향상
# - LoRA에서는 B 행렬에만 적용 (A는 제외 가능)

# AdamW에서의 구현:
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,  # L2 정규화
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### 4.4 Gradient Clipping

```python
# Gradient Clipping

# 개념:
# 그래디언트 norm이 임계값을 초과하면 스케일링

# 수학적 표현:
# if ||g|| > max_norm:
#     g = g * (max_norm / ||g||)

# 권장값:
max_grad_norm = 1.0

# 구현:
def clip_gradients(model, max_norm):
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm
    )
    return total_norm

# 효과:
# - 학습 안정성 향상 (gradient explosion 방지)
# - 특히 긴 시퀀스에서 중요
```

### 4.5 Label Smoothing

```python
# Label Smoothing

# 개념:
# Hard label (one-hot) 대신 soft label 사용
# 정답 확률: 1 - ε + ε/K
# 오답 확률: ε/K

# 예시 (K=5, ε=0.1):
# Hard: [0, 0, 1, 0, 0]
# Soft: [0.02, 0.02, 0.92, 0.02, 0.02]

# 구현:
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, vocab_size=32000):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        
    def forward(self, logits, targets):
        # logits: [batch, seq_len, vocab_size]
        # targets: [batch, seq_len]
        
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(
            log_probs.view(-1, self.vocab_size),
            targets.view(-1),
            reduction='none'
        )
        smooth_loss = -log_probs.mean(dim=-1).view(-1)
        
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# 권장값:
label_smoothing = 0.1  # 과적합 방지에 효과적

# 주의:
# - 메모리 제약 시 비활성화 (T4)
# - 데이터셋이 충분히 크면 불필요할 수 있음
```

---

## 5. Learning Rate Schedule

### 5.1 Warmup의 중요성

```
학습률 스케줄 (Warmup + Cosine Decay):

LR
│
│     ╭──────╮
│    ╱        ╲
│   ╱          ╲
│  ╱            ╲
│ ╱              ╲
│╱                ╲────
├─┬────────────────┬──▶ Step
│ │                │
│ Warmup          Decay
│ (3-10%)         (나머지)
```

### 5.2 Warmup 설정

```python
# Warmup 설정

# 개념:
# 학습 초기에 작은 LR에서 시작하여 점진적으로 증가

# 이유:
# 1. 초기 그래디언트 불안정 방지
# 2. 사전학습 가중치 급격한 변화 방지
# 3. AdamW의 running average 초기화 시간 확보

# 권장값:
warmup_configs = {
    "stage1": {"warmup_ratio": 0.10},  # 10% (정렬 단계)
    "stage2": {"warmup_ratio": 0.03},  # 3% (미세조정 단계)
}

# 계산 예시:
total_steps = 1000
warmup_ratio = 0.03
warmup_steps = int(total_steps * warmup_ratio)  # 30 steps
```

### 5.3 Cosine Annealing

```python
# Cosine Annealing Scheduler

# 수학적 표현:
# lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))

# 장점:
# - Smooth decay (급격한 변화 없음)
# - 학습 후반에도 적절한 LR 유지
# - 지역 최소점 탈출 가능성

import torch.optim.lr_scheduler as lr_scheduler

scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,  # 전체 스텝 수
    eta_min=1e-6        # 최소 LR
)

# HuggingFace Trainer에서:
training_args = TrainingArguments(
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    learning_rate=1e-4,
)
```

### 5.4 Cosine with Restarts

```python
# Cosine Annealing with Warm Restarts

# 개념:
# 주기적으로 LR을 다시 높여서 새로운 최적점 탐색

# 수학적 표현:
# lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * (t % T_i) / T_i))

# 장점:
# - 지역 최소점 탈출 용이
# - 더 다양한 해 탐색
# - 긴 학습에서 효과적

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=200,          # 첫 주기 길이
    T_mult=2,         # 주기 증가율 (200 -> 400 -> 800)
    eta_min=1e-6
)

# HuggingFace Trainer에서:
training_args = TrainingArguments(
    lr_scheduler_type="cosine_with_restarts",
    lr_scheduler_kwargs={"num_cycles": 2},  # restart 횟수
)
```

### 5.5 Learning Rate 선택 가이드

```python
# GPU별 권장 Learning Rate

lr_guide = {
    "T4": {
        "base_lr": 1e-4,
        "reason": "작은 배치로 안정적 학습 필요"
    },
    "L4": {
        "base_lr": 1e-4,
        "reason": "기본값, 안정적"
    },
    "A100": {
        "base_lr": 2e-4,
        "reason": "큰 배치로 더 높은 LR 가능"
    },
    "H100": {
        "base_lr": 2e-4,
        "reason": "최대 효율, 빠른 수렴"
    },
}

# Effective Batch Size에 따른 LR 조정:
# Linear Scaling Rule: lr = base_lr * (effective_batch_size / 16)
def adjusted_lr(base_lr, batch_size, gradient_accumulation):
    effective_batch = batch_size * gradient_accumulation
    reference_batch = 16
    return base_lr * (effective_batch / reference_batch)
```

---

## 6. 적응적 프레임 샘플링

### 6.1 샘플링 전략 비교

```
┌─────────────────────────────────────────────────────────────────┐
│                    프레임 샘플링 전략                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Uniform Sampling (균등 샘플링)                                 │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐                           │
│  │●│ │ │ │●│ │ │ │●│ │ │ │●│ │ │ │                           │
│  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘                           │
│  장점: 단순, 일관적                                             │
│  단점: 중요 장면 놓칠 수 있음                                   │
│                                                                  │
│  Keyframe Sampling (키프레임 샘플링)                            │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐                           │
│  │●│ │●│ │ │ │●│ │ │ │ │●│ │ │ │ │                           │
│  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘                           │
│        ↑         ↑           ↑                                  │
│      장면      장면        장면                                  │
│      변화      변화        변화                                  │
│  장점: 정보량 최대화                                            │
│  단점: 계산 비용                                                │
│                                                                  │
│  Adaptive Sampling (적응적 샘플링)                              │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐                           │
│  │●│●│ │ │●│ │ │●│●│ │ │ │●│ │ │●│                           │
│  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘                           │
│   동적      정적      동적    정적                              │
│   영역      영역      영역    영역                              │
│  장점: 최적 정보량                                              │
│  단점: 가장 복잡                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 장면 변화 감지 알고리즘

```python
import cv2
import numpy as np

def detect_scene_changes(video_path, threshold=30.0):
    """
    히스토그램 차이 기반 장면 변화 감지
    
    알고리즘:
    1. 각 프레임을 HSV로 변환
    2. 색상 히스토그램 계산
    3. 연속 프레임 간 히스토그램 비교
    4. 차이가 threshold 초과 시 장면 변화로 판정
    """
    cap = cv2.VideoCapture(video_path)
    scene_changes = []
    prev_hist = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 히스토그램 계산
        hist = cv2.calcHist(
            [hsv], [0, 1], None, 
            [50, 60], [0, 180, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        
        if prev_hist is not None:
            # 히스토그램 비교 (Bhattacharyya 거리)
            diff = cv2.compareHist(
                prev_hist, hist, 
                cv2.HISTCMP_BHATTACHARYYA
            )
            
            if diff > threshold / 100:
                scene_changes.append(frame_idx)
        
        prev_hist = hist
        frame_idx += 1
    
    cap.release()
    return scene_changes


def adaptive_frame_sampling(video_path, num_frames=8, method="scene_change"):
    """
    적응적 프레임 샘플링
    
    Parameters:
    - video_path: 비디오 파일 경로
    - num_frames: 추출할 프레임 수
    - method: "uniform", "scene_change", "optical_flow"
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if method == "uniform":
        # 균등 샘플링
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
    elif method == "scene_change":
        # 장면 변화 기반 샘플링
        scene_changes = detect_scene_changes(video_path)
        
        if len(scene_changes) >= num_frames:
            # 장면 변화 지점에서 균등하게 선택
            indices = np.array(scene_changes)[
                np.linspace(0, len(scene_changes)-1, num_frames, dtype=int)
            ]
        else:
            # 장면 변화 + 균등 샘플링 보완
            indices = list(scene_changes)
            remaining = num_frames - len(indices)
            uniform_indices = np.linspace(0, total_frames-1, remaining+2, dtype=int)[1:-1]
            indices.extend(uniform_indices)
            indices = sorted(set(indices))[:num_frames]
            
    elif method == "optical_flow":
        # Optical Flow 기반 (동적 영역 중심)
        indices = optical_flow_sampling(video_path, num_frames)
    
    return extract_frames_at_indices(video_path, indices)
```

### 6.3 GPU별 프레임 수 권장

| GPU | 권장 프레임 수 | 이유 |
|-----|---------------|------|
| T4 | 4 | 메모리 제약, 576*4=2304 토큰 |
| L4 | 8 | 균형, 576*8=4608 토큰 |
| A100 | 8-12 | 충분한 메모리 |
| H100 | 12-16 | 최대 정보량, 576*16=9216 토큰 |

---

## 7. Early Stopping

### 7.1 Early Stopping 개념

```
Validation Loss 추이:

Loss
│
│  ●
│   ●
│    ●
│     ●  ●
│      ●   ●
│            ●  ●  ●  ●  ← Patience 초과 (학습 중단)
│                    │
├────────────────────┼──▶ Epoch
│                    │
│              Best Model
│              (저장됨)
```

### 7.2 구현

```python
class EarlyStopping:
    """
    Early Stopping 구현
    
    Parameters:
    - patience: 개선 없이 기다리는 epoch 수
    - threshold: 개선으로 인정하는 최소 변화량
    - mode: 'min' (loss) 또는 'max' (accuracy)
    """
    
    def __init__(self, patience=5, threshold=0.0001, mode='min'):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.threshold
        else:
            improved = score > self.best_score + self.threshold
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

# 사용 예시
early_stopping = EarlyStopping(patience=5, threshold=0.0001)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 7.3 권장 설정

```python
early_stopping_config = {
    "patience": 5,           # 5 epoch 동안 개선 없으면 중단
    "threshold": 0.0001,     # 0.01% 이상 개선되어야 인정
    "metric": "eval_loss",   # 모니터링할 메트릭
    "mode": "min",           # 작을수록 좋음
    "save_best": True,       # 최고 모델 저장
}

# HuggingFace Trainer에서:
training_args = TrainingArguments(
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Early stopping은 EarlyStoppingCallback으로 추가
)

from transformers import EarlyStoppingCallback

trainer = Trainer(
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.0001
        )
    ]
)
```

---

## 8. GPU별 학습 설정 요약

### 8.1 T4 (16GB)

```python
t4_training_config = {
    # 모델
    "load_in_4bit": True,
    "gradient_checkpointing": True,
    
    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    
    # 배치
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_frames": 4,
    
    # 학습률
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.05,
    
    # 정규화
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.0,  # 메모리 절약
    
    # Early Stopping
    "patience": 5,
    
    # 기타
    "fp16": True,
    "bf16": False,
}
```

### 8.2 L4 (24GB)

```python
l4_training_config = {
    "load_in_4bit": True,
    "gradient_checkpointing": True,
    
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "num_frames": 8,
    
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.1,
    
    "patience": 5,
    
    "fp16": True,
    "bf16": False,
}
```

### 8.3 A100 (40GB)

```python
a100_training_config = {
    "load_in_4bit": True,
    "gradient_checkpointing": False,  # 속도 우선
    
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_frames": 8,
    
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.1,
    
    "patience": 5,
    
    "fp16": False,
    "bf16": True,  # A100은 bf16 지원
}
```

### 8.4 H100 (80GB)

```python
h100_training_config = {
    "load_in_4bit": True,
    "gradient_checkpointing": False,
    
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_frames": 16,
    
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine_with_restarts",
    "warmup_ratio": 0.03,
    
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.1,
    
    "patience": 5,
    
    "fp16": False,
    "bf16": True,
}
```

---

## 9. 참고 자료

### 논문
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)

### 코드 레포지토리
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
