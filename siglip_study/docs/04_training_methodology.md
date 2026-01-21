# 학습 방법론: 2-Stage Training & LoRA

> **학습 목표**: 2-Stage 학습의 이유, LoRA/DoRA 원리 이해

---

## 1. 왜 2-Stage Training인가?

### 1.1 문제 상황

VLM을 한 번에 학습하면:

```
Vision Encoder (Frozen)
        │
        ▼
   Projector (Random Init) ──▶ 잘못된 출력
        │
        ▼
   LLM (LoRA) ──▶ 잘못된 입력으로 학습
        │
        ▼
   Caption ──▶ 품질 저하
```

**문제**: 
- Projector가 초기에 무작위 값 출력
- LLM이 "쓰레기" 입력으로 학습
- 서로 잘못된 방향으로 최적화

### 1.2 해결책: 2-Stage Training

```
Stage 1: Projector Warm-up
─────────────────────────
Vision Encoder (Frozen)
        │
        ▼
   Projector (Training) ──▶ LLM 공간 학습
        │
        ▼
   LLM (Frozen)


Stage 2: Joint Fine-tuning
─────────────────────────
Vision Encoder (Frozen)
        │
        ▼
   Projector (Training) ──▶ 함께 최적화
        │
        ▼
   LLM (LoRA Training)
```

---

## 2. Stage 1: Projector Warm-up

### 2.1 목적

**Projector가 LLM이 이해할 수 있는 출력을 생성하도록 학습**

```
Before Stage 1:
  Projector 출력: [0.23, -1.5, 0.8, ...] (무작위)
  LLM 해석: ???
  
After Stage 1:
  Projector 출력: [0.12, 0.45, -0.23, ...] (의미 있는 벡터)
  LLM 해석: "이건 건물 같은데..."
```

### 2.2 학습 설정

```python
# Stage 1 Configuration
stage1_config = {
    "epochs": 2,
    "learning_rate": 1e-3,   # 상대적으로 높은 LR
    "trainable": ["projector"],
    "frozen": ["vision_encoder", "llm"]
}
```

### 2.3 Loss 함수

LLM의 Language Modeling Loss 사용:

$$
\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(y_t | y_{<t}, \mathbf{H}_v)
$$

- $y_t$: t번째 토큰 (정답 캡션)
- $\mathbf{H}_v$: Projector 출력 (Vision 정보)
- LLM은 frozen이지만 gradient는 Projector로 전파

### 2.4 코드 구현

```python
def train_stage1(model, train_loader, config, device):
    print("Stage 1: Projector Warm-up")
    
    # LLM freeze
    for param in model.llm.parameters():
        param.requires_grad = False
    
    # Projector만 학습
    for param in model.projector.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.projector.parameters(), 
        lr=config["stage1_lr"]  # 1e-3
    )
    
    for epoch in range(config["stage1_epochs"]):
        for batch in train_loader:
            outputs = model(
                batch["pixel_values"],
                batch["input_ids"],
                batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

---

## 3. Stage 2: Joint Fine-tuning with LoRA

### 3.1 목적

**Projector와 LLM이 함께 최적화되어 더 나은 캡션 생성**

### 3.2 LoRA (Low-Rank Adaptation)

#### 핵심 아이디어

LLM의 가중치를 직접 수정하지 않고, **저랭크 행렬**을 추가합니다.

```
기존:
  h = Wx

LoRA 적용:
  h = Wx + BAx
      ↑     ↑
   원본   저랭크 업데이트
  (Frozen)  (Trainable)
```

#### 수식

$$
\mathbf{W}' = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}
$$

- $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$: 원본 가중치 (frozen)
- $\mathbf{B} \in \mathbb{R}^{d \times r}$: 저랭크 행렬 (trainable)
- $\mathbf{A} \in \mathbb{R}^{r \times k}$: 저랭크 행렬 (trainable)
- $r \ll \min(d, k)$: 랭크 (보통 8~64)

#### 파라미터 절약

```
원본: d × k = 4096 × 4096 = 16,777,216 (16.7M)

LoRA (r=16):
  B: 4096 × 16 = 65,536
  A: 16 × 4096 = 65,536
  Total: 131,072 (0.13M)
  
절약: 99.2%!
```

#### LoRA 적용 위치

Transformer의 Attention 레이어:

```
┌─────────────────────────────────────────────┐
│              Attention Layer                 │
├─────────────────────────────────────────────┤
│  Query:  q = W_q·x + B_q·A_q·x  ← LoRA     │
│  Key:    k = W_k·x + B_k·A_k·x  ← LoRA     │
│  Value:  v = W_v·x + B_v·A_v·x  ← LoRA     │
│  Output: o = W_o·attn + B_o·A_o·attn ← LoRA│
└─────────────────────────────────────────────┘
```

추가로 FFN 레이어에도 적용:

```
┌─────────────────────────────────────────────┐
│                FFN Layer                     │
├─────────────────────────────────────────────┤
│  Gate:   g = W_gate·x + B_gate·A_gate·x     │
│  Up:     u = W_up·x + B_up·A_up·x           │
│  Down:   d = W_down·h + B_down·A_down·h     │
└─────────────────────────────────────────────┘
```

### 3.3 LoRA 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `r` | 16 | 랭크 (작을수록 효율적, 클수록 표현력↑) |
| `lora_alpha` | 32 | 스케일링 팩터 |
| `lora_dropout` | 0.05 | 드롭아웃 비율 |
| `target_modules` | qkvo + gate/up/down | 적용 레이어 |

#### Alpha와 Rank의 관계

$$
\Delta\mathbf{W} = \frac{\alpha}{r} \mathbf{B}\mathbf{A}
$$

- `alpha/r`가 실제 스케일링
- `alpha=32, r=16` → 스케일 = 2
- 큰 alpha = 더 강한 업데이트

### 3.4 코드 구현

```python
from peft import LoraConfig, get_peft_model, TaskType

def apply_lora(llm, config):
    lora_config = LoraConfig(
        r=config["lora_r"],                    # 16
        lora_alpha=config["lora_alpha"],       # 32
        lora_dropout=config["lora_dropout"],   # 0.05
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"       # FFN
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()
    # 출력: trainable params: 41,943,040 || all params: 8,030,261,248 || 0.52%
    
    return llm
```

### 3.5 Stage 2 학습

```python
def train_stage2(model, train_loader, val_loader, config, device):
    print("Stage 2: Projector + LoRA Fine-tuning")
    
    # Projector + LoRA 모두 학습
    trainable_params = [
        {"params": model.projector.parameters(), "lr": config["stage2_lr"]},
        {"params": [p for p in model.llm.parameters() if p.requires_grad], 
         "lr": config["stage2_lr"]}
    ]
    
    optimizer = torch.optim.AdamW(trainable_params)
    
    for epoch in range(config["stage2_epochs"]):
        # Training
        for batch in train_loader:
            outputs = model(...)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation
        metrics = evaluate_model(model, val_loader, ...)
        print(f"Epoch {epoch+1} - SigLIP: {metrics['siglip_score']:.4f}")
```

---

## 4. DoRA (Weight-Decomposed Low-Rank Adaptation)

### 4.1 LoRA의 한계

LoRA는 가중치 변화의 **크기(magnitude)**와 **방향(direction)**을 분리하지 않습니다.

### 4.2 DoRA 아이디어

가중치를 크기와 방향으로 분해:

$$
\mathbf{W}' = m \cdot \frac{\mathbf{W}_0 + \mathbf{B}\mathbf{A}}{||\mathbf{W}_0 + \mathbf{B}\mathbf{A}||}
$$

- $m$: learnable magnitude (스칼라)
- $\frac{\mathbf{W}}{||\mathbf{W}||}$: 방향 벡터 (단위 벡터)

### 4.3 직관적 이해

```
LoRA:
  W' = W + BA
  → 크기와 방향이 동시에 변함

DoRA:
  W' = magnitude × direction
  → 각각 독립적으로 조절 가능
  → 더 세밀한 조정
```

### 4.4 DoRA 구현

```python
from peft import LoraConfig, get_peft_model

dora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,  # DoRA 활성화
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

llm = get_peft_model(llm, dora_config)
```

### 4.5 LoRA vs DoRA 비교

| 항목 | LoRA | DoRA |
|------|------|------|
| 파라미터 | BA | BA + m |
| 학습 대상 | 방향+크기 혼합 | 방향과 크기 분리 |
| 성능 | 좋음 | 더 좋음 (일부 태스크) |
| 복잡도 | 낮음 | 약간 높음 |

---

## 5. Gradient Checkpointing

### 5.1 메모리 문제

LLM 학습 시 활성화 값(activations)이 메모리를 많이 차지합니다.

```
Forward Pass:
  Layer 1 → save activations (2GB)
  Layer 2 → save activations (2GB)
  ...
  Layer 32 → save activations (2GB)
  
Total: 64GB just for activations!
```

### 5.2 해결책: Gradient Checkpointing

일부 활성화만 저장하고, backward 시 재계산:

```
Forward Pass (with checkpointing):
  Layer 1 → save ✓
  Layer 2 → don't save
  Layer 3 → don't save
  Layer 4 → save ✓
  ...
  
Backward Pass:
  Need Layer 3 activations?
  → Recompute from Layer 1 checkpoint
```

### 5.3 Trade-off

| 항목 | Without | With Checkpointing |
|------|---------|-------------------|
| 메모리 | 높음 | **낮음** |
| 속도 | 빠름 | 약간 느림 (~20%) |

### 5.4 코드 구현

```python
# Gradient Checkpointing 활성화
llm.config.use_cache = False  # 필수!
llm.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

**주의**: `use_cache=True`와 gradient checkpointing은 호환 안 됨!

---

## 6. 4-bit 양자화

### 6.1 왜 양자화인가?

Qwen3-8B 메모리 요구량:

| Precision | 메모리 |
|-----------|--------|
| FP32 | ~32GB |
| FP16 | ~16GB |
| **4-bit** | **~4GB** |

### 6.2 NF4 (Normal Float 4-bit)

bitsandbytes 라이브러리의 양자화 방식:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # 이중 양자화
)
```

### 6.3 QLoRA

4-bit 양자화 + LoRA = QLoRA

```
4-bit 양자화된 LLM (Frozen, 4GB)
        │
        ├── LoRA Adapter (FP16, ~100MB)
        │   └── Trainable!
        │
        ▼
    Output
```

---

## 7. 전체 학습 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Model Loading                                            │
│     ├── Vision Encoder (CLIP-ViT, Frozen)                   │
│     ├── LLM (Qwen3-8B, 4-bit Quantized)                     │
│     └── Projector (Random Init)                              │
│                                                              │
│  2. LoRA Application                                         │
│     └── Apply LoRA to LLM (r=16, alpha=32)                  │
│                                                              │
│  3. Gradient Checkpointing                                   │
│     └── Enable for memory efficiency                         │
│                                                              │
│  4. Stage 1: Projector Warm-up                              │
│     ├── Freeze: LLM                                          │
│     ├── Train: Projector only                                │
│     ├── Epochs: 2                                            │
│     └── LR: 1e-3                                             │
│                                                              │
│  5. Stage 2: Joint Fine-tuning                              │
│     ├── Train: Projector + LoRA                              │
│     ├── Epochs: 3                                            │
│     ├── LR: 5e-5                                             │
│     └── Eval: SigLIP Score per epoch                         │
│                                                              │
│  6. Checkpoint Saving                                        │
│     ├── Best model (highest SigLIP)                          │
│     └── Per-epoch checkpoints (resume 지원)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 하이퍼파라미터 요약

| 항목 | Stage 1 | Stage 2 |
|------|---------|---------|
| Epochs | 2 | 3 |
| Learning Rate | 1e-3 | 5e-5 |
| Trainable | Projector | Projector + LoRA |
| Batch Size | 4 | 4 |
| Grad Accumulation | 2 | 2 |
| Effective Batch | 8 | 8 |

### LoRA 설정

| 파라미터 | 값 |
|----------|-----|
| r | 16 |
| alpha | 32 |
| dropout | 0.05 |
| target_modules | q,k,v,o_proj + gate,up,down_proj |

---

## 참고 자료

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
- [DoRA Paper](https://arxiv.org/abs/2402.09353) - Weight-Decomposed Low-Rank Adaptation
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Efficient Finetuning of Quantized LLMs
- [LLaVA Training](https://github.com/haotian-liu/LLaVA) - 2-Stage Training 원조
