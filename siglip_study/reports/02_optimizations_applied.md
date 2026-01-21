# 적용된 최적화 기법

## 개요

v2/v3 노트북에 적용된 최적화는 크게 두 가지 목적으로 분류됩니다:

1. **학습 품질 개선**: 안정적인 학습 및 Mode Collapse 탐지
2. **학습 시간 단축**: 리소스 효율적 학습

---

## 1. 학습 품질 개선

### 1.1 SigLIP2 (다국어 지원)

**변경 전:**
```python
model = "google/siglip-so400m-patch14-384"  # 영어 중심
```

**변경 후:**
```python
model = "google/siglip2-so400m-patch14-384"  # 100+ 언어 지원
```

**효과:**
- 한국어 캡션에 대한 정확한 Vision-Language 정렬 평가 가능
- 영어 중심 모델에서 발생하던 낮은 점수 문제 해결

---

### 1.2 LR Scheduler (Warmup + Cosine Decay)

**코드:**
```python
from transformers import get_cosine_schedule_with_warmup

total_steps = len(train_loader) * epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 매 step 후
scheduler.step()
```

**효과:**
- Warmup: 학습 초기 불안정 방지
- Cosine Decay: 후반부 수렴 개선
- Mode Collapse 예방에 도움 (일부)

---

### 1.3 Gradient Clipping

**코드:**
```python
max_grad_norm = 1.0

scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
scaler.step(optimizer)
```

**효과:**
- Gradient Explosion 방지
- 큰 모델에서 학습 안정성 향상
- 갑작스러운 loss 증가 방지

---

### 1.4 Diversity Monitoring

**코드:**
```python
def compute_diversity(captions):
    """캡션 다양성 계산 (0.0 ~ 1.0)"""
    if not captions:
        return 0.0
    unique = len(set(captions))
    return unique / len(captions)

# 평가 시
diversity = compute_diversity(generated_captions)
if diversity < 0.5:
    print(f"⚠️ WARNING: Low diversity ({diversity:.2f}) - Mode Collapse!")
```

**효과:**
- Mode Collapse 조기 탐지
- SigLIP만으로 놓칠 수 있는 문제 발견
- 실험 중단 판단 근거 제공

---

### 1.5 Early Stopping (v3만)

**코드:**
```python
patience = 3
min_delta = 0.001
patience_counter = 0

for epoch in range(max_epochs):
    # ... training ...
    
    if current_score > best_score + min_delta:
        best_score = current_score
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}!")
        break
```

**효과:**
- 불필요한 학습 epoch 방지
- 과적합 방지
- E5-v3에서 10 epoch 중 6 epoch에서 조기 종료

---

## 2. 학습 시간 단축

### 2.1 Vision Features 캐싱 (가장 큰 효과)

**원리:**
- Vision Encoder는 frozen → 같은 이미지에 대해 항상 같은 출력
- 매 epoch마다 반복 계산 불필요
- 한 번 계산해서 CPU 메모리에 저장

**코드:**
```python
class CachedVideoDataset(Dataset):
    def __init__(self, ..., vision_encoder, device):
        # 학습 전 모든 Vision Features 미리 계산
        self.vision_cache = {}
        print(f"Caching vision features for {len(samples)} samples...")
        
        with torch.no_grad():
            for idx in tqdm(range(len(samples))):
                frames = load_frames(samples[idx])
                pixel_values = image_processor(frames).to(device)
                features = vision_encoder(pixel_values).last_hidden_state
                self.vision_cache[idx] = features.cpu()  # CPU 저장
    
    def __getitem__(self, idx):
        return {
            "vision_features": self.vision_cache[idx],
            # pixel_values 대신 vision_features 반환
            ...
        }

# 모델 forward도 수정
def forward_with_cache(self, vision_features, ...):
    # vision_encoder 호출 생략
    projected = self.projector(vision_features)
    ...
```

**효과:**
```
캐싱 없음:
  - 매 epoch Vision Encoder 실행
  - 865 samples × 8 frames × 5 epochs = 34,600회 추론
  - 예상 시간: ~90분/epoch

캐싱 적용:
  - Vision Encoder 1회만 실행
  - 캐싱 시간: ~90분 (1회)
  - 이후 epoch: ~6분/epoch

10 epoch 기준:
  - 캐싱 없음: 900분
  - 캐싱 적용: 90 + 60 = 150분
  - 속도 향상: 6배 (83% 시간 절약)
```

---

### 2.2 Mixed Precision (bfloat16)

**코드:**
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

for batch in train_loader:
    with autocast('cuda', dtype=torch.bfloat16):
        loss = model(...)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**효과:**
- 메모리 사용량 ~50% 절약
- 연산 속도 향상 (Tensor Core 활용)
- 학습 정확도 유지 (bfloat16은 range 유지)

---

### 2.3 4-bit 양자화 (BitsAndBytes NF4)

**코드:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**효과:**
- Qwen3-8B 메모리: 32GB → ~8GB (1/4)
- A100 80GB에서 여유있게 학습 가능
- 품질 손실 최소화 (NF4는 정규분포 최적화)

---

### 2.4 Gradient Checkpointing

**코드:**
```python
llm.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

**효과:**
- Activation 메모리 대폭 절약
- 더 큰 batch size 가능
- 속도는 약간 감소 (recomputation)

---

### 2.5 DataLoader 최적화

**코드:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,           # 병렬 데이터 로딩
    pin_memory=True,         # GPU 전송 가속
    persistent_workers=True  # worker 재사용
)
```

**효과:**
- I/O 병목 해소
- GPU idle 시간 최소화
- 특히 비디오 프레임 로딩에 효과적

---

### 2.6 Gradient Accumulation

**코드:**
```python
gradient_accumulation = 2
effective_batch_size = batch_size * gradient_accumulation  # 16 * 2 = 32

for i, batch in enumerate(train_loader):
    loss = model(batch)
    loss.backward()
    
    if (i + 1) % gradient_accumulation == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**효과:**
- 작은 batch size로 큰 batch 효과
- GPU 메모리 절약
- 학습 안정성 향상

---

## 최적화 적용 요약표

| 최적화 | 목적 | 효과 | 적용 버전 |
|--------|------|------|----------|
| SigLIP2 | 품질 | 한국어 평가 가능 | v2/v3 |
| LR Scheduler | 품질 | 안정적 수렴 | v2/v3 |
| Gradient Clipping | 품질 | 학습 안정성 | v2/v3 |
| Diversity Monitoring | 품질 | Mode Collapse 탐지 | v2/v3 |
| Early Stopping | 품질/시간 | 과적합 방지 | v3만 |
| **Vision Caching** | **시간** | **30-40% 속도 향상** | v2/v3 |
| Mixed Precision | 시간 | 메모리/속도 개선 | v2/v3 |
| 4-bit 양자화 | 시간 | 메모리 1/4 절약 | v2/v3 |
| Gradient Checkpointing | 시간 | 메모리 절약 | v2/v3 |
| DataLoader 최적화 | 시간 | I/O 병목 해소 | v2/v3 |
| Gradient Accumulation | 시간 | 큰 배치 효과 | v2/v3 |

---

## 버그 수정

### GradScaler/autocast Deprecated API

**변경 전:**
```python
from torch.cuda.amp import autocast, GradScaler  # deprecated
scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
```

**변경 후:**
```python
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda')
with autocast('cuda', dtype=torch.bfloat16):
```

### use_cache Pyright 경고

**해결:**
```python
llm.config.use_cache = False  # type: ignore[attr-defined]
```
