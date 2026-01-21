# Training Methods - 학습 기법

> 💡 **핵심 질문**: 제한된 GPU 메모리에서 어떻게 대형 모델을 효율적으로 학습할 것인가?

효율적인 파인튜닝과 학습 전략의 발전 흐름을 정리합니다.

---

## 🎯 이 카테고리의 목표

7B 파라미터 모델을 Full Fine-tuning 하려면 **112GB GPU 메모리**가 필요합니다. 하지만 우리는 **T4 (16GB)**에서도 학습해야 합니다!

```mermaid
flowchart LR
    subgraph Problem["❌ 문제"]
        Full["Full Fine-tuning<br/>7B 모델<br/>────────────<br/>필요: 112GB<br/>보유: 16GB (T4)"]
    end

    subgraph Solution["✅ 해결책"]
        QLoRA["QLoRA<br/>────────────<br/>필요: 6GB<br/>T4에서 가능!"]
    end

    Problem -->|"PEFT + 양자화"| Solution

    style Problem fill:#ffe3e3
    style Solution fill:#d3f9d8
```

---

## 📊 메모리 사용량 비교

### 7B 모델 기준

```mermaid
xychart-beta
    title "학습 방법별 GPU 메모리 사용량 (GB)"
    x-axis ["Full FT", "LoRA", "QLoRA"]
    y-axis "메모리 (GB)" 0 --> 120
    bar [112, 56, 6]
```

### 왜 이렇게 차이가 나는가?

```mermaid
flowchart TB
    subgraph FullFT["Full Fine-tuning: 112GB"]
        F1["모델 가중치 (FP16)<br/>7B × 2 bytes = 14GB"]
        F2["Gradients<br/>7B × 2 bytes = 14GB"]
        F3["Optimizer States (Adam)<br/>7B × 8 bytes = 56GB"]
        F4["Activations<br/>~28GB"]
        
        F1 --> Total1["총: ~112GB"]
        F2 --> Total1
        F3 --> Total1
        F4 --> Total1
    end

    subgraph QLoRA_Mem["QLoRA: 6GB"]
        Q1["모델 가중치 (4-bit)<br/>7B × 0.5 bytes = 3.5GB"]
        Q2["LoRA 가중치<br/>~16M × 2 bytes = 32MB"]
        Q3["Optimizer (LoRA만)<br/>~16M × 8 bytes = 128MB"]
        Q4["Activations<br/>~2GB"]
        
        Q1 --> Total2["총: ~6GB"]
        Q2 --> Total2
        Q3 --> Total2
        Q4 --> Total2
    end

    style Total1 fill:#ffe3e3
    style Total2 fill:#d3f9d8
```

---

## 📈 PEFT 발전 흐름

```mermaid
flowchart TB
    subgraph Era2021["2021년: PEFT의 시작"]
        LoRA["🔧 LoRA<br/>Low-Rank Adaptation<br/>────────────────────<br/>• 가중치를 저차원으로 분해<br/>• 0.1~1% 파라미터만 학습<br/>• 메모리 ~50% 절약"]
    end

    subgraph Era2023["2023년: 극한의 효율"]
        QLoRA["⚡ QLoRA<br/>4-bit + LoRA<br/>────────────────────<br/>• NF4 양자화<br/>• Double Quantization<br/>• 메모리 ~90% 절약"]
    end

    subgraph Era2024["2024년: 품질 향상"]
        DoRA["📈 DoRA<br/>Weight-Decomposed<br/>────────────────────<br/>• Magnitude + Direction 분리<br/>• 같은 r로 더 나은 성능<br/>• 안정적 학습"]
    end

    LoRA --> QLoRA
    LoRA --> DoRA

    subgraph Project["🎯 우리 선택"]
        Choice["T4/L4: QLoRA (필수)<br/>A100+: LoRA/DoRA (선택)"]
    end

    QLoRA ==> Choice
    DoRA -.-> Choice

    style QLoRA fill:#d3f9d8,stroke:#2f9e44,stroke-width:3px
    style Choice fill:#ff6b6b,stroke:#c92a2a,color:#fff
```

---

## 🔬 LoRA 상세 설명

### 핵심 아이디어

일반적인 가중치 업데이트는 **전체 행렬**을 수정합니다. LoRA는 이를 **저차원 행렬의 곱**으로 근사합니다.

```mermaid
flowchart TB
    subgraph Original["기존 방식"]
        W["W (4096×4096)<br/>= 16.7M 파라미터"]
        DW["ΔW (4096×4096)<br/>= 16.7M 학습"]
        W --> |"W + ΔW"| WNew["W' (업데이트)"]
        DW --> WNew
    end

    subgraph LoRA_Way["LoRA 방식"]
        W2["W (4096×4096)<br/>Frozen ❄️"]
        A["A (4096×16)<br/>= 65K 학습"]
        B["B (16×4096)<br/>= 65K 학습"]
        W2 --> |"W + B×A"| WNew2["W' (업데이트)"]
        A --> |"저차원 곱"| BA["B×A"]
        B --> BA
        BA --> WNew2
    end

    subgraph Compare["비교"]
        C["16.7M → 130K<br/>파라미터 99% 감소!"]
    end

    style W fill:#ffe3e3
    style DW fill:#ffe3e3
    style A fill:#d3f9d8
    style B fill:#d3f9d8
    style C fill:#fff3bf
```

### 수학적 표현

```
원본:      h = W × x
LoRA:     h = W × x + (B × A) × x × (α/r)

여기서:
• W: 원본 가중치 (frozen)
• A: 4096 × r 행렬 (학습) - Down-projection
• B: r × 4096 행렬 (학습) - Up-projection
• r: rank (보통 8~64)
• α: scaling factor (보통 2×r)
```

### Rank 선택 가이드

```mermaid
flowchart TB
    subgraph Ranks["Rank별 특성"]
        R8["r=8<br/>────────<br/>파라미터: 최소<br/>메모리: 최소<br/>품질: 기본"]
        R16["r=16<br/>────────<br/>파라미터: 적음<br/>메모리: 적음<br/>품질: 좋음"]
        R32["r=32<br/>────────<br/>파라미터: 보통<br/>메모리: 보통<br/>품질: 매우 좋음"]
        R64["r=64<br/>────────<br/>파라미터: 많음<br/>메모리: 많음<br/>품질: 최고"]
    end

    subgraph GPU_Rec["GPU별 권장"]
        T4["T4 → r=8"]
        L4["L4 → r=16"]
        A100["A100 → r=32"]
        H100["H100 → r=64"]
    end

    R8 --> T4
    R16 --> L4
    R32 --> A100
    R64 --> H100

    style R8 fill:#fff3bf
    style R16 fill:#d3f9d8
    style R32 fill:#4dabf7
    style R64 fill:#e5dbff
```

---

## ⚡ QLoRA 상세 설명

### NF4 (NormalFloat 4-bit)

일반적인 INT4는 **균일한 간격**으로 양자화합니다. 하지만 실제 가중치는 **정규분포**를 따릅니다!

```mermaid
flowchart TB
    subgraph INT4["INT4 양자화"]
        I_Dist["균일 분포 가정<br/>[-8, -7, ..., 6, 7]<br/>────────────────<br/>실제 분포와 불일치<br/>양자화 오류 큼"]
    end

    subgraph NF4_Q["NF4 양자화"]
        N_Dist["정규 분포 가정<br/>[-1.0, -0.69, -0.52, ..., 0.95, 1.0]<br/>────────────────<br/>실제 분포와 일치<br/>양자화 오류 작음"]
    end

    INT4 -->|"개선"| NF4_Q

    style INT4 fill:#ffe3e3
    style NF4_Q fill:#d3f9d8
```

### Double Quantization

Scale 값도 양자화하여 메모리를 추가로 절약합니다.

```mermaid
flowchart TB
    subgraph Normal["일반 양자화"]
        N1["Weight Group (64개)"]
        N2["Scale: FP32 (4 bytes)"]
        N3["메모리: 64×0.5 + 4 = 36 bytes"]
    end

    subgraph Double["Double Quantization"]
        D1["Weight Group (64개)"]
        D2["Scale: FP8 (1 byte)"]
        D3["메모리: 64×0.5 + 1 = 33 bytes"]
        D4["Scale 저장 75% 절약!"]
    end

    Normal -->|"개선"| Double

    style D4 fill:#d3f9d8
```

### QLoRA 전체 구조

```mermaid
flowchart TB
    subgraph Model["모델 구조"]
        Base["Base Model<br/>────────────<br/>4-bit NF4 양자화<br/>Frozen ❄️"]
        
        LoRA_A["LoRA A<br/>────────────<br/>FP16<br/>학습 🔥"]
        
        LoRA_B["LoRA B<br/>────────────<br/>FP16<br/>학습 🔥"]
    end

    subgraph Forward["순전파"]
        Input["입력 x"]
        Input --> Base
        Input --> LoRA_A
        Base --> |"Dequant → FP16"| Add["합산"]
        LoRA_A --> LoRA_B --> |"스케일링"| Add
        Add --> Output["출력"]
    end

    style Base fill:#e7f5ff
    style LoRA_A fill:#d3f9d8
    style LoRA_B fill:#d3f9d8
```

---

## 📐 2-Stage Training

LLaVA에서 제안한 멀티모달 학습 전략입니다.

```mermaid
flowchart TB
    subgraph Stage1["Stage 1: Feature Alignment"]
        S1_Data["📊 데이터<br/>Image-Caption 쌍<br/>(CC3M 595K)"]
        S1_Train["🎯 학습 대상<br/>Projector만"]
        S1_Freeze["❄️ Frozen<br/>Vision Encoder<br/>LLM"]
        S1_Goal["💡 목표<br/>Vision ↔ Language<br/>공간 정렬"]
        S1_Setting["⚙️ 설정<br/>Epochs: 1<br/>LR: 1e-3"]
        
        S1_Data --> S1_Train
        S1_Freeze --> S1_Train
        S1_Train --> S1_Goal
        S1_Goal --> S1_Setting
    end

    subgraph Stage2["Stage 2: Instruction Tuning"]
        S2_Data["📊 데이터<br/>Instruction 데이터<br/>(AI-Hub 캡셔닝)"]
        S2_Train["🎯 학습 대상<br/>Projector + LLM (LoRA)"]
        S2_Freeze["❄️ Frozen<br/>Vision Encoder"]
        S2_Goal["💡 목표<br/>태스크 특화<br/>능력 학습"]
        S2_Setting["⚙️ 설정<br/>Epochs: 3<br/>LR: 2e-5"]
        
        S2_Data --> S2_Train
        S2_Freeze --> S2_Train
        S2_Train --> S2_Goal
        S2_Goal --> S2_Setting
    end

    Stage1 --> Stage2

    style Stage1 fill:#e7f5ff
    style Stage2 fill:#fff3bf
```

### 왜 2단계로 나누는가?

```mermaid
flowchart TB
    subgraph Problem["❌ 한 번에 학습하면"]
        P1["Vision 특징이 LLM에<br/>제대로 전달 안됨"]
        P2["학습 불안정"]
        P3["수렴 어려움"]
    end

    subgraph Solution["✅ 2단계로 나누면"]
        S1["Stage 1: 먼저 '언어'를 가르침<br/>(Vision-Language 정렬)"]
        S2["Stage 2: 그 다음 '태스크'를 가르침<br/>(캡셔닝 능력)"]
    end

    Problem --> |"해결"| Solution

    style Problem fill:#ffe3e3
    style Solution fill:#d3f9d8
```

---

## 🎯 우리 프로젝트 적용

### GPU별 설정

```mermaid
flowchart TB
    subgraph T4_Config["🟡 T4 (16GB)"]
        T4_Method["방법: QLoRA (필수)"]
        T4_R["r=8, α=16"]
        T4_Batch["batch=1, grad_accum=16"]
        T4_Target["target: attention만"]
    end

    subgraph L4_Config["🟢 L4 (24GB)"]
        L4_Method["방법: QLoRA"]
        L4_R["r=16, α=32"]
        L4_Batch["batch=2, grad_accum=8"]
        L4_Target["target: attention만"]
    end

    subgraph A100_Config["🔵 A100 (40GB)"]
        A100_Method["방법: LoRA 또는 QLoRA"]
        A100_R["r=32, α=64"]
        A100_Batch["batch=4, grad_accum=4"]
        A100_Target["target: attention + MLP"]
    end

    subgraph H100_Config["🟣 H100 (80GB)"]
        H100_Method["방법: LoRA"]
        H100_R["r=64, α=128"]
        H100_Batch["batch=8, grad_accum=2"]
        H100_Target["target: attention + MLP"]
    end

    style T4_Config fill:#fff3bf
    style L4_Config fill:#d3f9d8
    style A100_Config fill:#d0ebff
    style H100_Config fill:#e5dbff
```

### 코드 예시

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",           # NF4 사용
    bnb_4bit_use_double_quant=True,      # Double Quantization
)

# 2. 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# 3. LoRA 준비
model = prepare_model_for_kbit_training(model)

# 4. LoRA 설정
lora_config = LoraConfig(
    r=16,                                 # Rank
    lora_alpha=32,                        # Scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 출력: trainable params: 16,777,216 || all params: 7,000,000,000 || trainable%: 0.24%
```

---

## 📚 논문 목록

| 파일 | 논문 | 핵심 포인트 | 중요도 |
|------|------|------------|--------|
| [lora.md](lora.md) | LoRA (2021) | PEFT의 기초 | ⭐⭐⭐⭐ |
| [qlora.md](qlora.md) | QLoRA (2023) | **T4/L4 필수** | ⭐⭐⭐⭐⭐ |
| [dora.md](dora.md) | DoRA (2024) | LoRA 개선 | ⭐⭐⭐ |
| [llava_2stage.md](llava_2stage.md) | 2-Stage (2023) | **멀티모달 학습 전략** | ⭐⭐⭐⭐⭐ |
