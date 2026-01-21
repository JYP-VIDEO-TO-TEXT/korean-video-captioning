# LLM Backbones - 대규모 언어 모델

> 💡 **핵심 질문**: 어떤 언어 모델이 한국어를 가장 자연스럽게 생성하는가?

VLM에서 텍스트 생성을 담당하는 LLM Backbone의 발전 흐름을 정리합니다.

---

## 🎯 이 카테고리의 목표

LLM은 Visual Tokens을 받아 **자연스러운 한국어 캡션**을 생성합니다. 한국어 성능이 프로젝트의 핵심입니다.

```mermaid
flowchart LR
    subgraph Input["입력"]
        VT["🔢 Visual Tokens<br/>(Vision Encoder에서)"]
        PT["📝 Prompt<br/>'이 영상을 설명해주세요'"]
    end

    subgraph LLM["LLM Backbone"]
        Model["Qwen3 / Vicuna<br/>Auto-regressive 생성"]
    end

    subgraph Output["출력"]
        Caption["📝 한국어 캡션<br/>'푸른 바다 위로<br/>하얀 파도가...'"]
    end

    VT --> Model
    PT --> Model
    Model --> Caption

    style Model fill:#69db7c,stroke:#2f9e44
```

---

## 📊 한국어 성능 비교

### MMLU-Ko 벤치마크 (한국어 추론 능력)

```mermaid
xychart-beta
    title "한국어 벤치마크 성능 비교"
    x-axis ["Vicuna-7B", "LLaMA-7B", "Qwen-7B", "Qwen2-7B", "Qwen3-8B", "Qwen3-14B"]
    y-axis "MMLU-Ko (%)" 30 --> 80
    bar [38.2, 34.5, 52.1, 62.1, 68.5, 72.3]
```

### 핵심 인사이트

```mermaid
flowchart TB
    subgraph Insight["💡 핵심 발견"]
        I1["Vicuna-7B (LLaVA 기본)<br/>한국어 성능: 38.2%<br/>────────────────"]
        I2["Qwen3-8B (업그레이드 대상)<br/>한국어 성능: 68.5%<br/>────────────────"]
        I3["성능 향상: +79%<br/>비슷한 크기로 거의 2배!"]
    end

    I1 --> I3
    I2 --> I3

    style I1 fill:#ffe3e3
    style I2 fill:#d3f9d8
    style I3 fill:#fff3bf
```

---

## 📈 LLM 발전 흐름

```mermaid
flowchart TB
    subgraph Era2023["2023년: 오픈소스 LLM 시대"]
        LLaMA["🦙 LLaMA (Meta)<br/>────────────<br/>• 오픈소스 시작<br/>• 7B~65B 파라미터<br/>• 영어 중심"]
        
        LLaMA --> Vicuna["💬 Vicuna<br/>────────────<br/>• ShareGPT 학습<br/>• 대화 특화<br/>• LLaVA 기본 LLM"]
        
        LLaMA --> Qwen["🌏 Qwen (Alibaba)<br/>────────────<br/>• 다국어 특화<br/>• 한국어 성능 ↑<br/>• 8K 컨텍스트"]
    end

    subgraph Era2024["2024년: 성능 향상"]
        LLaMA --> LLaMA3["🦙 LLaMA 3<br/>────────────<br/>• 8B~400B<br/>• 성능 대폭 향상"]
        
        Qwen --> Qwen2["🌏 Qwen2<br/>────────────<br/>• GQA 적용<br/>• 128K 컨텍스트<br/>• 메모리 효율 ↑"]
    end

    subgraph Era2025["2025년: 최신"]
        Qwen2 --> Qwen3["⭐ Qwen3<br/>────────────<br/>• MoE 지원<br/>• 0.6B~235B<br/>• 최고 성능"]
    end

    subgraph Project["🎯 우리 선택"]
        Choice["Vicuna → Qwen3<br/>한국어 성능 2배 ↑"]
    end

    Vicuna -.-> Choice
    Qwen3 ==> Choice

    style Qwen3 fill:#69db7c,stroke:#2f9e44,stroke-width:3px
    style Choice fill:#ff6b6b,stroke:#c92a2a,color:#fff
```

---

## 🔬 아키텍처 상세

### Attention 메커니즘 진화

```mermaid
flowchart TB
    subgraph MHA["MHA (LLaMA, Vicuna)"]
        MHA_Desc["Multi-Head Attention<br/>────────────────────<br/>Q, K, V 모두 동일한 head 수<br/>예: 32 heads 전부"]
        MHA_KV["KV Cache 크기<br/>32 × head_dim × seq_len<br/>= 큼"]
    end

    subgraph GQA["GQA (Qwen2, Qwen3)"]
        GQA_Desc["Grouped Query Attention<br/>────────────────────<br/>Q: 32 heads<br/>K, V: 8 heads (그룹 공유)"]
        GQA_KV["KV Cache 크기<br/>8 × head_dim × seq_len<br/>= 4배 감소!"]
    end

    MHA --> |발전| GQA

    subgraph Benefit["장점"]
        B["• 긴 컨텍스트 처리 가능<br/>• 메모리 효율 향상<br/>• 속도 향상"]
    end

    GQA --> Benefit

    style GQA fill:#d3f9d8
    style Benefit fill:#fff3bf
```

### MoE (Mixture of Experts) - Qwen3

```mermaid
flowchart TB
    subgraph Dense["Dense Model (일반)"]
        D_Input["입력"] --> D_All["모든 파라미터<br/>활성화"]
        D_All --> D_Output["출력"]
        D_Note["8B 모델 = 8B 연산"]
    end

    subgraph MoE["MoE Model (Qwen3-30B-A3B)"]
        M_Input["입력"] --> Router["Router<br/>(어떤 Expert?)"]
        
        Router --> E1["Expert 1"]
        Router --> E2["Expert 2<br/>✓ 선택"]
        Router --> E3["Expert 3"]
        Router --> E4["Expert 4<br/>✓ 선택"]
        Router --> E5["..."]
        Router --> E8["Expert 8"]
        
        E2 --> M_Output["출력"]
        E4 --> M_Output
        
        M_Note["30B 총 파라미터<br/>3B만 활성화<br/>= Dense 3B 연산량으로<br/>30B급 성능!"]
    end

    style E2 fill:#d3f9d8
    style E4 fill:#d3f9d8
    style M_Note fill:#fff3bf
```

---

## 📊 모델별 상세 비교

### Qwen3 라인업

```mermaid
flowchart TB
    subgraph Dense["Dense Models"]
        Q06["Qwen3-0.6B<br/>CPU 가능"]
        Q17["Qwen3-1.7B<br/>Draft용"]
        Q4["Qwen3-4B<br/>T4 권장"]
        Q8["Qwen3-8B<br/>L4 권장"]
        Q14["Qwen3-14B<br/>A100 권장"]
        Q32["Qwen3-32B<br/>H100 권장"]
    end

    subgraph MoE_Models["MoE Models"]
        Q30A3["Qwen3-30B-A3B<br/>────────────<br/>30B 총 파라미터<br/>3B 활성화<br/>효율적!"]
        Q235A22["Qwen3-235B-A22B<br/>────────────<br/>235B 총 파라미터<br/>22B 활성화<br/>최고 성능!"]
    end

    style Q8 fill:#d3f9d8
    style Q14 fill:#4dabf7
    style Q30A3 fill:#fff3bf
```

### 상세 비교표

| 모델 | 파라미터 | 컨텍스트 | 한국어 | 라이선스 | GPU 권장 |
|------|----------|----------|--------|----------|----------|
| Vicuna-7B | 7B | 4K | 38.2% | 연구용 | T4 |
| Qwen-7B | 7B | 32K | 52.1% | 일부 상업 | T4 |
| Qwen2-7B | 7B | 128K | 62.1% | Apache-2.0 | L4 |
| **Qwen3-4B** | 4B | 32K | ~60% | Apache-2.0 | **T4** |
| **Qwen3-8B** | 8B | 128K | 68.5% | Apache-2.0 | **L4** |
| **Qwen3-14B** | 14B | 128K | 72.3% | Apache-2.0 | **A100** |
| **Qwen3-32B** | 32B | 128K | ~75% | Apache-2.0 | **H100** |

---

## 🎯 우리 프로젝트 적용

### LLM 교체 결정 트리

```mermaid
flowchart TB
    Start["LLM 선택"] --> Q1{"GPU 종류?"}
    
    Q1 -->|"T4 (16GB)"| T4_Choice["Qwen3-4B-Instruct<br/>4-bit: ~3GB"]
    
    Q1 -->|"L4 (24GB)"| L4_Choice["Qwen3-8B-Instruct<br/>4-bit: ~5GB"]
    
    Q1 -->|"A100 (40GB)"| A100_Choice["Qwen3-14B-Instruct<br/>4-bit: ~8GB"]
    
    Q1 -->|"H100 (80GB)"| H100_Q{"MoE 사용?"}
    H100_Q -->|"예"| H100_MoE["Qwen3-30B-A3B<br/>효율적 + 고성능"]
    H100_Q -->|"아니오"| H100_Dense["Qwen3-32B-Instruct<br/>최고 성능"]

    style T4_Choice fill:#fff3bf
    style L4_Choice fill:#d3f9d8
    style A100_Choice fill:#4dabf7
    style H100_MoE fill:#e5dbff
```

### LLM 교체 시 필요한 작업

```mermaid
flowchart TB
    subgraph Change["Vicuna → Qwen3 교체"]
        C1["기존: LLaVA + Vicuna-7B"]
        C2["변경: LLaVA + Qwen3-8B"]
    end

    subgraph Tasks["필요 작업"]
        T1["1️⃣ Projector 재학습<br/>────────────<br/>Vision 출력 → Qwen 입력<br/>차원 정렬 필요"]
        
        T2["2️⃣ 프롬프트 형식 변경<br/>────────────<br/>Vicuna: USER: ... ASSISTANT:<br/>Qwen3: <|im_start|>..."]
        
        T3["3️⃣ 토크나이저 변경<br/>────────────<br/>어휘 크기, special tokens"]
        
        T4["4️⃣ 학습 설정 조정<br/>────────────<br/>LR, batch size 등"]
    end

    Change --> T1 --> T2 --> T3 --> T4

    style T1 fill:#ffe3e3
    style T2 fill:#fff3bf
```

### 프롬프트 형식 비교

#### Vicuna (LLaVA 기본)
```
USER: <video>이 영상을 한국어로 상세히 묘사해주세요.
ASSISTANT: 이 영상은 푸른 바다와...
```

#### Qwen3 (업그레이드)
```
<|im_start|>system
당신은 비디오 캡셔닝 전문가입니다.<|im_end|>
<|im_start|>user
<video>이 영상을 한국어로 상세히 묘사해주세요.<|im_end|>
<|im_start|>assistant
이 영상은 푸른 바다와...<|im_end|>
```

### 코드 예시

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen3 로드
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct")

# 대화 형식 적용
messages = [
    {"role": "system", "content": "당신은 비디오 캡셔닝 전문가입니다."},
    {"role": "user", "content": "이 영상을 한국어로 상세히 묘사해주세요."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## 📚 논문 목록

| 파일 | 논문 | 핵심 포인트 | 중요도 |
|------|------|------------|--------|
| [llama.md](llama.md) | LLaMA (2023) | 오픈소스 LLM 기초 | ⭐⭐⭐ |
| [qwen.md](qwen.md) | Qwen (2023) | 다국어 특화 시작 | ⭐⭐⭐ |
| [qwen2.md](qwen2.md) | Qwen2 (2024) | GQA, 128K 컨텍스트 | ⭐⭐⭐⭐ |
| [qwen3.md](qwen3.md) | Qwen3 (2025) | **권장 업그레이드 대상** | ⭐⭐⭐⭐⭐ |

---

## 💻 GPU별 권장

| GPU | 권장 LLM | 4-bit 메모리 | 한국어 성능 |
|-----|----------|-------------|------------|
| **T4 (16GB)** | Qwen3-4B-Instruct | ~3GB | ~60% |
| **L4 (24GB)** | Qwen3-8B-Instruct | ~5GB | 68.5% |
| **A100 (40GB)** | Qwen3-14B-Instruct | ~8GB | 72.3% |
| **H100 (80GB)** | Qwen3-32B-Instruct | ~18GB | ~75% |
