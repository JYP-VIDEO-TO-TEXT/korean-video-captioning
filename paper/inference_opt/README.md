# Inference Optimization - 추론 최적화

> 💡 **핵심 질문**: 학습된 모델을 어떻게 빠르고 효율적으로 서빙할 것인가?

추론 속도와 효율성을 향상시키는 기법들의 발전 흐름을 정리합니다.

---

## 🎯 이 카테고리의 목표

학습이 끝난 후, 실제 서비스에서는 **속도**와 **비용**이 중요합니다. 최적화 기법들을 조합하면 **8배 이상** 속도를 향상시킬 수 있습니다.

```mermaid
flowchart LR
    subgraph Before["최적화 전"]
        B["1x 속도<br/>높은 비용"]
    end

    subgraph After["최적화 후"]
        A["8x+ 속도<br/>낮은 비용"]
    end

    Before -->|"AWQ + vLLM<br/>+ Speculative"| After

    style Before fill:#ffe3e3
    style After fill:#d3f9d8
```

---

## 📊 속도 향상 비교

### 최적화 기법 조합 효과

```mermaid
xychart-beta
    title "최적화 기법별 속도 향상 (배수)"
    x-axis ["기본", "+AWQ", "+vLLM", "+AWQ+vLLM", "+SpecDec"]
    y-axis "속도 (배)" 0 --> 10
    bar [1, 3.2, 4.8, 6.4, 8.5]
```

---

## 📈 최적화 기법 발전 흐름

```mermaid
flowchart TB
    subgraph Quantization["🗜️ 양자화 (모델 압축)"]
        direction TB
        GPTQ["GPTQ (2022)<br/>────────────<br/>• Post-training 양자화<br/>• Layer-wise 최적화<br/>• 재학습 불필요"]
        
        AWQ["⭐ AWQ (2023)<br/>────────────<br/>• Activation-aware<br/>• 중요 가중치 보호<br/>• 더 나은 품질"]
        
        GPTQ --> AWQ
    end

    subgraph Serving["🚀 서빙 최적화"]
        direction TB
        vLLM["⭐ vLLM (2023)<br/>────────────<br/>• PagedAttention<br/>• Continuous Batching<br/>• 처리량 2-4x↑"]
    end

    subgraph Generation["⚡ 생성 가속"]
        direction TB
        SpecDec["Speculative Decoding<br/>(2023)<br/>────────────<br/>• Draft-Verify<br/>• 작은 모델로 초안<br/>• 큰 모델로 검증"]
    end

    subgraph Combination["🎯 조합 전략"]
        Combine["AWQ + vLLM<br/>────────────<br/>양자화된 모델을<br/>효율적으로 서빙<br/><br/>+ Speculative<br/>────────────<br/>H100에서<br/>추가 가속"]
    end

    Quantization --> Combination
    Serving --> Combination
    Generation --> Combination

    style AWQ fill:#d3f9d8,stroke:#2f9e44
    style vLLM fill:#d3f9d8,stroke:#2f9e44
    style Combination fill:#fff3bf
```

---

## 🔬 핵심 기술 상세

### 1. AWQ (Activation-aware Weight Quantization)

#### 핵심 아이디어

모든 가중치가 **똑같이 중요하지 않습니다**. Activation이 큰 채널의 가중치가 더 중요합니다!

```mermaid
flowchart TB
    subgraph Analysis["분석: 어떤 가중치가 중요한가?"]
        Input["입력 데이터<br/>(Calibration)"]
        Act["Activation 크기 측정"]
        Importance["중요도 계산<br/>Activation이 큰 채널 = 중요"]
        
        Input --> Act --> Importance
    end

    subgraph Quantization["양자화 전략"]
        Important["중요한 가중치<br/>────────────<br/>스케일 업 후 양자화<br/>정밀도 보존"]
        
        NotImportant["덜 중요한 가중치<br/>────────────<br/>일반 양자화<br/>오류 허용"]
    end

    Importance --> Important
    Importance --> NotImportant

    subgraph Result["결과"]
        R["4-bit 양자화에서도<br/>FP16에 가까운 품질!"]
    end

    Important --> Result
    NotImportant --> Result

    style Important fill:#d3f9d8
    style R fill:#fff3bf
```

#### AWQ vs GPTQ 비교

```mermaid
flowchart LR
    subgraph GPTQ_Way["GPTQ"]
        G1["많은 Calibration 데이터"]
        G2["Layer별 순차 최적화"]
        G3["Perplexity: 5.85"]
    end

    subgraph AWQ_Way["AWQ"]
        A1["적은 Calibration 데이터"]
        A2["Activation 기반 중요도"]
        A3["Perplexity: 5.72 ✨"]
    end

    style AWQ_Way fill:#d3f9d8
```

---

### 2. vLLM (PagedAttention)

#### 기존 KV Cache 문제

```mermaid
flowchart TB
    subgraph Problem["❌ 기존 방식의 문제"]
        subgraph Memory["GPU 메모리"]
            R1["Request 1: ████████████░░░░░░░░░░░░"]
            R2["Request 2: ██████████████████████░░"]
            R3["Request 3: ████░░░░░░░░░░░░░░░░░░░░"]
        end
        
        Issue["문제점:<br/>• 최대 길이만큼 사전 할당<br/>• 짧은 시퀀스도 큰 공간 차지<br/>• 메모리 파편화<br/>• 동시 요청 수 제한"]
    end

    style Issue fill:#ffe3e3
```

#### PagedAttention 해결책

```mermaid
flowchart TB
    subgraph Solution["✅ PagedAttention"]
        subgraph Physical["Physical Blocks"]
            B0["Block 0"]
            B1["Block 1"]
            B2["Block 2"]
            B3["Block 3"]
            B4["Block 4"]
            B5["Block 5"]
        end
        
        subgraph Mapping["Page Table 매핑"]
            R1_Map["Request 1: [0→B0, 1→B2, 2→B4]"]
            R2_Map["Request 2: [0→B1, 1→B3]"]
            R3_Map["Request 3: [0→B5]"]
        end
        
        Benefit["장점:<br/>• 필요한 만큼만 동적 할당<br/>• 블록 단위 재사용<br/>• 메모리 파편화 최소화<br/>• 동시 요청 2-4배 증가"]
    end

    Physical --> Mapping
    Mapping --> Benefit

    style Benefit fill:#d3f9d8
```

#### Continuous Batching

```mermaid
flowchart TB
    subgraph Static["❌ Static Batching"]
        S_Desc["모든 요청이 끝날 때까지 대기"]
        S1["Req 1: ████████████████████"]
        S2["Req 2: ████████............"]
        S3["Req 3: ██████████████████████████"]
        S_Note["← 짧은 요청도 대기<br/>← GPU 유휴 시간 발생"]
    end

    subgraph Continuous["✅ Continuous Batching"]
        C_Desc["완료 즉시 새 요청 시작"]
        C1["Req 1: ████████████████████"]
        C2["Req 2: ████████|Req 4: █████████|"]
        C3["Req 3: ██████████████████████████"]
        C_Note["← 완료 즉시 새 요청<br/>← GPU 항상 활용"]
    end

    Static --> |"개선"| Continuous

    style Static fill:#ffe3e3
    style Continuous fill:#d3f9d8
```

---

### 3. Speculative Decoding

#### 핵심 아이디어

큰 모델의 **토큰 생성은 느립니다**. 작은 모델로 **초안**을 만들고, 큰 모델로 **검증**하면 빠릅니다!

```mermaid
flowchart TB
    subgraph Traditional["❌ 기존 Auto-regressive"]
        T1["Token 1"] --> T2["Token 2"] --> T3["Token 3"] --> T4["Token 4"] --> T5["Token 5"]
        T_Note["각 토큰마다 14B 모델 실행<br/>= 5번의 대형 연산"]
    end

    subgraph Speculative["✅ Speculative Decoding"]
        subgraph Draft["Draft Model (1.7B) - 빠름"]
            D["한 번에 5개 토큰 생성<br/>[D1, D2, D3, D4, D5]"]
        end
        
        subgraph Verify["Target Model (14B) - 병렬 검증"]
            V["5개 동시 검증<br/>[T1✓, T2✓, T3✓, T4✗, T5✗]"]
        end
        
        subgraph Output["결과"]
            O["[T1, T2, T3] 수락<br/>T4부터 재생성"]
        end
        
        Draft --> Verify --> Output
        
        S_Note["Draft 5번 (소형) + Target 1번 (대형)<br/>≈ 기존의 1-2번 연산량"]
    end

    Traditional --> |"개선"| Speculative

    style T_Note fill:#ffe3e3
    style S_Note fill:#d3f9d8
```

#### 수학적 분석

```mermaid
flowchart LR
    subgraph Analysis["효율성 분석"]
        A1["Draft 모델 (1.7B)<br/>5 토큰 × 소형 연산<br/>≈ 1 대형 연산"]
        
        A2["Target 모델 (14B)<br/>5 토큰 병렬 검증<br/>= 1 대형 연산"]
        
        A3["총 연산량<br/>≈ 2 대형 연산<br/>결과: 3-5 토큰"]
        
        A4["기존 방식<br/>3-5 토큰<br/>= 3-5 대형 연산"]
    end

    A1 --> A3
    A2 --> A3
    
    subgraph Result["결과"]
        R["2-3배 속도 향상!"]
    end
    
    A3 --> Result
    A4 --> Result

    style Result fill:#d3f9d8
```

---

## 🎯 우리 프로젝트 적용

### GPU별 최적화 전략

```mermaid
flowchart TB
    subgraph T4["🟡 T4 (16GB)"]
        T4_Opt["최적화 제한적<br/>────────────<br/>• QLoRA 학습만<br/>• 추론 최적화 어려움<br/>• 메모리 여유 없음"]
    end

    subgraph L4["🟢 L4 (24GB)"]
        L4_Opt["기본 최적화<br/>────────────<br/>• QLoRA 학습<br/>• AWQ 양자화 가능<br/>• vLLM 단독 가능"]
    end

    subgraph A100["🔵 A100 (40GB)"]
        A100_Opt["권장 조합<br/>────────────<br/>• LoRA 학습<br/>• AWQ 양자화<br/>• vLLM 서빙<br/>• 6x 속도 향상"]
    end

    subgraph H100["🟣 H100 (80GB)"]
        H100_Opt["최대 최적화<br/>────────────<br/>• LoRA 학습<br/>• AWQ + vLLM<br/>• + Speculative Dec<br/>• 8x+ 속도 향상"]
    end

    style T4 fill:#fff3bf
    style L4 fill:#d3f9d8
    style A100 fill:#d0ebff
    style H100 fill:#e5dbff
```

### 배포 파이프라인

```mermaid
flowchart TB
    subgraph Training["1️⃣ 학습 Phase"]
        T1["QLoRA/LoRA Fine-tuning"]
        T2["LoRA Adapter 저장"]
        T1 --> T2
    end

    subgraph Merge["2️⃣ 병합 Phase"]
        M1["Base Model 로드"]
        M2["LoRA Adapter 병합"]
        M3["전체 가중치 모델"]
        M1 --> M2 --> M3
    end

    subgraph Quantize["3️⃣ 양자화 Phase"]
        Q1["AWQ 양자화"]
        Q2["Calibration (128 샘플)"]
        Q3["4-bit 모델 저장"]
        Q1 --> Q2 --> Q3
    end

    subgraph Deploy["4️⃣ 배포 Phase"]
        D1["vLLM으로 모델 로드"]
        D2["API 서버 시작"]
        D3["PagedAttention 자동 적용"]
        D1 --> D2 --> D3
    end

    Training --> Merge --> Quantize --> Deploy

    style Training fill:#e7f5ff
    style Merge fill:#fff3bf
    style Quantize fill:#d3f9d8
    style Deploy fill:#e5dbff
```

### 코드 예시

#### AWQ 양자화

```python
from awq import AutoAWQForCausalLM

# 모델 로드
model = AutoAWQForCausalLM.from_pretrained(model_path)

# AWQ 양자화
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
    },
    calib_data=calibration_samples,  # 128개면 충분
)

# 저장
model.save_quantized("model-awq")
```

#### vLLM 서빙

```python
from vllm import LLM, SamplingParams

# AWQ 모델을 vLLM으로 로드
llm = LLM(
    model="model-awq",
    quantization="awq",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

# 샘플링 설정
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

# 배치 추론
outputs = llm.generate(prompts, sampling_params)
```

#### Speculative Decoding (H100)

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-14B-Instruct-AWQ",
    speculative_model="Qwen/Qwen3-1.7B-Instruct",  # Draft 모델
    num_speculative_tokens=5,
)
```

---

## 📚 논문 목록

| 파일 | 논문 | 핵심 포인트 | 중요도 |
|------|------|------------|--------|
| [gptq.md](gptq.md) | GPTQ (2022) | Post-training Quantization | ⭐⭐⭐ |
| [awq.md](awq.md) | AWQ (2023) | **권장 양자화** | ⭐⭐⭐⭐⭐ |
| [vllm.md](vllm.md) | vLLM (2023) | **권장 서빙 프레임워크** | ⭐⭐⭐⭐⭐ |
| [speculative_decoding.md](speculative_decoding.md) | Speculative (2023) | H100 추가 최적화 | ⭐⭐⭐⭐ |

---

## 💻 적용 우선순위

```mermaid
flowchart TB
    subgraph Priority["적용 우선순위"]
        P1["1️⃣ vLLM (A100+)<br/>────────────<br/>설치만으로 2-4x 향상<br/>가장 쉬운 최적화"]
        
        P2["2️⃣ AWQ (A100+)<br/>────────────<br/>메모리 절약 + 속도<br/>vLLM과 함께 사용"]
        
        P3["3️⃣ Speculative (H100)<br/>────────────<br/>추가 설정 필요<br/>최대 성능 추구 시"]
    end

    P1 --> P2 --> P3

    style P1 fill:#d3f9d8
    style P2 fill:#fff3bf
    style P3 fill:#e5dbff
```
