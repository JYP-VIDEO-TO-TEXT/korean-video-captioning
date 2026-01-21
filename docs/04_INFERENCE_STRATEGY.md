# 04. 추론 전략 상세

## 개요

본 문서에서는 대한민국 배경영상 캡셔닝 모델의 추론 최적화 전략을 상세히 설명합니다. vLLM 통합, KV Cache 최적화, 양자화 전략, Speculative Decoding, 생성 파라미터 튜닝 등을 다룹니다.

---

## 1. 추론 최적화의 필요성

### 1.1 LLM 추론의 특성

```
LLM 추론 단계:

┌─────────────────────────────────────────────────────────────────┐
│                    LLM 추론 과정                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Prefill (프리필) 단계                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Input: [Vision Tokens] + [Prompt Tokens]                  │  │
│  │         ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓                 │  │
│  │  모든 토큰을 한 번에 처리 (병렬)                            │  │
│  │  KV Cache 생성                                             │  │
│  │  특징: Compute-bound, 높은 GPU 활용률                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  2. Decode (디코드) 단계                                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Output: [Token1] → [Token2] → [Token3] → ...              │  │
│  │          ↓         ↓          ↓                            │  │
│  │  토큰을 하나씩 순차 생성 (자기회귀)                         │  │
│  │  KV Cache 재사용                                           │  │
│  │  특징: Memory-bound, 낮은 GPU 활용률                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  병목: Decode 단계 (전체 시간의 90% 이상)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 추론 최적화 목표

| 목표 | 설명 | 측정 지표 |
|------|------|----------|
| **처리량 (Throughput)** | 단위 시간당 처리하는 요청 수 | requests/sec |
| **지연시간 (Latency)** | 요청 처리에 걸리는 시간 | ms/request |
| **메모리 효율** | GPU 메모리 사용 최적화 | GB |
| **품질 유지** | 최적화 후에도 생성 품질 유지 | METEOR score |

---

## 2. vLLM 통합

### 2.1 vLLM 개요

**vLLM**은 LLM 추론을 위한 고성능 라이브러리로, **PagedAttention** 알고리즘을 통해 메모리 효율과 처리량을 크게 향상시킵니다.

```
vLLM vs 기존 HuggingFace 추론:

처리량 비교 (배치 크기 32):
┌────────────────────────────────────────┐
│  HuggingFace │████████ 8 req/s        │
│  vLLM        │████████████████████████│
│              │ 50 req/s (6.25x 향상)  │
└────────────────────────────────────────┘

메모리 사용량 비교:
┌────────────────────────────────────────┐
│  HuggingFace │████████████████ 80%    │
│              │ (고정 할당)             │
│  vLLM        │████████ 40%            │
│              │ (동적 할당)             │
└────────────────────────────────────────┘
```

### 2.2 PagedAttention 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                    PagedAttention                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  기존 방식: 연속 메모리 할당                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Seq1 KV Cache │████████████████████████████████████████│    │
│  │               │     (최대 길이만큼 미리 할당)            │    │
│  │ Seq2 KV Cache │████████████████████████████████████████│    │
│  │               │     (사용하지 않는 공간 낭비)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  PagedAttention: 페이지 단위 동적 할당                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Block Table:                                             │    │
│  │ Seq1: [B0] → [B3] → [B7] → [B2]                         │    │
│  │ Seq2: [B1] → [B4] → [B5]                                │    │
│  │                                                          │    │
│  │ Physical Blocks:                                         │    │
│  │ ┌────┬────┬────┬────┬────┬────┬────┬────┐              │    │
│  │ │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │              │    │
│  │ │Seq1│Seq2│Seq1│Seq1│Seq2│Seq2│Free│Seq1│              │    │
│  │ └────┴────┴────┴────┴────┴────┴────┴────┘              │    │
│  │                                                          │    │
│  │ 장점:                                                    │    │
│  │ - 필요한 만큼만 메모리 할당                              │    │
│  │ - 여러 시퀀스가 블록 공유 가능                           │    │
│  │ - 메모리 단편화 최소화                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 vLLM 설정

```python
from vllm import LLM, SamplingParams

# vLLM 모델 로드
llm = LLM(
    model="Qwen/Qwen3-8B-Instruct",
    
    # GPU 메모리 설정
    gpu_memory_utilization=0.9,  # GPU 메모리의 90% 사용
    
    # 텐서 병렬화 (멀티 GPU)
    tensor_parallel_size=1,  # 단일 GPU
    
    # KV Cache 설정
    max_num_batched_tokens=4096,  # 배치당 최대 토큰
    max_num_seqs=32,  # 동시 처리 시퀀스 수
    
    # 양자화
    quantization="awq",  # 또는 "gptq", "squeezellm"
    
    # 기타
    trust_remote_code=True,
    dtype="bfloat16",
)

# 샘플링 파라미터
sampling_params = SamplingParams(
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)

# 추론 실행
outputs = llm.generate(prompts, sampling_params)
```

### 2.4 vLLM 설정 가이드

| GPU | gpu_memory_utilization | max_num_seqs | tensor_parallel |
|-----|----------------------|--------------|-----------------|
| T4 (16GB) | 0.85 | 8 | 1 |
| L4 (24GB) | 0.90 | 16 | 1 |
| A100 (40GB) | 0.90 | 32 | 1 |
| H100 (80GB) | 0.90 | 64 | 1 |

### 2.5 vLLM + LoRA 어댑터

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

# 기본 모델 로드
llm = LLM(
    model="llava-hf/LLaVA-NeXT-Video-7B-hf",
    enable_lora=True,
    max_lora_rank=64,
)

# LoRA 어댑터 요청
lora_request = LoRARequest(
    lora_name="korean_caption_adapter",
    lora_int_id=1,
    lora_local_path="outputs/lora_adapter",
)

# LoRA와 함께 추론
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=lora_request
)
```

---

## 3. KV Cache 최적화

### 3.1 KV Cache 개념

```
KV Cache의 역할:

Attention 계산:
Attention(Q, K, V) = softmax(QK^T / √d) × V

자기회귀 생성 시:
- 매 토큰마다 이전 모든 토큰의 K, V 필요
- KV Cache: 이전 K, V 값을 저장하여 재계산 방지

메모리 사용량:
KV Cache Size = 2 × num_layers × hidden_size × seq_len × batch_size × dtype_size

예시 (Qwen-7B, seq_len=2048, batch=1, FP16):
= 2 × 32 × 4096 × 2048 × 1 × 2 bytes
= 1.07 GB
```

### 3.2 Prefix Caching

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prefix Caching                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  시나리오: 동일한 시스템 프롬프트를 여러 요청에서 사용            │
│                                                                  │
│  기존 방식:                                                      │
│  Request 1: [System Prompt] + [User 1] → Prefill 전체           │
│  Request 2: [System Prompt] + [User 2] → Prefill 전체           │
│  Request 3: [System Prompt] + [User 3] → Prefill 전체           │
│                                                                  │
│  Prefix Caching:                                                 │
│  ┌────────────────────┐                                         │
│  │  [System Prompt]   │ → 캐시됨 (한 번만 계산)                 │
│  └────────────────────┘                                         │
│            ↓                                                     │
│  Request 1: [Cached] + [User 1] → 빠른 Prefill                  │
│  Request 2: [Cached] + [User 2] → 빠른 Prefill                  │
│  Request 3: [Cached] + [User 3] → 빠른 Prefill                  │
│                                                                  │
│  속도 향상: 시스템 프롬프트 길이에 비례 (최대 10x)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# vLLM에서 Prefix Caching 활성화
llm = LLM(
    model="Qwen/Qwen3-8B-Instruct",
    enable_prefix_caching=True,  # Prefix Caching 활성화
)

# 공통 프롬프트
system_prompt = """당신은 한국의 배경영상을 상세히 묘사하는 AI입니다.
영상에 보이는 자연, 도시, 건축물 등을 한국어로 자세히 설명해주세요.
객체, 동작, 속성, 배경을 모두 포함하여 설명합니다."""

# 여러 요청 - 시스템 프롬프트 KV Cache 재사용
prompts = [
    f"{system_prompt}\n\n영상 설명: {video_embedding_1}",
    f"{system_prompt}\n\n영상 설명: {video_embedding_2}",
    f"{system_prompt}\n\n영상 설명: {video_embedding_3}",
]
```

### 3.3 LMCache

**LMCache**는 KV Cache를 디스크나 분산 스토리지에 저장하여 메모리를 절약하는 기술입니다.

```
LMCache 아키텍처:

┌─────────────────────────────────────────────────────────────────┐
│                    LMCache 시스템                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU Memory (Hot Cache)                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  최근 사용된 KV Cache (자주 접근)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↕ (Eviction/Load)                   │
│  CPU Memory (Warm Cache)                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  중간 빈도 KV Cache                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↕ (Eviction/Load)                   │
│  SSD/Disk (Cold Cache)                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  오래된 KV Cache (필요시 로드)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  장점:                                                           │
│  - GPU 메모리 절약                                              │
│  - 더 긴 컨텍스트 지원                                          │
│  - 더 많은 동시 요청 처리                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Squeezed Attention

```
Squeezed Attention 개념:

기존 Attention:
- 모든 KV 토큰에 대해 계산
- 시간 복잡도: O(seq_len²)

Squeezed Attention:
- 중요한 KV 토큰만 선택적으로 계산
- 덜 중요한 토큰은 압축 또는 제거

방법:
1. Attention Score 기반 중요도 계산
2. Top-K 중요 토큰만 유지
3. 나머지는 평균으로 압축

효과:
- 메모리: 50-70% 절감
- 속도: 1.5-2x 향상
- 품질: 1-2% 저하 (trade-off)
```

---

## 4. 양자화 전략

### 4.1 양자화 개요

```
양자화 수준 비교:

FP32 (32-bit):
████████████████████████████████ 32 bits
정밀도: 최고 | 메모리: 100% | 속도: 기준

FP16 (16-bit):
████████████████ 16 bits
정밀도: 높음 | 메모리: 50% | 속도: 2x

INT8 (8-bit):
████████ 8 bits
정밀도: 좋음 | 메모리: 25% | 속도: 2-4x

INT4 (4-bit):
████ 4 bits
정밀도: 양호 | 메모리: 12.5% | 속도: 3-6x
```

### 4.2 AWQ vs GPTQ 비교

| 특성 | AWQ | GPTQ |
|------|-----|------|
| **원리** | Activation-aware Weight Quantization | Post-Training Quantization |
| **캘리브레이션** | 활성화 기반 중요도 분석 | 헤시안 기반 최적화 |
| **속도** | 빠른 양자화 | 느린 양자화 |
| **품질** | 약간 높음 | 좋음 |
| **메모리** | 동일 | 동일 |
| **추천 상황** | 빠른 배포 | 최고 품질 필요 |

### 4.3 AWQ 적용

```python
# AWQ 양자화된 모델 사용

# 방법 1: 사전 양자화된 모델 로드
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Instruct-AWQ",
    device_map="auto",
    trust_remote_code=True,
)

# 방법 2: 직접 양자화
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    device_map="auto",
)

# 양자화 설정
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

# 양자화 실행 (캘리브레이션 데이터 필요)
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_dataset,
)

# 저장
model.save_quantized("Qwen3-8B-AWQ")
```

### 4.4 GPTQ 적용

```python
# GPTQ 양자화

from transformers import AutoModelForCausalLM, GPTQConfig

# GPTQ 설정
gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",  # 캘리브레이션 데이터셋
    tokenizer=tokenizer,
    group_size=128,
    desc_act=True,  # 활성화 기반 정렬
)

# 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    quantization_config=gptq_config,
    device_map="auto",
)
```

### 4.5 양자화 선택 가이드

```python
def select_quantization(gpu_vram, quality_priority):
    """
    GPU VRAM과 품질 우선순위에 따른 양자화 선택
    """
    if gpu_vram <= 16:  # T4
        return {
            "method": "4bit",
            "type": "nf4",  # bitsandbytes nf4
            "compute_dtype": "float16",
            "double_quant": True,  # 메모리 절약
        }
    elif gpu_vram <= 24:  # L4
        if quality_priority:
            return {
                "method": "awq",
                "bits": 4,
                "group_size": 128,
            }
        else:
            return {
                "method": "4bit",
                "type": "nf4",
                "compute_dtype": "float16",
            }
    elif gpu_vram <= 40:  # A100
        if quality_priority:
            return {
                "method": "8bit",  # INT8
                "type": "llm.int8()",
            }
        else:
            return {
                "method": "awq",
                "bits": 4,
            }
    else:  # H100
        if quality_priority:
            return {"method": "fp16"}  # 양자화 없음
        else:
            return {
                "method": "awq",
                "bits": 4,
            }
```

---

## 5. Speculative Decoding

### 5.1 Speculative Decoding 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                    Speculative Decoding                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  기존 자기회귀 생성:                                            │
│  [Input] → T1 → T2 → T3 → T4 → T5 (순차적, 느림)               │
│           ↓    ↓    ↓    ↓    ↓                                 │
│          LLM  LLM  LLM  LLM  LLM (매번 full forward)            │
│                                                                  │
│  Speculative Decoding:                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Draft Model (작은 모델)로 K개 토큰 추측                  │  │
│  │    [Input] → T1' → T2' → T3' → T4' (빠름)                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 2. Target Model (큰 모델)로 한 번에 검증                    │  │
│  │    [Input, T1', T2', T3', T4'] → 병렬 검증                 │  │
│  │    결과: T1' ✓, T2' ✓, T3' ✗ (여기서 다시 시작)           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 3. 결과                                                     │  │
│  │    - 검증 통과: 한 번의 forward로 여러 토큰 확정            │  │
│  │    - 검증 실패: 올바른 토큰으로 대체 후 계속                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  속도 향상: 2-3x (Draft 모델 정확도에 따라)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Speculative Decoding 구현

```python
# vLLM에서 Speculative Decoding 설정

from vllm import LLM, SamplingParams

# Target Model (큰 모델)
llm = LLM(
    model="Qwen/Qwen3-32B-Instruct",
    
    # Speculative Decoding 설정
    speculative_model="Qwen/Qwen3-8B-Instruct",  # Draft 모델
    num_speculative_tokens=5,  # 한 번에 추측할 토큰 수
    speculative_draft_tensor_parallel_size=1,
    
    # 기타 설정
    gpu_memory_utilization=0.9,
)

# 추론
sampling_params = SamplingParams(
    max_tokens=256,
    temperature=0.7,
)

outputs = llm.generate(prompts, sampling_params)
```

### 5.3 Draft 모델 선택 가이드

```python
# Draft 모델 선택 기준

draft_model_guide = {
    "Qwen3-32B": {
        "draft": "Qwen3-8B",
        "speedup": "2.5-3x",
        "acceptance_rate": "70-80%",
    },
    "Qwen3-14B": {
        "draft": "Qwen3-8B",
        "speedup": "1.8-2.2x",
        "acceptance_rate": "75-85%",
    },
    "LLaVA-NeXT-Video-34B": {
        "draft": "LLaVA-NeXT-Video-7B",
        "speedup": "2.2-2.8x",
        "acceptance_rate": "65-75%",
    },
}

# 선택 기준:
# 1. 같은 아키텍처/토크나이저 사용
# 2. Target의 1/4 ~ 1/5 크기
# 3. 동일한 학습 데이터로 사전학습된 모델
```

### 5.4 Speculative Decoding 효과

| Target Model | Draft Model | Speedup | Memory 증가 |
|--------------|-------------|---------|-------------|
| 32B | 7B | 2.5-3.0x | +15% |
| 14B | 7B | 1.8-2.2x | +25% |
| 7B | 1B (추가 학습) | 1.5-1.8x | +10% |

---

## 6. 생성 파라미터 튜닝

### 6.1 Temperature

```
Temperature 효과:

낮은 Temperature (0.1-0.3):
- 결정적 (deterministic) 출력
- 반복적인 패턴
- 안전한 선택

logits: [2.0, 1.5, 1.0, 0.5]
T=0.1:  [0.99, 0.01, 0.00, 0.00]  → 거의 항상 첫 번째 선택

중간 Temperature (0.5-0.8):
- 균형 잡힌 출력
- 적절한 다양성

T=0.5:  [0.70, 0.20, 0.08, 0.02]  → 다양성 있음

높은 Temperature (1.0-1.5):
- 창의적 출력
- 예측 불가능
- 오류 가능성

T=1.5:  [0.35, 0.30, 0.22, 0.13]  → 매우 다양
```

```python
# 권장 Temperature 설정

temperature_guide = {
    "factual_description": 0.3,  # 사실적 묘사
    "balanced": 0.7,              # 균형 (기본값)
    "creative": 1.0,              # 창의적 묘사
}

# 한국어 캡셔닝 권장: 0.5-0.7
# 이유: 사실적이면서도 자연스러운 문장 생성
```

### 6.2 Top-p (Nucleus Sampling)

```
Top-p Sampling:

확률 분포: [0.4, 0.3, 0.15, 0.1, 0.03, 0.02]

Top-p = 0.9:
누적 확률이 0.9가 될 때까지의 토큰만 고려
[0.4, 0.3, 0.15, 0.1] = 0.95 > 0.9
→ 상위 3개 토큰 중에서 샘플링: [0.4, 0.3, 0.15]

장점:
- 동적으로 후보 수 조절
- 낮은 확률 토큰 자동 제외
- Top-k보다 적응적
```

```python
# Top-p 설정
top_p_guide = {
    "conservative": 0.8,   # 안전한 선택
    "balanced": 0.9,       # 균형 (기본값)
    "diverse": 0.95,       # 다양한 표현
}
```

### 6.3 Top-k

```
Top-k Sampling:

확률 분포: [0.4, 0.3, 0.15, 0.1, 0.03, 0.02]

Top-k = 3:
상위 3개 토큰만 고려
[0.4, 0.3, 0.15] → 정규화 → [0.47, 0.35, 0.18]

장점:
- 단순하고 빠름
- 예측 가능한 다양성

단점:
- 고정된 후보 수
- 확률 분포와 무관하게 적용
```

```python
# Top-k 설정 (Top-p와 함께 사용 권장)
top_k_guide = {
    "focused": 10,   # 집중된 선택
    "balanced": 50,  # 균형
    "diverse": 100,  # 다양한
}
```

### 6.4 Repetition Penalty

```
Repetition Penalty:

개념:
이미 생성된 토큰의 확률을 낮춤

수식:
if token in generated:
    logit[token] = logit[token] / penalty

예시 (penalty = 1.2):
원래 logits:  [2.0, 1.5, 1.0]  (첫 번째가 이미 생성됨)
조정 후:      [1.67, 1.5, 1.0]  (첫 번째 확률 감소)

효과:
- 반복 방지
- 다양한 어휘 사용

주의:
- 너무 높으면 부자연스러운 문장
- 너무 낮으면 반복 발생
```

```python
# Repetition Penalty 설정
repetition_penalty_guide = {
    "minimal": 1.0,    # 페널티 없음
    "light": 1.1,      # 가벼운 페널티
    "balanced": 1.2,   # 균형 (권장)
    "strong": 1.5,     # 강한 페널티
}

# 한국어 특성상 1.1-1.2 권장
# 이유: 조사, 어미 등이 반복될 수 있음
```

### 6.5 Length Penalty

```
Length Penalty (Beam Search):

개념:
생성 길이에 따른 점수 조정

수식:
score = log_prob / (length ** length_penalty)

length_penalty > 1.0: 긴 문장 선호
length_penalty = 1.0: 중립
length_penalty < 1.0: 짧은 문장 선호
```

```python
# Length Penalty 설정
length_penalty_guide = {
    "short": 0.8,     # 짧은 캡션
    "neutral": 1.0,   # 중립 (기본값)
    "long": 1.2,      # 긴 캡션
}

# 상세 묘사 태스크: 1.0-1.2 권장
```

### 6.6 권장 생성 파라미터 조합

```python
# 한국어 배경영상 캡셔닝 권장 설정

generation_config = {
    # 기본 파라미터
    "max_new_tokens": 256,
    "min_new_tokens": 50,
    
    # 샘플링 파라미터
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    
    # 반복 방지
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,  # 3-gram 반복 방지
    
    # 길이 조절
    "length_penalty": 1.0,
    
    # 종료 조건
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

# 품질 우선 설정 (느리지만 정확)
quality_config = {
    "do_sample": False,  # Greedy decoding
    "num_beams": 4,      # Beam search
    "early_stopping": True,
    "length_penalty": 1.2,
}

# 속도 우선 설정 (빠르지만 다양성)
speed_config = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 128,
}
```

---

## 7. 배치 추론 최적화

### 7.1 동적 배칭

```python
# vLLM 동적 배칭

# vLLM은 자동으로 동적 배칭 수행
# - 요청이 들어오면 큐에 추가
# - 가능한 많은 요청을 하나의 배치로 묶음
# - 연속 배칭 (Continuous Batching) 지원

llm = LLM(
    model="Qwen/Qwen3-8B-Instruct",
    max_num_seqs=32,  # 최대 동시 시퀀스
    max_num_batched_tokens=4096,  # 배치당 최대 토큰
)

# 여러 요청 동시 처리
prompts = [
    "영상 설명: ...",  # Request 1
    "영상 설명: ...",  # Request 2
    "영상 설명: ...",  # Request 3
    # ... 더 많은 요청
]

# 자동으로 최적 배치 크기 결정
outputs = llm.generate(prompts, sampling_params)
```

### 7.2 연속 배칭 (Continuous Batching)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Continuous Batching                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  기존 Static Batching:                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Batch 1: [Req1] [Req2] [Req3] → 모두 완료될 때까지 대기  │    │
│  │          ████████████████████████████████████████        │    │
│  │          ████████████████████                            │    │
│  │          ████████████████████████████████████████████████│    │
│  │                              ↓ 모든 요청 완료             │    │
│  │ Batch 2: [Req4] [Req5] → 시작                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Continuous Batching:                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ [Req1] ████████████████████ ← 완료 즉시 새 요청 추가    │    │
│  │ [Req2] ████████████████████████████████████████████████ │    │
│  │ [Req3] ████████████████████████████████ ← Req4 추가     │    │
│  │                    [Req4] ██████████████████████████████ │    │
│  │                                    [Req5] ██████████████ │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  장점:                                                           │
│  - GPU 활용률 최대화                                            │
│  - 지연시간 감소                                                │
│  - 처리량 증가                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. GPU별 추론 설정 요약

### 8.1 T4 (16GB)

```python
t4_inference_config = {
    "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    
    # 양자화
    "quantization": "4bit",
    "bnb_4bit_compute_dtype": "float16",
    
    # vLLM 설정
    "gpu_memory_utilization": 0.85,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 2048,
    
    # 생성 파라미터
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    
    # 최적화
    "enable_prefix_caching": True,
}

# 예상 성능:
# - 처리량: ~5 req/s
# - 지연시간: ~2s/req
```

### 8.2 L4 (24GB)

```python
l4_inference_config = {
    "model": "Qwen/Qwen3-8B-Instruct",
    
    # 양자화
    "quantization": "awq",
    
    # vLLM 설정
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 4096,
    
    # 생성 파라미터
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    
    # 최적화
    "enable_prefix_caching": True,
}

# 예상 성능:
# - 처리량: ~12 req/s
# - 지연시간: ~1.5s/req
```

### 8.3 A100 (40GB)

```python
a100_inference_config = {
    "model": "Qwen/Qwen3-14B-Instruct",
    
    # 양자화
    "quantization": "awq",
    
    # vLLM 설정
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 8192,
    
    # 생성 파라미터
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    
    # 최적화
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
}

# 예상 성능:
# - 처리량: ~25 req/s
# - 지연시간: ~0.8s/req
```

### 8.4 H100 (80GB)

```python
h100_inference_config = {
    "model": "Qwen/Qwen3-32B-Instruct",
    
    # 양자화 (선택적)
    "quantization": "awq",  # 또는 None for FP16
    
    # vLLM 설정
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 16384,
    
    # Speculative Decoding
    "speculative_model": "Qwen/Qwen3-8B-Instruct",
    "num_speculative_tokens": 5,
    
    # 생성 파라미터
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    
    # 최적화
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
}

# 예상 성능:
# - 처리량: ~50 req/s
# - 지연시간: ~0.5s/req
```

---

## 9. 참고 자료

### 논문
- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

### 코드 레포지토리
- [vLLM](https://github.com/vllm-project/vllm)
- [llm-compressor](https://github.com/vllm-project/llm-compressor)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
