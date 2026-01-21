# 논문 정리

대한민국 배경영상 캡셔닝 프로젝트에 필요한 핵심 논문들을 정리합니다.

---

## 🎯 프로젝트 목표

> **AI-Hub 베이스라인 METEOR 0.3052 → 0.40+ 달성**

이를 위해 Vision-Language Model의 최신 연구들을 분석하고 적용합니다.

---

## 📊 전체 기술 발전 흐름

### Vision-Language Model 생태계

```mermaid
flowchart TB
    subgraph Foundation["🏗️ 기반 기술"]
        direction LR
        ViT[ViT<br/>2020] --> CLIP[CLIP<br/>2021]
        GPT[GPT 계열] --> LLaMA[LLaMA<br/>2023]
    end

    subgraph Vision["👁️ Vision Encoder 발전"]
        CLIP --> SigLIP[SigLIP<br/>2023<br/>다국어+Sigmoid]
        CLIP --> DINO[DINO<br/>2021]
        DINO --> DINOv2[DINOv2<br/>2023<br/>Dense Features]
        DINOv2 --> DINOv3[DINOv3<br/>2024<br/>Gram Anchoring]
    end

    subgraph LLM["🧠 LLM 발전"]
        LLaMA --> Vicuna[Vicuna<br/>2023<br/>대화 특화]
        LLaMA --> Qwen[Qwen<br/>2023<br/>다국어]
        Qwen --> Qwen2[Qwen2<br/>2024<br/>GQA+128K]
        Qwen2 --> Qwen3[Qwen3<br/>2025<br/>MoE+최고성능]
    end

    subgraph VLM["🔗 VLM 통합"]
        LLaVA[LLaVA<br/>2023<br/>Visual Instruction]
        LLaVA --> LLaVA_NeXT[LLaVA-NeXT<br/>2024<br/>AnyRes]
        LLaVA_NeXT --> LLaVA_Video[LLaVA-NeXT-Video<br/>2024<br/>비디오 특화]
        LLaVA_NeXT --> Video_LLaVA[Video-LLaVA<br/>2024<br/>통합 학습]
    end

    CLIP --> LLaVA
    Vicuna --> LLaVA

    subgraph Project["🎯 우리 프로젝트"]
        Apply[한국어 비디오 캡셔닝<br/>METEOR 0.40+ 목표]
    end

    LLaVA_Video ==> Apply
    Qwen3 -.-> Apply
    SigLIP -.-> Apply
    DINOv3 -.-> Apply

    style Apply fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style LLaVA_Video fill:#4dabf7,stroke:#1971c2
    style Qwen3 fill:#69db7c,stroke:#2f9e44
```

---

## 🔧 우리 프로젝트의 기술 스택

### 핵심 의존 관계

```mermaid
flowchart TB
    subgraph Core["📦 기반 모델"]
        Base[LLaVA-NeXT-Video-7B<br/>Hugging Face에서 제공]
    end

    subgraph Components["🧩 주요 컴포넌트"]
        Vision[Vision Encoder<br/>CLIP ViT-L/14<br/>📷 이미지→토큰]
        Proj[Projector<br/>Linear Layer<br/>🔗 공간 연결]
        LLM[LLM Backbone<br/>Vicuna-7B<br/>📝 텍스트 생성]
    end

    subgraph Upgrade["⬆️ 업그레이드 옵션"]
        Vision_Up[SigLIP<br/>다국어 이해 ↑<br/>─────────<br/>DINOv3<br/>세밀한 묘사 ↑]
        LLM_Up[Qwen3<br/>한국어 성능 ↑<br/>─────────<br/>MoE로 효율 ↑]
    end

    subgraph Training["🎓 학습 방법"]
        QLoRA[QLoRA<br/>4-bit 양자화<br/>메모리 90% 절약]
        Stage[2-Stage Training<br/>1. 정렬 학습<br/>2. 태스크 학습]
    end

    subgraph Deploy["🚀 배포 최적화"]
        AWQ[AWQ 양자화<br/>추론 속도 3x↑]
        vLLM[vLLM 서빙<br/>처리량 4x↑]
    end

    Base --> Vision
    Base --> Proj
    Base --> LLM

    Vision -.->|교체 가능| Vision_Up
    LLM -.->|교체 가능| LLM_Up

    Components --> Training
    Training --> Deploy

    style Base fill:#e7f5ff,stroke:#1971c2
    style QLoRA fill:#fff3bf,stroke:#f59f00
    style vLLM fill:#d3f9d8,stroke:#2f9e44
```

---

## 📚 카테고리별 논문 정리

### 1. [VLM Core](vlm_core/) - Vision-Language Model 핵심

> 💡 **핵심 질문**: 이미지/비디오를 어떻게 이해하고 설명할 것인가?

```mermaid
flowchart LR
    subgraph Evolution["VLM 진화"]
        A[LLaVA] -->|고해상도| B[LLaVA-NeXT]
        B -->|비디오| C[LLaVA-NeXT-Video]
    end
    
    C -->|우리 선택| D[한국어 캡셔닝]
    
    style D fill:#ff6b6b,stroke:#c92a2a,color:#fff
```

| 논문 | 연도 | 핵심 아이디어 | 우리 프로젝트 관련성 |
|------|------|--------------|-------------------|
| [LLaVA](vlm_core/llava.md) | 2023 | GPT-4로 학습 데이터 생성, 2-Stage 학습 | 학습 전략의 기초 |
| [LLaVA-NeXT](vlm_core/llava_next.md) | 2024 | AnyRes로 다양한 해상도 지원 | 고해상도 처리 방식 이해 |
| [LLaVA-NeXT-Video](vlm_core/llava_next_video.md) | 2024 | 이미지만으로 학습해도 비디오 이해 | ⭐ **우리 기본 모델** |
| [Video-LLaVA](vlm_core/video_llava.md) | 2024 | 이미지+비디오 동시 학습 | 대안 모델 |

---

### 2. [Vision Encoders](vision_encoders/) - 비전 인코더

> 💡 **핵심 질문**: 이미지에서 어떤 특징을 추출할 것인가?

```mermaid
flowchart TB
    subgraph Paradigm["학습 패러다임"]
        Sup[Supervised<br/>텍스트-이미지 쌍 필요]
        Self[Self-Supervised<br/>이미지만으로 학습]
    end

    subgraph Supervised_Models["Contrastive Learning"]
        CLIP[CLIP<br/>영어 중심]
        SigLIP[SigLIP<br/>109개 언어]
    end

    subgraph Self_Models["Self-Distillation"]
        DINOv2[DINOv2<br/>일반 특징]
        DINOv3[DINOv3<br/>Dense 특징]
    end

    Sup --> CLIP --> SigLIP
    Self --> DINOv2 --> DINOv3

    subgraph Choose["선택 기준"]
        C1[다국어 필요?<br/>→ SigLIP]
        C2[세밀한 묘사?<br/>→ DINOv3]
        C3[메모리 제약?<br/>→ CLIP]
    end

    style SigLIP fill:#69db7c,stroke:#2f9e44
    style DINOv3 fill:#4dabf7,stroke:#1971c2
```

| 논문 | 연도 | 학습 방식 | 강점 | 약점 |
|------|------|----------|------|------|
| [CLIP](vision_encoders/clip.md) | 2021 | Contrastive | Zero-shot, 안정적 | 영어 편향, Dense 약함 |
| [SigLIP](vision_encoders/siglip.md) | 2023 | Sigmoid CE | 다국어, 한국어↑ | 토큰 수 증가 |
| [DINOv2](vision_encoders/dinov2.md) | 2023 | Self-distill | Dense features | 텍스트 정렬 필요 |
| [DINOv3](vision_encoders/dinov3.md) | 2024 | Gram Anchor | 최고 품질 | ⚠️ 승인 필요 |

---

### 3. [LLM Backbones](llm_backbones/) - LLM 백본

> 💡 **핵심 질문**: 어떤 언어 모델이 한국어를 잘 생성하는가?

```mermaid
flowchart LR
    subgraph Performance["한국어 성능 순위"]
        direction TB
        P1["🥇 Qwen3-14B<br/>72.3%"]
        P2["🥈 Qwen3-8B<br/>68.5%"]
        P3["🥉 Qwen2-7B<br/>62.1%"]
        P4["4위 Qwen-7B<br/>52.1%"]
        P5["5위 Vicuna-7B<br/>38.2%"]
    end

    subgraph Choice["우리 선택"]
        C[Vicuna → Qwen3<br/>한국어 성능 2배↑]
    end

    P1 --> C
    
    style P1 fill:#ffd43b,stroke:#f59f00
    style C fill:#ff6b6b,stroke:#c92a2a,color:#fff
```

| 논문 | 연도 | 파라미터 | 한국어 MMLU | 특징 |
|------|------|----------|------------|------|
| [LLaMA](llm_backbones/llama.md) | 2023 | 7B-65B | 34.5% | 오픈소스 시작 |
| [Qwen](llm_backbones/qwen.md) | 2023 | 7B-72B | 52.1% | 다국어 특화 |
| [Qwen2](llm_backbones/qwen2.md) | 2024 | 7B-72B | 62.1% | GQA, 128K 컨텍스트 |
| [Qwen3](llm_backbones/qwen3.md) | 2025 | 0.6B-235B | **72.3%** | ⭐ MoE, 최고 성능 |

---

### 4. [Training Methods](training_methods/) - 학습 기법

> 💡 **핵심 질문**: 제한된 GPU에서 어떻게 효율적으로 학습할 것인가?

```mermaid
flowchart TB
    subgraph Memory["GPU 메모리 사용량"]
        Full[Full Fine-tuning<br/>112GB 💀]
        LoRA[LoRA<br/>56GB 😓]
        QLoRA[QLoRA<br/>6GB ✨]
    end

    Full -->|파라미터 효율화| LoRA
    LoRA -->|4-bit 양자화| QLoRA

    subgraph Result["결과"]
        R[T4 16GB에서<br/>7B 모델 학습 가능!]
    end

    QLoRA --> R

    style QLoRA fill:#69db7c,stroke:#2f9e44
    style R fill:#ffd43b,stroke:#f59f00
```

| 논문 | 연도 | 메모리 절약 | 품질 | 우리 적용 |
|------|------|-----------|------|----------|
| [LoRA](training_methods/lora.md) | 2021 | ~50% | 좋음 | A100+ |
| [QLoRA](training_methods/qlora.md) | 2023 | ~90% | 좋음 | ⭐ T4/L4 필수 |
| [DoRA](training_methods/dora.md) | 2024 | ~50% | 더 좋음 | 선택적 |
| [2-Stage](training_methods/llava_2stage.md) | 2023 | - | - | ⭐ 학습 전략 |

---

### 5. [Inference Optimization](inference_opt/) - 추론 최적화

> 💡 **핵심 질문**: 학습된 모델을 어떻게 빠르게 서빙할 것인가?

```mermaid
flowchart LR
    subgraph Speed["속도 향상 조합"]
        Base[기본<br/>1x] -->|양자화| AWQ[+AWQ<br/>3x]
        AWQ -->|서빙| vLLM[+vLLM<br/>6x]
        vLLM -->|추론| Spec[+SpecDec<br/>8x+]
    end

    subgraph GPU["GPU별 적용"]
        T4[T4: 기본만]
        L4[L4: AWQ]
        A100[A100: AWQ+vLLM]
        H100[H100: 전부 적용]
    end

    style Spec fill:#69db7c,stroke:#2f9e44
    style H100 fill:#ffd43b,stroke:#f59f00
```

| 논문 | 연도 | 속도 향상 | 핵심 기술 | 적용 시점 |
|------|------|----------|----------|----------|
| [GPTQ](inference_opt/gptq.md) | 2022 | 2-3x | Post-training 양자화 | 배포 시 |
| [AWQ](inference_opt/awq.md) | 2023 | 3-4x | Activation-aware | ⭐ A100+ |
| [vLLM](inference_opt/vllm.md) | 2023 | 2-4x | PagedAttention | ⭐ A100+ |
| [Speculative](inference_opt/speculative_decoding.md) | 2023 | 2-3x | Draft-Verify | H100 |

---

## 🗺️ 프로젝트 로드맵

```mermaid
timeline
    title 프로젝트 진행 단계
    
    section Phase 1
        기본 구축 : LLaVA-NeXT-Video-7B 설정
                 : QLoRA 학습 환경 구성
                 : AI-Hub 데이터 전처리
    
    section Phase 2
        LLM 업그레이드 : Vicuna → Qwen3 교체
                      : 한국어 성능 향상 검증
                      : METEOR 0.35 목표
    
    section Phase 3
        Vision 업그레이드 : CLIP → SigLIP 또는 DINOv3
                         : Stage 1 재학습
                         : METEOR 0.40 목표
    
    section Phase 4
        배포 최적화 : AWQ 양자화
                   : vLLM 서빙
                   : API 서버 구축
```

---

## 💻 GPU별 권장 구성

```mermaid
flowchart TB
    subgraph T4["🟡 T4 (16GB)"]
        T4_V[Vision: CLIP]
        T4_L[LLM: Qwen3-4B]
        T4_T[학습: QLoRA r=8]
        T4_I[추론: 기본]
    end

    subgraph L4["🟢 L4 (24GB)"]
        L4_V[Vision: SigLIP]
        L4_L[LLM: Qwen3-8B]
        L4_T[학습: QLoRA r=16]
        L4_I[추론: AWQ]
    end

    subgraph A100["🔵 A100 (40GB)"]
        A100_V[Vision: DINOv3-L]
        A100_L[LLM: Qwen3-14B]
        A100_T[학습: LoRA r=32]
        A100_I[추론: AWQ+vLLM]
    end

    subgraph H100["🟣 H100 (80GB)"]
        H100_V[Vision: DINOv3-H]
        H100_L[LLM: Qwen3-32B]
        H100_T[학습: LoRA r=64]
        H100_I[추론: 전부]
    end

    style T4 fill:#fff3bf
    style L4 fill:#d3f9d8
    style A100 fill:#d0ebff
    style H100 fill:#e5dbff
```

---

## 📖 추천 학습 순서

### 🌱 입문자

```mermaid
flowchart LR
    A[1. LLaVA<br/>VLM 기본 개념] --> B[2. LoRA<br/>효율적 학습]
    B --> C[3. QLoRA<br/>실제 적용]
    
    style A fill:#e7f5ff
    style B fill:#fff3bf
    style C fill:#d3f9d8
```

### 🌳 심화 학습

```mermaid
flowchart TB
    subgraph Vision_Path["Vision 심화"]
        V1[CLIP] --> V2[SigLIP] --> V3[DINOv3]
    end
    
    subgraph LLM_Path["LLM 심화"]
        L1[LLaMA] --> L2[Qwen] --> L3[Qwen3]
    end
    
    subgraph Opt_Path["최적화 심화"]
        O1[vLLM] --> O2[AWQ] --> O3[SpecDec]
    end
```

### 🎯 프로젝트 직접 관련

1. **[LLaVA-NeXT-Video](vlm_core/llava_next_video.md)** - 우리 기본 모델
2. **[Qwen3](llm_backbones/qwen3.md)** - LLM 업그레이드 대상
3. **[QLoRA](training_methods/qlora.md)** - 학습 필수 기법
4. **[2-Stage Training](training_methods/llava_2stage.md)** - 학습 전략
