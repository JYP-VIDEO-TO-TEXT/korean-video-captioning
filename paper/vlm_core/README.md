# VLM Core - Vision-Language Model 핵심

> 💡 **핵심 질문**: 이미지/비디오를 어떻게 이해하고 자연어로 설명할 것인가?

Vision-Language Model의 발전 흐름과 핵심 논문들을 정리합니다.

---

## 🎯 이 카테고리의 목표

VLM은 **시각 정보**를 **자연어**로 변환하는 모델입니다. 우리 프로젝트에서는 비디오를 입력받아 한국어 캡션을 생성해야 합니다.

```mermaid
flowchart LR
    subgraph Input["입력"]
        Video[🎬 비디오<br/>한국 배경영상]
    end

    subgraph Model["VLM"]
        Process[LLaVA-NeXT-Video<br/>시각 이해 + 언어 생성]
    end

    subgraph Output["출력"]
        Caption[📝 한국어 캡션<br/>'푸른 바다 위로<br/>하얀 파도가...']
    end

    Video --> Process --> Caption

    style Process fill:#4dabf7,stroke:#1971c2
```

---

## 📊 VLM 발전 흐름

```mermaid
flowchart TB
    subgraph Era1["2023년: VLM의 시작"]
        LLaVA[🌟 LLaVA<br/>Visual Instruction Tuning<br/>────────────────<br/>• GPT-4로 학습 데이터 생성<br/>• 간단한 Linear Projector<br/>• 2-Stage Training 제안]
    end

    subgraph Era2["2024년 초: 고해상도"]
        LLaVA_NeXT[📸 LLaVA-NeXT<br/>AnyRes 고해상도<br/>────────────────<br/>• 다양한 해상도/종횡비<br/>• 최대 672×672 지원<br/>• 더 큰 LLM 옵션]
        
        Video_LLaVA[🎥 Video-LLaVA<br/>통합 비주얼 학습<br/>────────────────<br/>• 이미지+비디오 동시 학습<br/>• LanguageBind 인코더<br/>• 시간적 맥락 이해]
    end

    subgraph Era3["2024년 중: 비디오 특화"]
        LLaVA_Video[⭐ LLaVA-NeXT-Video<br/>Zero-shot 비디오<br/>────────────────<br/>• 이미지만으로 학습해도<br/>  비디오 이해 가능<br/>• 프레임별 인코딩<br/>• DPO로 품질 향상]
    end

    LLaVA --> LLaVA_NeXT
    LLaVA_NeXT --> LLaVA_Video
    LLaVA_NeXT --> Video_LLaVA

    subgraph Project["🎯 우리 프로젝트"]
        Apply[LLaVA-NeXT-Video 기반<br/>+ 한국어 Fine-tuning<br/>+ Qwen3 LLM 교체]
    end

    LLaVA_Video ==> Apply

    style LLaVA_Video fill:#4dabf7,stroke:#1971c2,stroke-width:3px
    style Apply fill:#ff6b6b,stroke:#c92a2a,color:#fff
```

---

## 🏗️ VLM 아키텍처 상세

### 공통 구조

모든 LLaVA 계열 모델은 동일한 기본 구조를 공유합니다:

```mermaid
flowchart TB
    subgraph Input["📥 입력 처리"]
        Img[이미지/비디오]
        Text[텍스트 프롬프트]
    end

    subgraph Vision["👁️ Vision Encoder"]
        VE[CLIP ViT-L/14<br/>────────────<br/>• 이미지 → 패치 분할<br/>• 각 패치 → 토큰<br/>• 336px: 576 토큰]
    end

    subgraph Projection["🔗 Projector"]
        Proj[Linear Layer<br/>────────────<br/>• Vision 차원 → LLM 차원<br/>• 768-d → 4096-d<br/>• 학습 대상]
    end

    subgraph Language["🧠 LLM Backbone"]
        LLM[Vicuna-7B / Qwen3<br/>────────────<br/>• Visual + Text 토큰 입력<br/>• Auto-regressive 생성<br/>• LoRA로 효율적 학습]
    end

    subgraph Output["📤 출력"]
        Caption[생성된 캡션]
    end

    Img --> VE
    VE --> |Visual Tokens| Proj
    Proj --> LLM
    Text --> |Text Tokens| LLM
    LLM --> Caption

    style VE fill:#e7f5ff,stroke:#1971c2
    style Proj fill:#fff3bf,stroke:#f59f00
    style LLM fill:#d3f9d8,stroke:#2f9e44
```

### 학습 시 각 컴포넌트 역할

```mermaid
flowchart LR
    subgraph Frozen["❄️ Frozen (학습 안함)"]
        VE[Vision Encoder<br/>CLIP]
    end

    subgraph Trainable["🔥 Trainable (학습)"]
        Proj[Projector<br/>Linear]
        LoRA[LLM LoRA<br/>Adapter]
    end

    VE --> Proj --> LoRA

    style VE fill:#e7f5ff
    style Proj fill:#fff3bf
    style LoRA fill:#ffe3e3
```

---

## 📐 모델별 상세 비교

### 입력 처리 방식

```mermaid
flowchart TB
    subgraph LLaVA_Input["LLaVA: 단일 이미지"]
        I1[336×336] --> T1[576 tokens]
    end

    subgraph NeXT_Input["LLaVA-NeXT: AnyRes"]
        I2[1024×768] --> Split[Grid 분할]
        Split --> G1[336×336 ×4]
        Split --> G2[Thumbnail ×1]
        G1 --> T2[2,304 tokens]
        G2 --> T3[576 tokens]
        T2 --> Total2[총 2,880 tokens]
        T3 --> Total2
    end

    subgraph Video_Input["LLaVA-NeXT-Video: 프레임"]
        V[8 frames] --> F1[Frame 1: 576t]
        V --> F2[Frame 2: 576t]
        V --> F8[Frame 8: 576t]
        F1 --> Pool[Spatial Pool 2×2]
        F2 --> Pool
        F8 --> Pool
        Pool --> T4[8 × 144 = 1,152 tokens]
    end

    style Total2 fill:#fff3bf
    style T4 fill:#d3f9d8
```

### 상세 비교표

| 특성 | LLaVA | LLaVA-NeXT | LLaVA-NeXT-Video | Video-LLaVA |
|------|-------|------------|------------------|-------------|
| **입력** | 이미지 | 이미지 (고해상도) | **비디오** ⭐ | 이미지+비디오 |
| **Vision Encoder** | CLIP-L/14 | CLIP-L/14@336 | CLIP-L/14@336 | LanguageBind |
| **해상도** | 224/336 | 최대 672 | 336/frame | 336 |
| **Projector** | Linear | Linear | Linear + Pool | Linear |
| **LLM** | Vicuna-7B | 다양함 | Vicuna-7B | Vicuna-7B |
| **비디오 지원** | ❌ | ❌ | ✅ (zero-shot) | ✅ (native) |
| **메모리 (추론)** | ~14GB | ~16GB | ~14GB | ~14GB |

---

## 🔑 핵심 개념 상세 설명

### 1. Visual Instruction Tuning (LLaVA)

기존의 단순 캡셔닝을 넘어, **다양한 질문**에 답할 수 있도록 학습합니다.

```mermaid
flowchart TB
    subgraph Old["기존 방식"]
        O_Img[🖼️ 고양이 이미지] --> O_Cap["A cat sitting on a couch"]
    end

    subgraph New["LLaVA 방식"]
        N_Img[🖼️ 고양이 이미지]
        
        N_Img --> Q1["Q: 상세히 설명해주세요"]
        Q1 --> A1["A: 주황색 털을 가진 고양이가<br/>회색 소파 위에 편안하게..."]
        
        N_Img --> Q2["Q: 고양이가 뭘 하고 있나요?"]
        Q2 --> A2["A: 고양이가 소파에 앉아서<br/>휴식을 취하고 있습니다."]
        
        N_Img --> Q3["Q: 배경에 무엇이 있나요?"]
        Q3 --> A3["A: 뒤에 창문이 있고<br/>햇빛이 들어오고 있습니다."]
    end

    style New fill:#d3f9d8
```

### 2. AnyRes (LLaVA-NeXT)

다양한 해상도와 종횡비를 효율적으로 처리합니다.

```mermaid
flowchart TB
    subgraph Input["원본 이미지"]
        Original["1024 × 768<br/>(가로로 긴 이미지)"]
    end

    subgraph Process["AnyRes 처리"]
        direction TB
        
        subgraph Grid["Grid 분할"]
            G1["336×336"] 
            G2["336×336"]
            G3["336×336"]
            G4["336×336"]
        end
        
        Thumb["Thumbnail<br/>336×336<br/>(전체 맥락)"]
    end

    subgraph Tokens["Visual Tokens"]
        T_Grid["Grid: 576×4 = 2,304"]
        T_Thumb["Thumb: 576"]
        T_Total["총: 2,880 tokens"]
    end

    Original --> Grid
    Original --> Thumb
    Grid --> T_Grid
    Thumb --> T_Thumb
    T_Grid --> T_Total
    T_Thumb --> T_Total

    style T_Total fill:#fff3bf
```

### 3. Video Frame Processing (LLaVA-NeXT-Video)

비디오를 프레임 시퀀스로 처리하면서 메모리를 효율적으로 관리합니다.

```mermaid
flowchart TB
    subgraph Video["🎬 입력 비디오"]
        V["10초 영상<br/>240 frames"]
    end

    subgraph Sample["📊 프레임 샘플링"]
        S["Uniform Sampling<br/>8 frames 선택"]
    end

    subgraph Encode["👁️ 프레임별 인코딩"]
        F1["Frame 1"] --> E1["CLIP → 576 tokens"]
        F2["Frame 2"] --> E2["CLIP → 576 tokens"]
        F3["..."] --> E3["..."]
        F8["Frame 8"] --> E8["CLIP → 576 tokens"]
    end

    subgraph Pool["🔄 Spatial Pooling"]
        P["2×2 Average Pool<br/>576 → 144 tokens/frame"]
    end

    subgraph Final["📤 최종 입력"]
        Total["8 × 144 = 1,152 tokens<br/>+ Text tokens<br/>→ LLM"]
    end

    V --> S --> Encode
    E1 --> Pool
    E2 --> Pool
    E8 --> Pool
    Pool --> Total

    style Total fill:#d3f9d8
```

---

## 🎯 우리 프로젝트 적용

### 선택: LLaVA-NeXT-Video-7B

```mermaid
flowchart TB
    subgraph Why["✅ 선택 이유"]
        R1["Zero-shot 비디오 이해"]
        R2["Hugging Face 즉시 사용"]
        R3["커뮤니티 지원 풍부"]
        R4["4-bit로 T4 구동 가능"]
    end

    subgraph How["📝 적용 방법"]
        H1["Stage 1: Projector 정렬<br/>(선택적)"]
        H2["Stage 2: LoRA Fine-tuning<br/>(필수)"]
        H3["한국어 캡션 데이터 사용"]
    end

    Why --> How

    style Why fill:#d3f9d8
    style How fill:#fff3bf
```

### 프롬프트 형식

```python
# 기본 프롬프트
prompt = "USER: <video>이 영상을 한국어로 상세히 묘사해주세요. ASSISTANT:"

# 상세 프롬프트 (더 나은 결과)
prompt = """USER: <video>
이 영상에 나타난 장면을 한국어로 상세하게 묘사해주세요.
다음 요소들을 포함해주세요:
- 주요 배경과 환경
- 눈에 띄는 특징
- 전반적인 분위기
ASSISTANT:"""
```

### Fine-tuning 전략

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: Feature Alignment"]
        D1["📊 데이터<br/>한국어 이미지-캡션"]
        T1["🎯 학습<br/>Projector만"]
        G1["💡 목표<br/>Vision-Language 정렬"]
        
        D1 --> T1 --> G1
    end

    subgraph Stage2["Stage 2: Instruction Tuning"]
        D2["📊 데이터<br/>AI-Hub 비디오 캡셔닝"]
        T2["🎯 학습<br/>Projector + LLM (LoRA)"]
        G2["💡 목표<br/>비디오 캡셔닝 능력"]
        
        D2 --> T2 --> G2
    end

    Stage1 --> Stage2

    style Stage1 fill:#e7f5ff
    style Stage2 fill:#fff3bf
```

---

## 📚 논문 목록

| 파일 | 논문 | 핵심 포인트 | 중요도 |
|------|------|------------|--------|
| [llava.md](llava.md) | LLaVA (2023) | VLM의 기초, 2-Stage Training | ⭐⭐⭐ |
| [llava_next.md](llava_next.md) | LLaVA-NeXT (2024) | AnyRes, 고해상도 처리 | ⭐⭐ |
| [llava_next_video.md](llava_next_video.md) | LLaVA-NeXT-Video (2024) | **우리 기본 모델** | ⭐⭐⭐⭐⭐ |
| [video_llava.md](video_llava.md) | Video-LLaVA (2024) | 대안 모델, 통합 학습 | ⭐⭐ |

---

## 🔗 추가 참고 자료

- [LLaVA 공식 GitHub](https://github.com/haotian-liu/LLaVA)
- [LLaVA-NeXT Blog](https://llava-vl.github.io/blog/)
- [Hugging Face LLaVA Collection](https://huggingface.co/collections/llava-hf/)
