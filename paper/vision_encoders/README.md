# Vision Encoders - 비전 인코더

> 💡 **핵심 질문**: 이미지에서 어떤 특징을 추출해야 언어 모델이 잘 이해할 수 있는가?

이미지/비디오에서 시각적 특징을 추출하는 Vision Encoder의 발전 흐름을 정리합니다.

---

## 🎯 이 카테고리의 목표

Vision Encoder는 **픽셀 데이터**를 **의미 있는 벡터**로 변환합니다. 이 벡터가 LLM에 전달되어 텍스트 생성의 기반이 됩니다.

```mermaid
flowchart LR
    subgraph Input["입력"]
        Img["🖼️ 이미지<br/>336×336×3<br/>≈338K 값"]
    end

    subgraph Encoder["Vision Encoder"]
        VE["패치 분할 → Transformer<br/>특징 추출"]
    end

    subgraph Output["출력"]
        Tokens["🔢 Visual Tokens<br/>576×768<br/>≈442K 값<br/>(의미 압축됨)"]
    end

    Img --> VE --> Tokens

    style VE fill:#4dabf7,stroke:#1971c2
```

---

## 📊 발전 흐름

### 두 가지 학습 패러다임

```mermaid
flowchart TB
    subgraph ViT["🏗️ 기반: Vision Transformer (2020)"]
        Base["이미지를 패치로 분할<br/>Transformer로 처리"]
    end

    ViT --> Supervised
    ViT --> SelfSupervised

    subgraph Supervised["📝 Supervised Learning"]
        direction TB
        S_Desc["텍스트-이미지 쌍으로 학습<br/>언어와 정렬된 특징 추출"]
        
        CLIP["CLIP (2021)<br/>────────────<br/>• Contrastive Learning<br/>• 4억 이미지-텍스트 쌍<br/>• Zero-shot 분류 가능"]
        
        SigLIP["SigLIP (2023)<br/>────────────<br/>• Sigmoid Loss<br/>• 109개 언어 지원<br/>• 한국어 성능 ↑"]
        
        CLIP --> SigLIP
    end

    subgraph SelfSupervised["🔄 Self-Supervised Learning"]
        direction TB
        SS_Desc["이미지만으로 학습<br/>Dense한 특징 추출"]
        
        DINO["DINO (2021)<br/>────────────<br/>• Self-distillation<br/>• Teacher-Student"]
        
        DINOv2["DINOv2 (2023)<br/>────────────<br/>• 1.4억 이미지<br/>• Dense features"]
        
        DINOv3["DINOv3 (2024)<br/>────────────<br/>• Gram Anchoring<br/>• 최고 품질"]
        
        DINO --> DINOv2 --> DINOv3
    end

    subgraph Choice["🎯 선택 기준"]
        C1["다국어 필요? → SigLIP"]
        C2["세밀한 묘사? → DINOv3"]
        C3["메모리 제약? → CLIP"]
    end

    Supervised --> Choice
    SelfSupervised --> Choice

    style CLIP fill:#e7f5ff
    style SigLIP fill:#d3f9d8
    style DINOv3 fill:#fff3bf
```

---

## 🔬 학습 방식 상세 비교

### Contrastive Learning (CLIP, SigLIP)

```mermaid
flowchart TB
    subgraph Training["학습 과정"]
        subgraph Batch["배치 내 N개 쌍"]
            I1["🖼️ Image 1"] --- T1["📝 Text 1"]
            I2["🖼️ Image 2"] --- T2["📝 Text 2"]
            IN["🖼️ Image N"] --- TN["📝 Text N"]
        end

        subgraph Encoders["인코더"]
            VE["Vision<br/>Encoder"]
            TE["Text<br/>Encoder"]
        end

        subgraph Embed["임베딩 공간"]
            Matrix["유사도 행렬<br/>N×N"]
        end
    end

    I1 --> VE
    I2 --> VE
    IN --> VE
    T1 --> TE
    T2 --> TE
    TN --> TE
    VE --> Matrix
    TE --> Matrix

    subgraph Goal["학습 목표"]
        G["대각선(같은 쌍): 높은 유사도 ✅<br/>비대각선(다른 쌍): 낮은 유사도 ❌"]
    end

    Matrix --> Goal

    style Goal fill:#d3f9d8
```

**장점:**
- ✅ Zero-shot 분류 가능 (텍스트로 카테고리 지정)
- ✅ 언어와 정렬된 특징 (VLM에서 바로 사용)
- ✅ 검증된 성능

**단점:**
- ❌ 대량의 텍스트-이미지 쌍 필요
- ❌ Dense prediction (세그멘테이션 등) 약함
- ❌ 학습 데이터의 언어 편향

### Self-Supervised Learning (DINO 계열)

```mermaid
flowchart TB
    subgraph Training["학습 과정"]
        Img["🖼️ 원본 이미지"]
        
        subgraph Augment["데이터 증강"]
            Aug1["View 1<br/>(크롭, 색상 변환)"]
            Aug2["View 2<br/>(다른 크롭)"]
        end

        subgraph Models["모델"]
            Student["Student<br/>(학습 중)"]
            Teacher["Teacher<br/>(EMA 업데이트)"]
        end

        subgraph Output["출력"]
            S_Out["Student 출력"]
            T_Out["Teacher 출력"]
        end
    end

    Img --> Aug1 --> Student --> S_Out
    Img --> Aug2 --> Teacher --> T_Out

    subgraph Goal["학습 목표"]
        G["같은 이미지의 다른 뷰<br/>→ 같은 표현을 가지도록"]
    end

    S_Out --> Goal
    T_Out --> Goal

    style Goal fill:#fff3bf
```

**장점:**
- ✅ 텍스트 데이터 불필요
- ✅ Dense features (픽셀 수준 정보 보존)
- ✅ 세그멘테이션, 깊이 추정에 강함

**단점:**
- ❌ Zero-shot 분류 어려움
- ❌ VLM에서 추가 정렬 학습 필요
- ❌ 텍스트와 직접 연결 안 됨

---

## 📐 토큰 수와 메모리 영향

Vision Encoder의 출력 토큰 수는 GPU 메모리 사용량에 직접적인 영향을 미칩니다.

```mermaid
flowchart TB
    subgraph Comparison["모델별 토큰 수 (336×336 입력)"]
        CLIP_T["CLIP ViT-L/14@336<br/>────────────<br/>(336÷14)² = 576 tokens"]
        SigLIP_T["SigLIP-So400M@384<br/>────────────<br/>(384÷14)² = 729 tokens"]
        DINO_T["DINOv3 ViT-L@518<br/>────────────<br/>(518÷14)² = 1,369 tokens"]
    end

    subgraph Video["비디오 (8 frames)"]
        CLIP_V["CLIP<br/>576×8 = 4,608<br/>→ Pool → 1,152"]
        SigLIP_V["SigLIP<br/>729×8 = 5,832<br/>→ Pool → 1,458"]
        DINO_V["DINOv3<br/>1,369×8 = 10,952<br/>→ Pool → 2,738"]
    end

    CLIP_T --> CLIP_V
    SigLIP_T --> SigLIP_V
    DINO_T --> DINO_V

    subgraph Memory["메모리 영향"]
        M["DINOv3는 CLIP 대비<br/>~2.4배 더 많은 토큰<br/>→ Attention 연산량 증가"]
    end

    DINO_V --> M

    style CLIP_V fill:#d3f9d8
    style DINO_V fill:#ffe3e3
```

### 메모리 사용량 비교 (7B LLM 기준)

```mermaid
xychart-beta
    title "Vision Encoder별 추론 메모리 (GB)"
    x-axis ["CLIP", "SigLIP", "DINOv2", "DINOv3"]
    y-axis "메모리 (GB)" 0 --> 20
    bar [12, 14, 16, 18]
```

---

## 📊 상세 비교표

| 특성 | CLIP | SigLIP | DINOv2 | DINOv3 |
|------|------|--------|--------|--------|
| **학습 방식** | Contrastive | Sigmoid CE | Self-distill | Gram Anchor |
| **학습 데이터** | 4억 쌍 | 10억+ 쌍 | 1.4억 이미지 | 16.8억 이미지 |
| **텍스트 정렬** | ✅ 강함 | ✅ 강함 | ❌ 약함 | ❌ 약함 |
| **Dense features** | ⚠️ 약함 | ⚠️ 보통 | ✅ 강함 | ✅ 매우 강함 |
| **다국어** | ❌ 영어 위주 | ✅ 109개 언어 | N/A | N/A |
| **기본 해상도** | 224/336 | 384 | 518 | 518 |
| **VLM 적용** | 바로 사용 | 바로 사용 | 정렬 필요 | 정렬 필요 |
| **접근성** | ✅ 공개 | ✅ 공개 | ✅ 공개 | ⚠️ **승인 필요** |

---

## 🎯 우리 프로젝트 적용

### 선택 가이드

```mermaid
flowchart TB
    Start["Vision Encoder 선택"] --> Q1{"GPU 메모리는?"}
    
    Q1 -->|"T4 (16GB)"| CLIP_Choice["CLIP ViT-L/14@336<br/>안정적, 검증됨"]
    
    Q1 -->|"L4 (24GB)"| Q2{"한국어 성능 중요?"}
    Q2 -->|"예"| SigLIP_Choice["SigLIP-So400M<br/>다국어 이해 ↑"]
    Q2 -->|"아니오"| CLIP_Choice
    
    Q1 -->|"A100 (40GB)"| Q3{"최고 품질 필요?"}
    Q3 -->|"예"| DINOv3_Choice["DINOv3 ViT-L<br/>Dense features"]
    Q3 -->|"아니오"| SigLIP_Choice
    
    Q1 -->|"H100 (80GB)"| DINOv3_H["DINOv3 ViT-H<br/>최대 품질"]

    style CLIP_Choice fill:#e7f5ff
    style SigLIP_Choice fill:#d3f9d8
    style DINOv3_Choice fill:#fff3bf
    style DINOv3_H fill:#ffd43b
```

### 기본 선택: CLIP ViT-L/14@336

```mermaid
flowchart LR
    subgraph Why["✅ 선택 이유"]
        R1["LLaVA 기본 인코더"]
        R2["검증된 성능"]
        R3["메모리 효율적"]
        R4["추가 정렬 불필요"]
    end

    subgraph Code["코드"]
        C["from transformers import<br/>  CLIPVisionModel<br/><br/>model = CLIPVisionModel<br/>  .from_pretrained(<br/>    'openai/clip-vit-<br/>     large-patch14-336'<br/>  )"]
    end

    Why --> Code

    style Why fill:#d3f9d8
```

### 업그레이드 옵션 1: SigLIP

```mermaid
flowchart TB
    subgraph Pros["✅ 장점"]
        P1["한국어 텍스트 이해 ↑"]
        P2["다국어 정렬"]
        P3["CLIP과 유사한 메모리"]
    end

    subgraph Cons["❌ 단점"]
        C1["Projector 재학습 필요"]
        C2["토큰 수 약간 증가"]
    end

    subgraph When["🎯 권장 상황"]
        W["L4 이상에서<br/>한국어 성능 중시 시"]
    end

    Pros --> When
    Cons --> When

    style When fill:#d3f9d8
```

### 업그레이드 옵션 2: DINOv3

```mermaid
flowchart TB
    subgraph Pros["✅ 장점"]
        P1["세밀한 공간 정보"]
        P2["배경 묘사에 강함"]
        P3["Dense features"]
    end

    subgraph Cons["❌ 단점"]
        C1["⚠️ Meta 승인 필요"]
        C2["메모리 사용량 높음"]
        C3["텍스트 정렬 재학습"]
    end

    subgraph When["🎯 권장 상황"]
        W["A100/H100에서<br/>최고 품질 추구 시"]
    end

    Pros --> When
    Cons --> When

    style When fill:#fff3bf
    style C1 fill:#ffe3e3
```

---

## 🔄 Vision Encoder 교체 시 주의사항

```mermaid
flowchart TB
    subgraph Change["Vision Encoder 교체"]
        Old["CLIP"] --> New["SigLIP / DINOv3"]
    end

    subgraph Required["필요한 작업"]
        R1["1️⃣ Projector 차원 확인<br/>hidden_size가 다를 수 있음"]
        R2["2️⃣ Stage 1 재학습<br/>Vision-Language 재정렬"]
        R3["3️⃣ 전처리 변경<br/>해상도, 정규화 값"]
        R4["4️⃣ 메모리 재계산<br/>토큰 수 변화 반영"]
    end

    Change --> R1 --> R2 --> R3 --> R4

    style R2 fill:#ffe3e3
```

---

## 📚 논문 목록

| 파일 | 논문 | 핵심 포인트 | 중요도 |
|------|------|------------|--------|
| [clip.md](clip.md) | CLIP (2021) | VLM의 표준 Vision Encoder | ⭐⭐⭐⭐⭐ |
| [siglip.md](siglip.md) | SigLIP (2023) | 다국어 + Sigmoid Loss | ⭐⭐⭐⭐ |
| [dinov2.md](dinov2.md) | DINOv2 (2023) | Self-supervised, Dense | ⭐⭐⭐ |
| [dinov3.md](dinov3.md) | DINOv3 (2024) | Gram Anchoring, 최고 품질 | ⭐⭐⭐⭐ |

---

## 💻 GPU별 권장

| GPU | 권장 Vision Encoder | 이유 |
|-----|---------------------|------|
| **T4 (16GB)** | CLIP ViT-L/14@336 | 메모리 제약, 기본 선택 |
| **L4 (24GB)** | SigLIP-So400M@384 | 다국어 이점, 여유 있음 |
| **A100 (40GB)** | DINOv3 ViT-L@518 | Dense features, 고품질 |
| **H100 (80GB)** | DINOv3 ViT-H@518 | 최대 품질 |
