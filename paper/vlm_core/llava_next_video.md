# LLaVA-NeXT-Video: A Strong Zero-shot Video Understanding Model

> ⭐ **우리 프로젝트의 기본 모델**: 이미지 학습만으로 비디오를 이해하는 Zero-shot 능력

- **저자**: Haotian Liu et al.
- **기관**: ByteDance, University of Wisconsin-Madison
- **연도**: 2024
- **링크**: [Blog](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)

---

## 💡 핵심 기여

```mermaid
flowchart TB
    subgraph Contributions["LLaVA-NeXT-Video의 핵심 기여"]
        C1["1️⃣ Zero-shot Video<br/>────────────────────<br/>이미지만으로 학습해도<br/>비디오 이해 가능!"]
        
        C2["2️⃣ AnyRes for Video<br/>────────────────────<br/>프레임별 고해상도 처리<br/>+ Spatial Pooling"]
        
        C3["3️⃣ DPO Training<br/>────────────────────<br/>Direct Preference Opt.<br/>Hallucination 감소"]
        
        C4["4️⃣ 효율적 인코딩<br/>────────────────────<br/>프레임별 독립 인코딩<br/>메모리 효율적"]
    end

    style C1 fill:#d3f9d8
    style C2 fill:#fff3bf
    style C3 fill:#e7f5ff
    style C4 fill:#e5dbff
```

---

## 아키텍처

![VLM Architecture](../../../model_viz/outputs/vlm_architecture.png)

![Video Frame Processing](../../../model_viz/outputs/video_frame_processing.png)

### 전체 구조

```mermaid
flowchart TB
    subgraph Input["📥 비디오 입력"]
        Video["🎬 비디오<br/>T frames 추출"]
    end

    subgraph Sampling["📊 프레임 샘플링"]
        Sample["Uniform Sampling<br/>────────────────<br/>기본 8 frames<br/>(조절 가능: 4~32)"]
    end

    subgraph PerFrame["👁️ 프레임별 인코딩"]
        F1["Frame 1"] --> E1["CLIP<br/>576 tokens"]
        F2["Frame 2"] --> E2["CLIP<br/>576 tokens"]
        F3["..."] --> E3["..."]
        FT["Frame T"] --> ET["CLIP<br/>576 tokens"]
    end

    subgraph Pooling["🔄 Spatial Pooling"]
        Pool["2×2 Average Pool<br/>────────────────<br/>576 → 144 tokens/frame<br/>총: T × 144 tokens"]
    end

    subgraph Projector["🔗 Projector"]
        Proj["Linear Layer<br/>Visual → Language"]
    end

    subgraph LLM["🧠 LLM"]
        Model["Vicuna-7B<br/>+ Text Prompt"]
    end

    subgraph Output["📤 출력"]
        Caption["생성된 캡션"]
    end

    Video --> Sample --> PerFrame
    E1 --> Pool
    E2 --> Pool
    ET --> Pool
    Pool --> Proj --> LLM --> Caption

    style Pool fill:#fff3bf
    style LLM fill:#d3f9d8
```

### 토큰 수 계산

```mermaid
flowchart LR
    subgraph Calculation["토큰 수 계산"]
        C1["프레임당 원본<br/>336÷14 = 24<br/>24×24 = 576 tokens"]
        
        C2["Spatial Pooling 후<br/>2×2 average<br/>24÷2 = 12<br/>12×12 = 144 tokens"]
        
        C3["8 frames 기준<br/>144 × 8 = 1,152 tokens<br/>+ Text tokens"]
    end

    C1 --> C2 --> C3

    style C3 fill:#d3f9d8
```

---

## 📊 Zero-shot Video Understanding

### 왜 가능한가?

```mermaid
flowchart TB
    subgraph Why["Zero-shot이 가능한 이유"]
        W1["이미지 = 비디오의 한 프레임<br/>────────────────────<br/>이미지 이해 능력이<br/>비디오로 전이됨"]
        
        W2["프레임별 독립 인코딩<br/>────────────────────<br/>시간 순서대로 처리하면<br/>비디오 맥락 이해"]
        
        W3["LLM의 일반화 능력<br/>────────────────────<br/>여러 프레임 정보를<br/>통합하여 이해"]
    end

    W1 --> W2 --> W3

    subgraph Result["결과"]
        R["비디오 학습 없이도<br/>비디오 QA, 캡셔닝 가능!"]
    end

    W3 --> Result

    style Result fill:#d3f9d8
```

### 한계점

```mermaid
flowchart TB
    subgraph Limitations["한계점"]
        L1["❌ Temporal 관계 약함<br/>────────────────────<br/>프레임별 독립 인코딩<br/>→ 시간적 관계 암묵적"]
        
        L2["❌ 긴 비디오 어려움<br/>────────────────────<br/>프레임 수 제한<br/>정보 손실 가능"]
        
        L3["❌ 빠른 동작 캡처 어려움<br/>────────────────────<br/>Uniform sampling<br/>→ 중요 순간 놓칠 수 있음"]
    end

    style L1 fill:#ffe3e3
    style L2 fill:#ffe3e3
    style L3 fill:#ffe3e3
```

---

## 🎓 학습 전략

### Stage 1: Image Pre-training

```mermaid
flowchart TB
    subgraph Stage1["Stage 1: 이미지 학습"]
        D1["LLaVA-NeXT 이미지 모델<br/>그대로 사용"]
        N1["비디오 데이터 없이 학습!"]
    end

    style Stage1 fill:#e7f5ff
```

### Stage 2: Video Fine-tuning (Optional)

```mermaid
flowchart TB
    subgraph Stage2["Stage 2: 비디오 Fine-tuning (선택적)"]
        subgraph Data["📊 데이터"]
            D2["Video-ChatGPT<br/>ActivityNet-QA<br/>NExT-QA"]
        end

        subgraph Method["🎯 방법"]
            M1["Video Instruction Tuning"]
            M2["DPO (Preference Learning)"]
        end

        subgraph Goal["💡 목표"]
            G["비디오 특화 능력 강화<br/>Hallucination 감소"]
        end
    end

    Data --> Method --> Goal

    style Method fill:#fff3bf
```

### DPO (Direct Preference Optimization)

```mermaid
flowchart TB
    subgraph DPO["DPO Training"]
        Input["비디오 + 질문"]
        
        subgraph Responses["응답 쌍"]
            Good["✅ 선호 응답<br/>(정확한 설명)"]
            Bad["❌ 비선호 응답<br/>(Hallucination)"]
        end

        subgraph Training["학습"]
            T["선호 응답 확률 ↑<br/>비선호 응답 확률 ↓"]
        end

        Input --> Responses --> Training
    end

    subgraph Effect["효과"]
        E["Hallucination 감소<br/>더 정확한 캡션 생성"]
    end

    DPO --> Effect

    style Good fill:#d3f9d8
    style Bad fill:#ffe3e3
    style Effect fill:#d3f9d8
```

---

## 🎯 우리 프로젝트 적용

### 프레임 수 설정

```mermaid
flowchart TB
    subgraph FrameGuide["GPU별 프레임 수 권장"]
        T4["🟡 T4 (16GB)<br/>────────────<br/>frames: 4<br/>tokens: 576<br/>메모리 제약"]
        
        L4["🟢 L4 (24GB)<br/>────────────<br/>frames: 8 (기본)<br/>tokens: 1,152<br/>권장 설정"]
        
        A100["🔵 A100 (40GB)<br/>────────────<br/>frames: 16<br/>tokens: 2,304<br/>고품질"]
        
        H100["🟣 H100 (80GB)<br/>────────────<br/>frames: 32<br/>tokens: 4,608<br/>최대 품질"]
    end

    style T4 fill:#fff3bf
    style L4 fill:#d3f9d8
    style A100 fill:#d0ebff
    style H100 fill:#e5dbff
```

### 코드 예시

```python
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# 모델 로드
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
processor = LlavaNextVideoProcessor.from_pretrained(model_id)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 비디오 프레임 준비 (PIL Images 리스트)
frames = extract_frames(video_path, num_frames=8)

# 프롬프트
prompt = "USER: <video>이 영상을 한국어로 상세히 묘사해주세요. ASSISTANT:"

# 추론
inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
caption = processor.decode(outputs[0], skip_special_tokens=True)
```

### Fine-tuning 전략

```mermaid
flowchart TB
    subgraph Strategy["한국어 비디오 캡셔닝 Fine-tuning"]
        S1["Stage 1 (선택적)<br/>────────────────<br/>한국어 이미지-캡션으로<br/>Projector 정렬"]
        
        S2["Stage 2 (필수)<br/>────────────────<br/>AI-Hub 비디오 데이터로<br/>QLoRA Fine-tuning"]
        
        S3["프롬프트<br/>────────────────<br/>USER: <video><br/>이 영상을 한국어로<br/>상세히 묘사해주세요.<br/>ASSISTANT:"]
    end

    S1 --> S2 --> S3

    style S2 fill:#d3f9d8
```

---

## 📈 성능 (Zero-shot)

| Benchmark | LLaVA-NeXT-Video-7B | LLaVA-NeXT-Video-7B-DPO |
|-----------|---------------------|------------------------|
| **ActivityNet-QA** | 53.5 | **56.2** |
| **MSVD-QA** | 67.8 | **70.1** |
| **MSRVTT-QA** | 53.2 | **55.8** |
| **TGIF-QA** | 67.1 | **69.3** |

---

## ⚠️ 구현 시 주의점

```mermaid
flowchart TB
    subgraph Cautions["주의사항"]
        C1["1️⃣ 메모리 관리<br/>────────────────<br/>8 frames × 144 = 1,152 tokens<br/>4-bit 양자화 권장 (T4/L4)"]
        
        C2["2️⃣ 프레임 샘플링<br/>────────────────<br/>Uniform이 기본<br/>장면 변화 기반 adaptive 고려"]
        
        C3["3️⃣ Spatial Pooling<br/>────────────────<br/>기본 2×2<br/>메모리 부족 시 3×3 가능"]
        
        C4["4️⃣ 프롬프트 형식<br/>────────────────<br/>반드시 <video> 토큰 포함<br/>ASSISTANT: 로 끝내기"]
    end

    style C1 fill:#ffe3e3
    style C2 fill:#fff3bf
    style C3 fill:#e7f5ff
    style C4 fill:#d3f9d8
```

---

## 🔗 관련 리소스

- **Hugging Face**: 
  - `llava-hf/LLaVA-NeXT-Video-7B-hf`
  - `llava-hf/LLaVA-NeXT-Video-7B-DPO-hf` (DPO 적용)
  - `llava-hf/LLaVA-NeXT-Video-34B-hf` (대형 모델)
- **GitHub**: [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- **Blog**: [llava-vl.github.io](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)

---

## 📚 인용

```bibtex
@misc{liu2024llavanext,
  title={LLaVA-NeXT: A Strong Zero-shot Video Understanding Model},
  author={Liu, Haotian and others},
  year={2024},
  howpublished={\url{https://llava-vl.github.io/blog/2024-04-30-llava-next-video/}}
}
```
