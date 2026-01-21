# SigLIP Score 평가 방법론

> **학습 목표**: CLIP과 SigLIP의 차이, SigLIP Score 계산 원리 이해

---

## 1. 왜 SigLIP인가?

### 1.1 기존 평가 지표의 문제

| 지표 | 문제점 |
|------|--------|
| METEOR | 번역 품질 측정 → LLM 능력과 혼재 |
| CIDEr | Reference 필요 → 정답 의존적 |
| BERTScore | 텍스트만 비교 → Vision 정렬 측정 불가 |
| CLIP Score | Softmax 기반 → 배치 크기 의존 |

### 1.2 SigLIP의 장점

- **Vision-Language 정렬 직접 측정**: 이미지와 텍스트가 얼마나 잘 맞는지
- **Reference 불필요**: 정답 캡션 없이도 평가 가능
- **배치 크기 독립**: Sigmoid 함수 사용
- **다국어 지원**: 한국어 포함 100+ 언어

---

## 2. CLIP vs SigLIP 비교

### 2.1 CLIP (Contrastive Language-Image Pre-training)

```
배치 내 이미지-텍스트 쌍:

        Text1   Text2   Text3   Text4
Image1  [ ✓ ]   [ ✗ ]   [ ✗ ]   [ ✗ ]
Image2  [ ✗ ]   [ ✓ ]   [ ✗ ]   [ ✗ ]
Image3  [ ✗ ]   [ ✗ ]   [ ✓ ]   [ ✗ ]
Image4  [ ✗ ]   [ ✗ ]   [ ✗ ]   [ ✓ ]

✓ = Positive pair (같은 쌍)
✗ = Negative pair (다른 쌍)
```

#### CLIP Loss (InfoNCE)

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ij} / \tau)}
$$

- $s_{ij} = \mathbf{x}_i^T \mathbf{y}_j$: 이미지 $i$와 텍스트 $j$의 유사도
- $\tau$: Temperature 파라미터
- **Softmax**: 배치 내 모든 샘플과 비교

#### 문제점

```
배치 크기 = 4:   softmax([0.9, 0.1, 0.05, 0.02]) → 높은 확률
배치 크기 = 1000: softmax([0.9, 0.1, ..., 0.001]) → 낮은 확률

→ 같은 유사도여도 배치 크기에 따라 값이 달라짐!
```

### 2.2 SigLIP (Sigmoid Loss for Language-Image Pre-training)

#### 핵심 차이: Sigmoid 사용

```
각 쌍을 독립적으로 평가:

(Image1, Text1): σ(similarity) = 0.95  ← Positive
(Image1, Text2): σ(similarity) = 0.02  ← Negative
(Image1, Text3): σ(similarity) = 0.01  ← Negative
...

→ 배치 크기와 무관하게 일관된 점수!
```

#### SigLIP Loss

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N}\sum_{i,j} \log \sigma(z_{ij} \cdot (s_{ij} - b))
$$

- $z_{ij} = \begin{cases} +1 & \text{if } i = j \text{ (positive pair)} \\ -1 & \text{if } i \neq j \text{ (negative pair)} \end{cases}$
- $s_{ij} = \mathbf{x}_i^T \mathbf{y}_j$: 코사인 유사도
- $b$: Learnable bias
- $\sigma(x) = \frac{1}{1 + e^{-x}}$: Sigmoid 함수

#### 수식 전개

**Positive pair** ($i = j$):
$$
\mathcal{L}_{+} = -\log \sigma(s_{ii} - b) = -\log \frac{1}{1 + e^{-(s_{ii} - b)}}
$$

**Negative pair** ($i \neq j$):
$$
\mathcal{L}_{-} = -\log \sigma(-(s_{ij} - b)) = -\log \frac{1}{1 + e^{(s_{ij} - b)}}
$$

---

## 3. Sigmoid vs Softmax 직관적 이해

```
Softmax: "이 중에서 어떤 게 정답이야?"
         → 상대적 비교 (합 = 1)
         
Sigmoid: "이 쌍은 맞아? 틀려?"
         → 절대적 판단 (각각 0~1)
```

### 예시

| 상황 | Softmax | Sigmoid |
|------|---------|---------|
| 배치에 유사한 negative 많음 | 점수 ↓ | 점수 유지 |
| 배치에 쉬운 negative만 | 점수 ↑ | 점수 유지 |
| 배치 크기 변경 | 점수 변동 | **점수 일관** |

---

## 4. SigLIP Score 계산 방법

### 4.1 평가 시 사용하는 공식

$$
\text{SigLIP Score} = \sigma(s) = \frac{1}{1 + e^{-s}}
$$

- $s = \mathbf{x}_{image}^T \mathbf{y}_{text}$: 정규화된 임베딩의 내적

### 4.2 코드 구현

```python
from transformers import AutoModel, AutoProcessor
import torch

class SigLIPEvaluator:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device="cuda"):
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def compute_score(self, frames: list, caption: str) -> float:
        """
        프레임들과 캡션의 SigLIP Score 계산
        
        Args:
            frames: PIL Image 리스트 (비디오 프레임들)
            caption: 생성된 캡션
        
        Returns:
            score: 0~1 사이 값 (높을수록 좋음)
        """
        # 전처리
        inputs = self.processor(
            text=[caption], 
            images=frames, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # logits_per_image: (num_frames, 1)
        # 각 프레임과 캡션의 유사도
        logits = outputs.logits_per_image
        
        # Sigmoid 적용 후 평균
        score = torch.sigmoid(logits).mean().item()
        
        return score
```

### 4.3 다중 프레임 처리

```
Frame 1 ──▶ SigLIP Image Encoder ──▶ h1
Frame 2 ──▶ SigLIP Image Encoder ──▶ h2
   ...
Frame 8 ──▶ SigLIP Image Encoder ──▶ h8

Caption ──▶ SigLIP Text Encoder ──▶ t

Scores:
  s1 = σ(h1 · t)
  s2 = σ(h2 · t)
  ...
  s8 = σ(h8 · t)

Final Score = mean(s1, s2, ..., s8)
```

---

## 5. SigLIP 모델 스펙

### 5.1 우리가 사용하는 모델

**google/siglip-so400m-patch14-384**

| 항목 | 값 |
|------|-----|
| Vision Encoder | ViT-SO400M |
| Image Size | 384 × 384 |
| Patch Size | 14 × 14 |
| Hidden Dim | 1152 |
| 학습 데이터 | WebLI (10B image-text pairs) |
| 언어 | 100+ languages (한국어 포함) |

### 5.2 모델 로드 예시

```python
from transformers import AutoModel, AutoProcessor

# 모델 로드 (약 1.6GB 다운로드)
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# 평가 시에만 사용 (학습 안 함)
model.eval()
for param in model.parameters():
    param.requires_grad = False
```

---

## 6. SigLIP Score 해석

### 6.1 점수 범위

```
0.0 ─────────────────────────────────────── 1.0
 │           │           │           │
무관         약한 관련    좋은 매칭   완벽한 매칭
```

### 6.2 실제 예시

| 이미지 | 캡션 | 예상 점수 |
|--------|------|----------|
| 강아지 사진 | "귀여운 강아지가 공원에서 뛰어놀고 있다" | 0.7-0.9 |
| 강아지 사진 | "고양이가 소파에 앉아있다" | 0.2-0.4 |
| 강아지 사진 | "주식 시장 분석 보고서" | 0.0-0.1 |

### 6.3 Projector 비교 시 기준

| 점수 범위 | 의미 |
|----------|------|
| 0.5+ | 기본적인 Vision-Language 정렬 |
| 0.6+ | 좋은 성능 |
| 0.7+ | 우수한 성능 |
| 차이 0.05+ | 유의미한 차이 |

---

## 7. 왜 METEOR 대신 SigLIP인가?

### 7.1 Projector 평가에 적합한 이유

```
우리의 목표: "어떤 Projector가 Vision 정보를 잘 전달하는가?"

METEOR로 평가:
  Vision → Projector → LLM → Caption
                        ↑
                     LLM 능력이 
                     결과에 영향
                     
SigLIP로 평가:
  Vision → Projector → LLM → Caption ←── SigLIP ──→ Vision
                                              ↑
                                        Vision과의 직접 비교
                                        (LLM 능력 분리)
```

### 7.2 AI-Hub 데이터셋 특수성

AI-Hub 베이스라인:
```
Video → 영어 캡션 생성 → GPT로 한국어 번역
```

우리 방식:
```
Video → 직접 한국어 캡션 생성
```

METEOR로 비교하면:
- GPT 번역 스타일 vs 우리 모델 스타일 비교
- Projector 효과가 아닌 스타일 차이 측정

SigLIP로 비교하면:
- 영상 내용과의 일치도 직접 측정
- 스타일과 무관하게 정보 전달력 평가

---

## 8. 한계점 및 주의사항

### 8.1 SigLIP의 한계

| 한계 | 설명 |
|------|------|
| 세부 정확도 | "빨간 공" vs "파란 공" 구분 어려움 |
| 시간 정보 | 비디오의 시간적 순서 고려 안 함 |
| 복잡한 관계 | "A가 B를 들고 있다" 같은 관계 약함 |

### 8.2 보완 방법

1. **정성 평가 병행**: 생성된 캡션 직접 확인
2. **실패 케이스 분석**: SigLIP 낮은 샘플 분석
3. **다중 지표**: BERTScore (ko-sbert) 참고용 추가

---

## 9. 실험에서의 활용

### 9.1 평가 프로세스

```python
def evaluate_model(model, val_loader, siglip_evaluator, prompt, max_samples=50):
    results = []
    
    for batch in val_loader:
        frames = batch["pil_frames"]
        
        # 1. 모델로 캡션 생성
        generated_caption = model.generate(batch["pixel_values"], prompt)
        
        # 2. SigLIP Score 계산
        score = siglip_evaluator.compute_score(frames, generated_caption)
        
        results.append({
            "caption": generated_caption,
            "siglip_score": score
        })
    
    # 3. 평균 점수 반환
    return {
        "siglip_score": np.mean([r["siglip_score"] for r in results]),
        "siglip_std": np.std([r["siglip_score"] for r in results])
    }
```

### 9.2 결과 해석 예시

```
실험 결과:
  Linear:      SigLIP = 0.42 ± 0.08
  MLP-2L:      SigLIP = 0.48 ± 0.07
  C-Abstractor: SigLIP = 0.45 ± 0.09
  Perceiver:   SigLIP = 0.52 ± 0.06

해석:
  - Perceiver가 가장 높은 Vision-Language 정렬
  - MLP > Linear: 비선형 변환의 효과 확인
  - Perceiver의 표준편차가 낮음: 안정적인 성능
```

---

## 참고 자료

- [SigLIP Paper](https://arxiv.org/abs/2303.15343) - Sigmoid Loss for Language Image Pre-Training
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models
- [Hugging Face SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
