# Mode Collapse 분석

## Mode Collapse란?

생성 모델이 입력에 관계없이 동일하거나 매우 유사한 출력만 생성하는 현상입니다.

```
정상 학습:
  영상 A → "포스코건설 회색 건물"
  영상 B → "한국학술진흥재단 베이지색 건물"
  영상 C → "2000년대 한옥, 갈색 대문"

Mode Collapse:
  영상 A → "도심의 한 건물과 그 주변 환경"
  영상 B → "도심의 한 건물과 그 주변 환경"  (동일)
  영상 C → "도심의 한 건물과 그 주변 환경"  (동일)
```

---

## C-Abstractor에서 발생한 현상

### 증상

| Epoch | Unique Captions | Diversity | 대표 캡션 |
|-------|-----------------|-----------|----------|
| 1 | 9/97 | 0.09 | "1970년대 후반에 촬영된 것으로 보이며..." |
| 2 | 6/97 | 0.06 | "도심의 한 구역을 담고 있습니다..." |
| 3 | 4/97 | 0.04 | "2017년 이후에 촬영된 것으로..." |
| 4 | **1/97** | **0.01** | "도시의 한 건물을 중심으로..." |
| 5 | 4/97 | 0.04 | "맑고 화창한 낮의 풍경을..." |
| 6 | 6/97 | 0.06 | "맑고 화창한 낮의 광경을..." |

### Diversity 추이 그래프

```
Diversity
1.0 |
0.9 |
0.8 |
0.7 |
0.6 |
0.5 | ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  (정상 기준선)
0.4 |
0.3 |
0.2 |
0.1 | ●   ●   ●   ●   ●   ●
0.0 |─────────────────────────────
    E1  E2  E3  E4  E5  E6
```

**모든 epoch에서 Diversity < 0.5 → Mode Collapse 발생**

---

## SigLIP vs Diversity 역상관 관계

### E5-v3 결과

```
Epoch:      1      2      3      4      5      6
──────────────────────────────────────────────────
SigLIP:   0.24   0.23   0.34   0.13   0.04   0.00
Diversity: 0.09   0.06   0.04   0.01   0.04   0.06
           │      │      │      │
           │      ↓      ↓      ↓
           └──────┴──────┴──────── SigLIP ↑ = Diversity ↓
```

### 왜 이런 현상이 발생하는가?

```
SigLIP 평가 방식:
  score = similarity(image_embedding, text_embedding)

"도심의 건물"이라는 일반적 캡션:
  - 대부분의 건물 이미지와 높은 유사도
  - 구체적인 "포스코건설 회색 건물"보다 "안전한" 선택

결과:
  - 모델이 "평균적인" 캡션으로 수렴
  - SigLIP 점수는 높지만 실제로 Vision 정보 무시
```

### 핵심 통찰

> **높은 SigLIP ≠ 좋은 Projector**
>
> **높은 SigLIP = "캡션이 이미지와 충돌하지 않는다"**
>
> **좋은 Projector = "캡션이 이미지의 세부 정보를 반영한다"**

---

## 시도한 해결책과 결과

### 1. LR 감소 (E5-v2)

**가설**: 높은 LR이 불안정한 학습을 유발

**시도:**
```yaml
stage1_lr: 1e-3 → 1e-4  (1/10)
stage2_lr: 5e-5 → 2e-5  (2/5)
```

**결과:**
- Diversity: 0.11 (11/97 unique)
- Mode Collapse **해결 실패**

---

### 2. Epoch 증가 (E5-v3)

**가설**: 학습 시간 부족으로 수렴하지 못함

**시도:**
```yaml
stage1_epochs: 2 → 5   (2.5배)
stage2_epochs: 3 → 10  (3.3배)
early_stopping_patience: 3
```

**결과:**
- Diversity: 0.06 (6/97 unique)
- Mode Collapse **해결 실패**, 오히려 **악화**
- Early stopping으로 epoch 6에서 조기 종료

---

### 3. 공통 최적화

**시도:**
- LR Scheduler (Warmup + Cosine Decay)
- Gradient Clipping (max_norm=1.0)
- Diversity Monitoring

**결과:**
- Mode Collapse **해결 실패**
- 단, Diversity Monitoring으로 문제 **조기 탐지**는 성공

---

## 근본 원인 분석

### 1. 과도한 파라미터 수

```
C-Abstractor: 206M params
Linear:       4M params
MLP-2L:       8M params

데이터셋: 865 train samples
```

- 206M 파라미터 vs 865 샘플 → **극심한 과적합 위험**
- 모델이 다양한 패턴을 학습하기보다 "평균"에 수렴

### 2. Cross-Attention 구조 문제

```python
class CAbstractor(nn.Module):
    def __init__(self):
        self.queries = nn.Parameter(torch.randn(64, 4096))  # 학습 가능한 쿼리
        self.cross_attn = nn.MultiheadAttention(...)
    
    def forward(self, vision_features):
        # 고정된 쿼리가 Vision 정보를 "요약"
        output = self.cross_attn(self.queries, vision_features, vision_features)
        return output  # (64, 4096)
```

**문제점:**
- 학습 가능한 쿼리가 특정 패턴에 과적합
- 다양한 Vision 정보를 구분하지 못하고 "평균화"

### 3. 토큰 압축으로 인한 정보 손실

```
Vision Encoder 출력: 4608 tokens (8 frames × 576 patches)
C-Abstractor 출력:   64 tokens  (압축)

압축률: 72:1

질문: 72배 압축으로 세부 정보 유지 가능한가?
```

---

## 결론

### C-Abstractor가 실패한 이유

| 원인 | 설명 |
|------|------|
| 과적합 | 206M params vs 865 samples |
| 구조적 한계 | Cross-attention 쿼리 과적합 |
| 정보 손실 | 72:1 토큰 압축 |

### 다음 단계 권장

1. **단순한 Projector 재검토**
   - Linear (4M): 압축 없이 단순 변환
   - MLP-2L (8M): 비선형 변환 추가

2. **C-Abstractor 구조 수정 (선택적)**
   - `num_queries` 축소: 64 → 16
   - Dropout 증가: 0.1 → 0.3
   - 쿼리 초기화 변경

3. **데이터 증강**
   - 더 많은 학습 샘플
   - 프레임 augmentation

---

## 핵심 교훈

> **"더 큰 모델이 항상 더 좋은 것은 아니다"**
>
> 특히 작은 데이터셋에서는 단순한 모델이 일반화에 유리할 수 있다.

> **"평가 지표는 다양한 관점에서 측정해야 한다"**
>
> SigLIP만으로는 Mode Collapse를 탐지할 수 없다.
> Diversity와 함께 평가해야 한다.
