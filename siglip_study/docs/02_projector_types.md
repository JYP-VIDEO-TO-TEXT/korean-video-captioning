# Projector 아키텍처 비교

> **학습 목표**: 4가지 Projector의 구조, 수식, 특성 이해

---

## 1. Projector의 역할

Vision Encoder의 출력을 LLM이 이해할 수 있는 형태로 변환합니다.

```
Vision Encoder Output          Projector           LLM Input
────────────────────    ──────────────────    ────────────────
(4608, 1024)        →   (?, 4096)          →   LLM 처리
                        
8 frames × 576 patches    차원 변환 +          LLM 임베딩 공간
× 1024 dim               (선택적) 토큰 압축
```

### 핵심 질문
- 단순 선형 변환으로 충분한가?
- 비선형 변환이 더 좋은가?
- 토큰 수를 줄이면 정보가 손실되는가?

---

## 2. Projector 유형 비교 요약

| Projector | 파라미터 | 출력 토큰 | 특징 |
|-----------|---------|----------|------|
| Linear | 4.2M | 4608 | 가장 단순 |
| MLP-2L | 33.6M | 4608 | 비선형 변환 |
| C-Abstractor | 67.4M | **64** | 토큰 압축 |
| Perceiver | 134.5M | **64** | 최고 표현력 |

---

## 3. Linear Projector

### 3.1 구조

가장 단순한 형태로, 단일 선형 변환만 수행합니다.

```
Input (4608, 1024) ──▶ Linear ──▶ Output (4608, 4096)
```

### 3.2 수식

$$
\mathbf{H}_{out} = \mathbf{W} \cdot \mathbf{H}_{in} + \mathbf{b}
$$

- $\mathbf{H}_{in} \in \mathbb{R}^{N \times 1024}$: Vision 특징
- $\mathbf{W} \in \mathbb{R}^{1024 \times 4096}$: 학습 가능한 가중치
- $\mathbf{b} \in \mathbb{R}^{4096}$: 편향
- $\mathbf{H}_{out} \in \mathbb{R}^{N \times 4096}$: LLM 입력

### 3.3 파라미터 계산

$$
\text{Parameters} = 1024 \times 4096 + 4096 = 4,198,400 \approx 4.2\text{M}
$$

### 3.4 코드 구현

```python
class LinearProjector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)
    
    def forward(self, x):
        # x: (N, 1024) → (N, 4096)
        return self.proj(x)
```

### 3.5 특징

| 장점 | 단점 |
|------|------|
| 구현 간단 | 표현력 제한 |
| 파라미터 적음 | 비선형 관계 학습 불가 |
| 빠른 추론 | 토큰 수 유지 (긴 시퀀스) |

---

## 4. MLP-2L Projector (LLaVA-1.5 스타일)

### 4.1 구조

2층 MLP로 비선형 변환을 추가합니다.

```
Input (4608, 1024) ──▶ Linear ──▶ GELU ──▶ Linear ──▶ Output (4608, 4096)
                      (1024→4096)          (4096→4096)
```

### 4.2 수식

$$
\mathbf{H}_{hidden} = \text{GELU}(\mathbf{W}_1 \cdot \mathbf{H}_{in} + \mathbf{b}_1)
$$

$$
\mathbf{H}_{out} = \mathbf{W}_2 \cdot \mathbf{H}_{hidden} + \mathbf{b}_2
$$

#### GELU (Gaussian Error Linear Unit)

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

근사식:
$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

### 4.3 파라미터 계산

$$
\text{Parameters} = (1024 \times 4096 + 4096) + (4096 \times 4096 + 4096)
$$
$$
= 4,198,400 + 16,781,312 + 4,096 + 4,096 = 33,591,808 \approx 33.6\text{M}
$$

### 4.4 코드 구현

```python
class MLPProjector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)
    
    def forward(self, x):
        # x: (N, 1024) → (N, 4096)
        x = self.fc1(x)      # (N, 4096)
        x = self.act(x)      # 비선형 활성화
        x = self.fc2(x)      # (N, 4096)
        return x
```

### 4.5 특징

| 장점 | 단점 |
|------|------|
| 비선형 관계 학습 | 파라미터 증가 |
| LLaVA-1.5에서 검증됨 | 토큰 수 유지 |
| 성능 향상 | 추론 시간 소폭 증가 |

### 4.6 왜 GELU인가?

```
ReLU vs GELU 비교:

ReLU: max(0, x)           GELU: x · Φ(x)
      │                         │
      │    /                    │      /
      │   /                     │    /
──────┼──/──────          ──────┼──/──────
      │ /                       │/
      │/                        /
      /                        /│
                              / │
                                │

ReLU: 0에서 불연속          GELU: 부드러운 곡선
      음수 완전 제거              음수 일부 유지
```

Transformer 계열 모델에서 GELU가 더 좋은 성능을 보입니다.

---

## 5. C-Abstractor (Cross-Attention Abstractor)

### 5.1 핵심 아이디어

**Learnable Queries**를 사용하여 vision 토큰을 압축합니다.

```
Vision Tokens (4608, 1024)
        │
        ▼
┌───────────────────────────┐
│  Learnable Queries (64)   │
│          │                │
│          ▼                │
│    Cross-Attention        │ ← Vision이 Key, Value
│          │                │    Queries가 Query
│          ▼                │
│       FFN + Norm          │
└───────────────────────────┘
        │
        ▼
Output (64, 4096)  ← 토큰 수 4608 → 64로 압축!
```

### 5.2 수식

#### Step 1: Input Projection

$$
\mathbf{K} = \mathbf{V} = \mathbf{W}_{in} \cdot \mathbf{H}_{vision}
$$

- $\mathbf{H}_{vision} \in \mathbb{R}^{4608 \times 1024}$
- $\mathbf{W}_{in} \in \mathbb{R}^{1024 \times 4096}$

#### Step 2: Cross-Attention

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

- $\mathbf{Q} \in \mathbb{R}^{64 \times 4096}$: Learnable Queries
- $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{4608 \times 4096}$: Vision 특징

```
Query (64, 4096)     Key (4608, 4096)ᵀ      Value (4608, 4096)
      │                    │                      │
      └────────┬───────────┘                      │
               ▼                                  │
         QKᵀ (64, 4608)                          │
               │                                  │
               ▼                                  │
      softmax(QKᵀ/√d) (64, 4608)                 │
               │                                  │
               └────────────────┬─────────────────┘
                                ▼
                    Output (64, 4096)
```

#### Step 3: FFN + Residual

$$
\mathbf{H}' = \text{LayerNorm}(\mathbf{Q} + \text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}))
$$

$$
\mathbf{H}_{out} = \text{LayerNorm}(\mathbf{H}' + \text{FFN}(\mathbf{H}'))
$$

### 5.3 파라미터 계산

| 컴포넌트 | 파라미터 |
|----------|---------|
| Learnable Queries | $64 \times 4096 = 262,144$ |
| Input Projection | $1024 \times 4096 = 4,194,304$ |
| Cross-Attention | $3 \times 4096 \times 4096 = 50,331,648$ |
| FFN | $4096 \times 4 \times 4096 \times 2 \approx 134M$ (과대 추정) |
| LayerNorm | $2 \times 4096 \times 2 = 16,384$ |
| **Total** | **~67.4M** |

### 5.4 코드 구현

```python
class CAbstractor(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096, 
                 num_queries=64, num_heads=8, dropout=0.1):
        super().__init__()
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, llm_dim) * 0.02)
        
        # Input projection
        self.input_proj = nn.Linear(vision_dim, llm_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            llm_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim * 4, llm_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # x: (N, 1024) → (64, 4096)
        x = self.input_proj(x).unsqueeze(0)  # (1, N, 4096)
        queries = self.queries.unsqueeze(0)   # (1, 64, 4096)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(queries, x, x)
        queries = self.norm1(queries + attn_out)
        
        # FFN
        out = self.norm2(queries + self.ffn(queries))
        return out.squeeze(0)  # (64, 4096)
```

### 5.5 Attention 시각화 해석

Cross-Attention weight를 시각화하면 각 Query가 어떤 vision 패치에 집중하는지 알 수 있습니다.

```
Query 1: [0.1, 0.3, 0.05, ..., 0.02]  ← 4608개 패치에 대한 attention
         Frame1    Frame2    ...        Frame8
         
         높은 값 = 해당 패치 정보를 많이 가져옴
```

---

## 6. Perceiver Resampler (Flamingo 스타일)

### 6.1 핵심 아이디어

C-Abstractor에 **Self-Attention**을 추가하여 Query 간 상호작용을 허용합니다.

```
┌─────────────────────────────────────────┐
│           Perceiver Layer (×L)           │
├─────────────────────────────────────────┤
│                                          │
│  Queries ──▶ Self-Attn ──▶ Q'           │
│                                          │
│  Q' + Vision ──▶ Cross-Attn ──▶ Q''     │
│                                          │
│  Q'' ──▶ FFN ──▶ Output Queries         │
│                                          │
└─────────────────────────────────────────┘
```

### 6.2 수식

#### Layer l의 처리

$$
\mathbf{Q}^{(l)}_{self} = \text{LayerNorm}(\mathbf{Q}^{(l-1)} + \text{SelfAttn}(\mathbf{Q}^{(l-1)}))
$$

$$
\mathbf{Q}^{(l)}_{cross} = \text{LayerNorm}(\mathbf{Q}^{(l)}_{self} + \text{CrossAttn}(\mathbf{Q}^{(l)}_{self}, \mathbf{X}_{vision}))
$$

$$
\mathbf{Q}^{(l)} = \text{LayerNorm}(\mathbf{Q}^{(l)}_{cross} + \text{FFN}(\mathbf{Q}^{(l)}_{cross}))
$$

#### 전체 (L layers)

$$
\mathbf{H}_{out} = f_L(f_{L-1}(...f_1(\mathbf{Q}^{(0)}, \mathbf{X}_{vision})...))
$$

### 6.3 Self-Attention의 역할

```
Query들 간의 관계:

Query 1: "객체 정보 담당"
Query 2: "배경 정보 담당"
Query 3: "동작 정보 담당"
    ...

Self-Attention으로 Query끼리 정보 교환
→ "객체가 배경 위에서 움직임" 같은 관계 학습
```

### 6.4 코드 구현

```python
class PerceiverLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, 
                                                dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, 
                                                 dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, queries, context):
        # Self-attention
        q = self.norm1(queries + self.self_attn(queries, queries, queries)[0])
        # Cross-attention
        q = self.norm2(q + self.cross_attn(q, context, context)[0])
        # FFN
        return self.norm3(q + self.ffn(q))


class PerceiverResampler(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096, 
                 num_queries=64, num_heads=8, num_layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, llm_dim) * 0.02)
        self.input_proj = nn.Linear(vision_dim, llm_dim)
        self.layers = nn.ModuleList([
            PerceiverLayer(llm_dim, num_heads) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(llm_dim)
    
    def forward(self, x):
        context = self.input_proj(x).unsqueeze(0)
        queries = self.queries.unsqueeze(0)
        
        for layer in self.layers:
            queries = layer(queries, context)
        
        return self.output_norm(queries).squeeze(0)
```

### 6.5 파라미터 계산 (num_layers=2)

| 컴포넌트 | 파라미터 |
|----------|---------|
| Learnable Queries | $64 \times 4096 = 262,144$ |
| Input Projection | $1024 \times 4096 = 4,194,304$ |
| Per Layer | ~$50M$ (Self-Attn + Cross-Attn + FFN) |
| **Total (2 layers)** | **~134.5M** |

---

## 7. 비교 분석

### 7.1 파라미터 vs 표현력 트레이드오프

```
파라미터 수 (M)
     │
 134 │                              ● Perceiver
     │
  67 │                    ● C-Abstractor
     │
  33 │          ● MLP-2L
     │
   4 │ ● Linear
     │
     └──────────────────────────────────────▶ 표현력
           낮음                       높음
```

### 7.2 시퀀스 길이 비교

```
LLM 입력 시퀀스:

Linear/MLP:     [Vision×4608] + [Text×N]  = 4608 + N 토큰
C-Abstractor:   [Vision×64]   + [Text×N]  = 64 + N 토큰
Perceiver:      [Vision×64]   + [Text×N]  = 64 + N 토큰
                      ↓
              약 72배 압축!
```

**압축의 의미**:
- LLM 추론 속도 향상
- 메모리 절약
- 긴 비디오 처리 가능

### 7.3 예상 성능

| Projector | SigLIP Score (예상) | 추론 속도 | 메모리 |
|-----------|-------------------|----------|--------|
| Linear | 0.30-0.40 | 빠름 | 높음 (긴 시퀀스) |
| MLP-2L | 0.40-0.50 | 빠름 | 높음 |
| C-Abstractor | 0.35-0.45 | 중간 | **낮음** |
| Perceiver | 0.45-0.55 | 느림 | **낮음** |

---

## 8. 어떤 Projector를 선택해야 하는가?

### 시나리오별 권장

| 상황 | 권장 Projector | 이유 |
|------|---------------|------|
| 빠른 프로토타이핑 | Linear | 구현 간단, 빠른 검증 |
| 성능 중시 | MLP-2L, Perceiver | 비선형 변환으로 성능 향상 |
| 긴 비디오 | C-Abstractor, Perceiver | 토큰 압축으로 메모리 절약 |
| 실시간 추론 | Linear, MLP-2L | 압축 없이 빠른 처리 |

---

## 참고 자료

- [LLaVA-1.5](https://arxiv.org/abs/2310.03744) - MLP Projector 도입
- [Honeybee](https://arxiv.org/abs/2312.06742) - C-Abstractor
- [Flamingo](https://arxiv.org/abs/2204.14198) - Perceiver Resampler
- [Perceiver](https://arxiv.org/abs/2103.03206) - 원본 Perceiver 논문
