# Custom VLM 실험 계획서

> **프로젝트**: Korean Video Captioning with Custom VLM  
> **목표**: CLIP Vision Encoder + Qwen3-8B LLM 조합으로 한국어 비디오 캡셔닝 성능 최적화  
> **환경**: Google Colab (A100 80GB)

---

## 1. 문제 정의와 목표

### 1.1 배경

기존 LLaVA-NeXT-Video-7B 모델은 영어 중심으로 학습되어 한국어 캡셔닝 성능이 제한적입니다.
- **Baseline METEOR**: 0.3052
- **목표 METEOR**: 0.40+ (+30% 향상)

### 1.2 핵심 가설

| 가설 | 검증 방법 |
|------|----------|
| Vision Encoder는 유지하고 **LLM만 한국어 강한 모델로 교체**하면 성능 개선 가능 | Qwen3-8B로 LLM 교체 후 성능 측정 |
| **Projector 구조**에 따라 vision-language alignment 품질이 달라짐 | 4종 Projector 비교 실험 |
| **PEFT 방법**(LoRA vs DoRA)에 따라 fine-tuning 효율이 다름 | 동일 Projector에서 LoRA/DoRA 비교 |

### 1.3 실험 목표

1. **Primary**: 한국어 캡셔닝 성능 최대화 (METEOR 기준)
2. **Secondary**: 최적의 Projector-PEFT 조합 도출
3. **Tertiary**: 학습 효율성 및 추론 비용 분석

---

## 2. 모델 아키텍처

### 2.1 구성 요소

```
┌─────────────────────────────────────────────────────────────┐
│                     Custom VLM Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Vision    │    │  Projector  │    │      LLM        │ │
│  │   Encoder   │───▶│  (학습)     │───▶│   (LoRA/DoRA)   │ │
│  │  (Frozen)   │    │             │    │                 │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│                                                             │
│  CLIP-ViT-L/14      4종 비교          Qwen3-8B (4-bit)     │
│  336px              - Linear          - LoRA (r=16)        │
│  1024-dim output    - MLP-2L          - DoRA (r=16)        │
│                     - C-Abstractor                         │
│                     - Perceiver                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 모듈별 설정

| 모듈 | 모델 | 상태 | 파라미터 |
|------|------|------|----------|
| Vision Encoder | `openai/clip-vit-large-patch14-336` | **Frozen** | 304M |
| LLM | `Qwen/Qwen3-8B` | 4-bit 양자화 + PEFT | 8B → ~5GB |
| Projector | 4종 비교 | **Full Training** | 4M~20M |

### 2.3 입출력 스펙

```python
# Vision Encoder 출력
input_frames: [batch, num_frames, 3, 336, 336]  # 8 frames
vision_output: [batch, num_frames, 577, 1024]   # CLS + 576 patches
patch_features: [batch, num_frames * 576, 1024] # 4608 tokens

# Projector 출력
# Linear/MLP: [batch, 4608, 4096] → 토큰 수 유지
# C-Abstractor/Perceiver: [batch, 64, 4096] → 토큰 압축

# LLM 입력
combined_embeds: [batch, vision_tokens + text_tokens, 4096]
```

---

## 3. Projector 비교 실험 (4종)

### 3.1 실험 대상

| ID | Projector | 구조 | 출력 토큰 | 파라미터 | 특징 |
|----|-----------|------|-----------|----------|------|
| P1 | **Linear** | 단일 선형 변환 | 4608 | 4.2M | Baseline, 빠른 학습 |
| P2 | **MLP-2L** | 2-layer MLP + GELU | 4608 | 20.9M | LLaVA-1.5 스타일 |
| P3 | **C-Abstractor** | Cross-Attention 압축 | 64 | 16.8M | 토큰 효율적, 빠른 추론 |
| P4 | **Perceiver** | Self+Cross Attention | 64 | 25.2M | 풍부한 표현력 |

### 3.2 공정 비교 조건

모든 실험에서 아래 조건 동일:

| 항목 | 설정 |
|------|------|
| Vision Encoder | CLIP-ViT-L/14-336 (frozen) |
| LLM | Qwen3-8B (4-bit, frozen base) |
| 입력 프레임 수 | 8 frames |
| 학습 데이터 | AI Hub 한국어 비디오 캡셔닝 (500 samples) |
| Batch Size | 4 (effective: 8) |
| Stage 1 Epochs | 2 (Projector warm-up) |
| Stage 2 Epochs | 5 (Joint fine-tuning) |
| Seed | 42 |

### 3.3 Projector 구조 상세

#### P1: Linear Projector
```python
class LinearProjector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)
    
    def forward(self, x):
        return self.proj(x)  # [B, 4608, 1024] → [B, 4608, 4096]
```

#### P2: MLP-2L Projector
```python
class MLPProjector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

#### P3: C-Abstractor (Cross-Attention)
```python
class CAbstractor(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096, num_queries=64):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, llm_dim))
        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads=8)
        self.input_proj = nn.Linear(vision_dim, llm_dim)
    
    def forward(self, x):
        # x: [B, 4608, 1024] → [B, 64, 4096]
        x = self.input_proj(x)
        queries = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.cross_attn(queries, x, x)
        return out
```

#### P4: Perceiver Resampler
```python
class PerceiverResampler(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096, num_queries=64, num_layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, llm_dim))
        self.input_proj = nn.Linear(vision_dim, llm_dim)
        self.layers = nn.ModuleList([
            PerceiverLayer(llm_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x = self.input_proj(x)
        queries = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        for layer in self.layers:
            queries = layer(queries, x)
        return queries
```

---

## 4. PEFT 비교 실험 (LoRA vs DoRA)

### 4.1 실험 설계

각 Projector에 대해 LoRA와 DoRA를 모두 테스트:

| 실험 ID | Projector | PEFT | 담당 |
|---------|-----------|------|------|
| E1 | Linear | LoRA | Person A |
| E2 | Linear | DoRA | Person A |
| E3 | MLP-2L | LoRA | Person A |
| E4 | MLP-2L | DoRA | Person B |
| E5 | C-Abstractor | LoRA | Person B |
| E6 | C-Abstractor | DoRA | Person B |
| E7 | Perceiver | LoRA | Person C |
| E8 | Perceiver | DoRA | Person C |

### 4.2 PEFT 공통 설정

```yaml
peft:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### 4.3 LoRA vs DoRA 비교 포인트

| 비교 항목 | LoRA | DoRA | 측정 방법 |
|----------|------|------|----------|
| **한국어 성능** | Baseline | 예상: +2~5% | METEOR, BERTScore |
| **학습 안정성** | 안정적 | 다소 불안정 가능 | Loss 곡선 분석 |
| **메모리 사용** | 기준 | +10~15% | nvidia-smi |
| **학습 속도** | 기준 | -5~10% | step/sec |
| **추론 병합** | 가능 | 가능 | merge_and_unload() |

### 4.4 DoRA 원리

```python
# LoRA: W' = W + BA
# DoRA: W' = m * (W + BA) / ||W + BA||
#       m = magnitude (learnable), direction = normalized

class DoRAConfig:
    use_dora: True  # LoraConfig에서 설정
```

---

## 5. 2-Stage 학습 전략

### 5.1 Stage 1: Projector Warm-up

**목표**: Vision-Language alignment 초기화

| 항목 | 설정 |
|------|------|
| 학습 대상 | Projector only |
| LLM | Frozen (PEFT 비활성) |
| Learning Rate | 1e-3 |
| Epochs | 2 |
| 목적 | Vision 토큰을 LLM 공간에 매핑 |

```python
# Stage 1: Projector만 학습
for param in model.llm.parameters():
    param.requires_grad = False
for param in model.projector.parameters():
    param.requires_grad = True
```

### 5.2 Stage 2: Joint Fine-tuning

**목표**: Projector + LLM(PEFT) 공동 최적화

| 항목 | 설정 |
|------|------|
| 학습 대상 | Projector + LLM (LoRA/DoRA) |
| Learning Rate | 5e-5 |
| Epochs | 5 |
| 목적 | 한국어 캡셔닝 품질 최적화 |

```python
# Stage 2: Projector + LoRA/DoRA 학습
for param in model.projector.parameters():
    param.requires_grad = True
# PEFT adapter는 자동으로 requires_grad=True
```

---

## 6. 평가 계획

### 6.1 평가 지표

| 지표 | 역할 | 기준 |
|------|------|------|
| **METEOR** | Primary | Baseline 0.3052, 목표 0.40+ |
| **CIDEr** | Secondary | 의미적 유사도 |
| **BERTScore (F1)** | Secondary | 한국어 의미 평가 |

### 6.2 평가 시점

- **매 Epoch**: Validation set 전체 평가
- **Best Model**: METEOR 기준 저장
- **최종 비교**: 8개 실험 결과 통합

### 6.3 결과 기록

```
custom_vlm_experiments/
├── E1_linear_lora/
│   ├── checkpoints/
│   │   ├── stage1_projector.pt
│   │   ├── best_model.pt
│   │   └── final_model.pt
│   ├── logs/
│   │   ├── training_log.csv
│   │   ├── training_curves.png
│   │   └── final_metrics.json
│   └── samples/
│       └── predictions.json
├── E2_linear_dora/
│   └── ...
└── summary/
    ├── all_results.csv
    └── comparison_plots.png
```

---

## 7. 실험 분담

### 7.1 담당 배분

| 담당자 | 실험 | Projector | PEFT | 우선순위 |
|--------|------|-----------|------|----------|
| **Person A** | E1, E2, E3 | Linear, MLP-2L | LoRA, DoRA | 높음 (Baseline) |
| **Person B** | E4, E5, E6 | MLP-2L, C-Abstractor | DoRA, LoRA | 중간 |
| **Person C** | E7, E8 | Perceiver | LoRA, DoRA | 낮음 (Advanced) |

### 7.2 실행 파일

```
korean_video_captioning/notebooks/
├── 06a_person_a_baseline.ipynb      # E1, E2, E3
├── 06b_person_b_intermediate.ipynb  # E4, E5, E6
└── 06c_person_c_advanced.ipynb      # E7, E8
```

### 7.3 실행 순서 (권장)

```
1. [Person A] E1_linear_lora     ← Baseline 확립
2. [Person A] E2_linear_dora     ← LoRA vs DoRA 초기 비교
3. [Person B] E4_mlp2l_dora      ← MLP + DoRA
4. [Person A] E3_mlp2l_lora      ← MLP + LoRA (비교용)
5. [Person B] E5_cabstractor_lora
6. [Person B] E6_cabstractor_dora
7. [Person C] E7_perceiver_lora
8. [Person C] E8_perceiver_dora
```

---

## 8. 예상 결과 및 분석 계획

### 8.1 예상 결과 (가설 기반)

| Projector | LoRA METEOR | DoRA METEOR | 예상 특징 |
|-----------|-------------|-------------|-----------|
| Linear | 0.32~0.35 | 0.33~0.36 | 빠른 학습, 기본 성능 |
| MLP-2L | 0.35~0.38 | 0.36~0.40 | 균형 잡힌 성능 |
| C-Abstractor | 0.33~0.37 | 0.34~0.38 | 빠른 추론, 압축 손실 |
| Perceiver | 0.36~0.40 | 0.38~0.42 | 최고 성능, 느린 학습 |

### 8.2 분석 항목

1. **Projector 영향도**
   - 토큰 압축 (4608 → 64) vs 유지의 성능 차이
   - 파라미터 수와 성능의 상관관계

2. **PEFT 영향도**
   - LoRA vs DoRA 평균 성능 차이
   - 학습 안정성 (loss 변동성)
   - 메모리/속도 트레이드오프

3. **최적 조합 도출**
   - 성능 우선: 예상 Perceiver + DoRA
   - 효율 우선: 예상 C-Abstractor + LoRA

---

## 9. 경량화 및 추론 최적화 (Optional)

### 9.1 현재 적용된 최적화

| 기법 | 적용 대상 | 효과 |
|------|----------|------|
| 4-bit 양자화 (NF4) | LLM | 메모리 ~75% 감소 |
| Gradient Checkpointing | LLM | 메모리 ~30% 감소 |
| Mixed Precision (bfloat16) | 전체 | 속도 향상 |

### 9.2 추가 최적화 후보

| 기법 | 예상 효과 | 난이도 |
|------|----------|--------|
| FlashAttention-2 | 추론 속도 +30% | 중 |
| KV Cache 최적화 | 긴 시퀀스 처리 개선 | 중 |
| Token Pruning | C-Abstractor와 조합 | 상 |
| 8-bit 양자화 | 품질 vs 속도 트레이드오프 | 하 |

### 9.3 추론 속도 측정 계획

```python
# 추론 시간 측정
import time

def measure_inference_time(model, test_samples, num_runs=10):
    times = []
    for _ in range(num_runs):
        start = time.time()
        for sample in test_samples:
            model.generate(sample["frames"], sample["prompt"])
        times.append(time.time() - start)
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "samples_per_sec": len(test_samples) / np.mean(times)
    }
```

---

## 10. 산출물 체크리스트

### 10.1 필수 산출물

- [ ] **Projector 4종 비교 결과표** (METEOR, CIDEr, BERTScore)
- [ ] **LoRA vs DoRA 비교 결론** (동일 Projector 기준)
- [ ] **최적 조합 추천** (성능/효율 기준)
- [ ] **학습된 모델 체크포인트** (8개 실험)

### 10.2 분석 산출물

- [ ] 학습 곡선 비교 플롯
- [ ] 메모리 사용량 비교
- [ ] 추론 속도 비교 (samples/sec)
- [ ] 실패 케이스 분석

### 10.3 문서 산출물

- [ ] 실험 결과 요약 보고서
- [ ] 최적 설정 가이드
- [ ] 재현 가능한 실행 명령어

---

## 11. 일정 (예상)

| 단계 | 작업 | 예상 시간 |
|------|------|----------|
| 1 | 환경 설정 및 데이터 로드 | 30분 |
| 2 | E1~E2 실험 (Person A) | 3-4시간 |
| 3 | E3~E4 실험 (Person A, B) | 3-4시간 |
| 4 | E5~E6 실험 (Person B) | 3-4시간 |
| 5 | E7~E8 실험 (Person C) | 3-4시간 |
| 6 | 결과 통합 및 분석 | 1-2시간 |
| **총계** | | **15-20시간** |

---

## 12. 참고 자료

- [LLaVA-1.5 Paper](https://arxiv.org/abs/2310.03744) - MLP Projector 설계
- [Honeybee Paper](https://arxiv.org/abs/2312.06742) - C-Abstractor 설계
- [Flamingo Paper](https://arxiv.org/abs/2204.14198) - Perceiver Resampler
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [DoRA Paper](https://arxiv.org/abs/2402.09353) - Weight-Decomposed LRA
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/) - LLM 스펙

---

*Last Updated: 2026-01-19*
