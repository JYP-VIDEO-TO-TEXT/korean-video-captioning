# 다음 단계 계획

## 현재 상태 요약

| 상태 | 내용 |
|------|------|
| **완료** | E5-v2 (C-Abstractor, LR 감소) → Mode Collapse |
| **완료** | E5-v3 (C-Abstractor, Epoch 증가) → Mode Collapse |
| **대기** | E1-v2 (Linear + 최적화) |
| **대기** | E3-v2 (MLP-2L + 최적화) |
| **미실행** | E7 (Perceiver) |

---

## 단기 계획 (즉시 실행)

### 1. E1-v2: Linear Projector 실험

**목적**: 가장 단순한 Projector가 Mode Collapse 없이 학습 가능한지 확인

**노트북**: `01_person_a_v2.ipynb`

**설정:**
```yaml
experiment_name: E1_v2_linear_optimized
projector_type: linear
projector_params: ~4,200,000

stage1_lr: 1e-3
stage2_lr: 5e-5
stage1_epochs: 2
stage2_epochs: 3

# 최적화 적용
warmup_ratio: 0.1
max_grad_norm: 1.0
vision_caching: true
diversity_monitoring: true
```

**예상 결과:**
- Diversity > 0.5 (Mode Collapse 없음)
- SigLIP: 0.15-0.25 (C-Abstractor보다 낮을 수 있음)
- METEOR: 비슷하거나 더 높음

---

### 2. E3-v2: MLP-2L Projector 실험

**목적**: 비선형 변환이 추가된 단순 Projector 성능 확인

**노트북**: `01_person_a_v3.ipynb`

**설정:**
```yaml
experiment_name: E3_v2_mlp_optimized
projector_type: mlp_2l
projector_params: ~8,400,000

stage1_lr: 1e-3
stage2_lr: 5e-5
stage1_epochs: 2
stage2_epochs: 3

# 최적화 적용
warmup_ratio: 0.1
max_grad_norm: 1.0
vision_caching: true
diversity_monitoring: true
```

**예상 결과:**
- Diversity > 0.5
- SigLIP: Linear보다 약간 높음
- 비선형 변환으로 더 나은 표현력

---

### 3. 결과 비교 분석

| 지표 | E1-v2 (Linear) | E3-v2 (MLP) | E5-v2 (C-Abst) |
|------|----------------|-------------|----------------|
| Params | 4M | 8M | 206M |
| Diversity | ? | ? | 0.11 |
| SigLIP | ? | ? | 0.11 |
| METEOR | ? | ? | 0.066 |
| Mode Collapse | ? | ? | Yes |

**성공 기준:**
- Diversity > 0.5
- 합리적인 SigLIP (> 0.1)
- METEOR 유지 또는 향상

---

## 중기 계획 (결과에 따라)

### 시나리오 A: Linear/MLP가 성공

→ 단순 Projector 채택, 다음 단계로

1. **LoRA Ablation**: r=8, 16, 32 비교
2. **Frame Ablation**: 4, 8, 12 frames 비교
3. **전체 데이터셋 학습**: 8,563 train samples

### 시나리오 B: 모든 Projector가 실패

→ 구조적 문제 심층 분석

1. **데이터 품질 검토**
   - GT 캡션 다양성 확인
   - 비디오 품질 검토

2. **학습 방식 변경**
   - Contrastive Learning 추가
   - Caption Diversity Loss 추가

3. **Projector 구조 수정**
   - C-Abstractor num_queries 축소 (64 → 16)
   - Dropout 증가 (0.1 → 0.3)

---

## 장기 계획

### Phase 1: Best Projector 확정

| 단계 | 내용 | 예상 시간 |
|------|------|----------|
| 1 | E1-v2, E3-v2 실험 | 8-10시간 |
| 2 | 결과 비교 분석 | 1시간 |
| 3 | Best Projector 선정 | - |

### Phase 2: Ablation Study

| 단계 | 내용 | 예상 시간 |
|------|------|----------|
| 1 | LoRA rank (8, 16, 32) | 12-15시간 |
| 2 | Frame count (4, 8, 12) | 12-15시간 |
| 3 | 최적 설정 확정 | - |

### Phase 3: Full Training

| 단계 | 내용 | 예상 시간 |
|------|------|----------|
| 1 | 전체 데이터셋 (8,563 samples) 학습 | 24-48시간 |
| 2 | METEOR/BERTScore 최종 평가 | 2시간 |
| 3 | 결과 보고서 작성 | - |

---

## 실행 명령어

### E1-v2 (Linear) 실행

```bash
# Google Colab에서
# 1. 01_person_a_v2.ipynb 열기
# 2. Runtime → Run all
# 3. 결과 확인: results3/E1_v2_linear_optimized/
```

### E3-v2 (MLP) 실행

```bash
# Google Colab에서
# 1. 01_person_a_v3.ipynb 열기
# 2. Runtime → Run all
# 3. 결과 확인: results3/E3_v2_mlp_optimized/
```

### 결과 확인

```python
import json
from pathlib import Path

results_dir = Path("results3")

for exp in ["E1_v2_linear_optimized", "E3_v2_mlp_optimized"]:
    metrics_file = results_dir / exp / "final_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        print(f"\n{exp}:")
        print(f"  SigLIP: {m['siglip_score']:.4f}")
        print(f"  Diversity: {m.get('diversity', 'N/A')}")
        print(f"  METEOR: {m.get('meteor', 'N/A')}")
```

---

## 평가 기준 정리

### Primary: Diversity

```
Diversity >= 0.5  →  정상
Diversity < 0.5   →  Mode Collapse 의심
Diversity < 0.2   →  심각한 Mode Collapse
```

### Secondary: METEOR

```
METEOR > 0.07    →  개선
METEOR ~ 0.066   →  유지
METEOR < 0.05    →  악화
```

### Auxiliary: SigLIP

```
SigLIP > 0.15    →  양호
SigLIP 0.05-0.15 →  보통
SigLIP < 0.05    →  낮음 (Diversity와 함께 해석)
```

---

## 핵심 가설

> **"단순한 Projector (Linear, MLP)가 Mode Collapse 없이 학습되고,
> C-Abstractor보다 낮은 SigLIP에도 불구하고 METEOR/BERTScore는 유사하거나 더 높을 것이다."**

이 가설이 맞다면:
- 복잡한 Projector는 현재 데이터셋에 불필요
- Linear 또는 MLP-2L을 최종 선택

이 가설이 틀리다면:
- 데이터셋 또는 학습 방식의 근본적 문제
- 추가 분석 필요
