# 실험 결과 상세

## E5-v2: C-Abstractor (LR 감소)

### 실험 설정

```yaml
experiment_name: E5_v2_lr_reduced
projector_type: c_abstractor
projector_params: 205,840,384

# LR 감소 (원본 대비)
stage1_lr: 1e-4    # 원본 1e-3 → 1/10
stage2_lr: 2e-5    # 원본 5e-5 → 2/5
stage1_epochs: 2
stage2_epochs: 3

# 공통 최적화
warmup_ratio: 0.1
max_grad_norm: 1.0
batch_size: 16
gradient_accumulation: 2
num_frames: 8
```

### 학습 결과

| Stage | Epoch | Train Loss | Val SigLIP | Diversity |
|-------|-------|------------|------------|-----------|
| 1 | 1 | 1.7908 | - | - |
| 1 | 2 | 1.6629 | - | - |
| 2 | 1 | 1.6425 | 0.0797 | 0.09 |
| 2 | 2 | 1.5722 | **0.1375** | 0.06 |
| 2 | 3 | 1.5151 | 0.1133 | **0.11** |

### 최종 결과

```
Final SigLIP2: 0.1133
Final Diversity: 0.11 (11/97 unique)
Final METEOR: 0.0661
Final BERTScore-F1: 0.7057
Total Time: 4h 47m
```

### 생성된 캡션 예시

| Sample | Generated | Ground Truth |
|--------|-----------|--------------|
| 1 | "영상은 도심의 한 구역을 촬영한 것으로, 화면 중앙에는 넓게 펼쳐진 건물들이..." | "영상은 한 도시에 있는 건축물을 담고 있습니다. 중앙에 보이는 건물의 색은 회색이며 직사각형..." |
| 2 | (동일) | "이 영상은 맑은 날씨에 촬영되었습니다. 화면 중앙에는 한국학술진흥재단 건물이..." |
| 3 | (동일) | "이 영상은 2000년대에 촬영된 것으로, 맑은 날 낮의 한옥을 주요 배경으로..." |

**문제점**: 97개 샘플 중 11개만 고유 캡션 → **Mode Collapse**

---

## E5-v3: C-Abstractor (Epoch 증가 + Early Stopping)

### 실험 설정

```yaml
experiment_name: E5_v3_epoch_increased
projector_type: c_abstractor
projector_params: 205,840,384

# Epoch 증가
stage1_lr: 1e-3    # 원본 유지
stage2_lr: 5e-5    # 원본 유지
stage1_epochs: 5   # 원본 2 → 5
stage2_epochs: 10  # 원본 3 → 10

# Early Stopping
early_stopping_patience: 3
early_stopping_min_delta: 0.001

# 공통 최적화
warmup_ratio: 0.1
max_grad_norm: 1.0
```

### 학습 결과

| Stage | Epoch | Train Loss | Val SigLIP | Diversity | Note |
|-------|-------|------------|------------|-----------|------|
| 1 | 1 | 1.8168 | - | - | |
| 1 | 2 | 1.6645 | - | - | |
| 1 | 3 | 1.6481 | - | - | |
| 1 | 4 | 1.6389 | - | - | |
| 1 | 5 | 1.6302 | - | - | |
| 2 | 1 | 1.6154 | 0.2355 | 0.09 | |
| 2 | 2 | 1.5601 | 0.2330 | 0.06 | No improve 1/3 |
| 2 | 3 | 1.4549 | **0.3383** | 0.04 | NEW BEST |
| 2 | 4 | 1.3828 | 0.1347 | **0.01** | No improve 1/3 |
| 2 | 5 | 1.3106 | 0.0430 | 0.04 | No improve 2/3 |
| 2 | 6 | 1.2581 | 0.0032 | 0.06 | **EARLY STOP** |

### SigLIP vs Diversity 추이

```
Epoch:     1      2      3      4      5      6
SigLIP:  0.24   0.23   0.34   0.13   0.04   0.00
Diversity: 9%    6%     4%     1%     4%     6%
          ↑      ↓      ↓      ↓↓     →      →
```

**관찰**: SigLIP이 최고점(0.34)일 때 Diversity가 최저(4%)
→ **높은 SigLIP ≠ 좋은 Projector**

### 최종 결과

```
Final SigLIP2: 0.0032
Final Diversity: 0.06 (6/97 unique)
Final METEOR: 0.0680
Final BERTScore-F1: 0.7002
Total Time: 7h 9m
Early Stopped at: Epoch 6/10
```

### 생성된 캡션 예시

| Sample | Generated | Ground Truth |
|--------|-----------|--------------|
| 1 | "영상은 맑고 화창한 낮의 광경을 담고 있습니다. 카메라는 고정된 상태로 촬영하고 있으며..." | "영상은 한 도시에 있는 건축물을 담고 있습니다..." |
| 2 | (동일) | "이 영상은 맑은 날씨에 촬영되었습니다. 화면 중앙에는 한국학술진흥재단 건물이..." |
| 3 | (동일) | "이 영상은 2000년대에 촬영된 것으로, 맑은 날 낮의 한옥을 주요 배경으로..." |

**문제점**: 97개 샘플 중 6개만 고유 캡션 → **심각한 Mode Collapse**

---

## 실험 비교

### E5-v2 vs E5-v3

| 지표 | E5-v2 (LR 감소) | E5-v3 (Epoch 증가) |
|------|-----------------|-------------------|
| Final SigLIP | 0.1133 | 0.0032 |
| Best SigLIP | 0.1375 | 0.3383 |
| Final Diversity | **0.11** | 0.06 |
| Unique Captions | 11/97 | 6/97 |
| METEOR | 0.0661 | 0.0680 |
| BERTScore-F1 | 0.7057 | 0.7002 |
| Training Time | 4h 47m | 7h 9m |
| Mode Collapse | **Yes** | **Yes (더 심각)** |

### 결론

| 시도 | 결과 |
|------|------|
| LR 감소 (1/10) | Mode Collapse 해결 실패 |
| Epoch 증가 (3.3배) | Mode Collapse 해결 실패, 오히려 악화 |
| Early Stopping | 학습 시간 절약에는 도움 |

**근본 원인**: C-Abstractor의 206M 파라미터가 과적합을 유발하거나, Cross-attention 구조가 다양한 Vision 정보를 구분하지 못함

---

## 다음 실험 (대기 중)

### E1-v2: Linear Projector (4M params)

```yaml
experiment_name: E1_v2_linear_optimized
projector_type: linear
projector_params: ~4,200,000  # 206M의 2%
```

### E3-v2: MLP-2L Projector (8M params)

```yaml
experiment_name: E3_v2_mlp_optimized
projector_type: mlp_2l
projector_params: ~8,400,000  # 206M의 4%
```

**가설**: 단순한 Projector가 Mode Collapse 없이 학습 가능
