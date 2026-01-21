# SigLIP Study 실험 보고서

> **연구 목표**: Vision Projector 아키텍처 비교를 통한 최적 Vision-Language 정렬 달성

## 프로젝트 개요

### 배경
- 기존 METEOR 평가는 LLM의 언어 능력과 Projector 효과를 분리 불가
- SigLIP Score를 활용한 Vision-Language 정렬 직접 측정 방식 채택

### 모델 구성

```
┌─────────────────────────────────────────────────────────────┐
│                     Custom VLM Architecture                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Vision    │    │  Projector  │    │      LLM        │ │
│  │   Encoder   │───▶│  (학습)     │───▶│   (LoRA)        │ │
│  │  (Frozen)   │    │             │    │                 │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│                                                             │
│  CLIP-ViT-L/14      4종 비교          Qwen3-8B (4-bit)     │
│  336px, 1024-dim    - Linear (4M)                          │
│                     - MLP-2L (8M)                          │
│                     - C-Abstractor (206M)                  │
│                     - Perceiver (134M)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 실험 결과 요약

### 실험 목록

| 실험 | Projector | Params | Diversity | SigLIP | METEOR | 상태 |
|------|-----------|--------|-----------|--------|--------|------|
| E1 | Linear | 4M | - | - | - | 기존 실험 |
| E3 | MLP-2L | 8M | - | - | - | 기존 실험 |
| E5 | C-Abstractor | 206M | - | - | - | 기존 실험 |
| **E5-v2** | C-Abstractor | 206M | **0.11** | 0.113 | 0.066 | ✅ 완료 |
| **E5-v3** | C-Abstractor | 206M | **0.06** | 0.003 | 0.068 | ✅ 완료 |
| E1-v2 | Linear | 4M | - | - | - | 대기 |
| E3-v2 | MLP-2L | 8M | - | - | - | 대기 |

### 핵심 발견

1. **C-Abstractor Mode Collapse 확인**
   - 206M 파라미터의 큰 모델이 Mode Collapse 발생
   - LR 감소, Epoch 증가 모두 해결 실패
   - Diversity 0.11 이하 (97개 샘플 중 6-11개만 고유 캡션)

2. **SigLIP 단독 평가의 한계**
   - 높은 SigLIP ≠ 좋은 Projector
   - 일반적인 캡션이 높은 SigLIP 획득 가능
   - **Diversity와 함께 평가 필수**

3. **다음 단계: 단순 Projector 재검토**
   - Linear, MLP가 Mode Collapse 없이 학습 가능한지 확인 필요

---

## 적용된 최적화

### 품질 개선

| 최적화 | 효과 |
|--------|------|
| SigLIP2 (다국어) | 한국어 캡션 평가 가능 |
| LR Scheduler | Warmup + Cosine Decay로 안정적 학습 |
| Gradient Clipping | 학습 안정성 향상 |
| Diversity Monitoring | Mode Collapse 조기 탐지 |
| Early Stopping | 불필요한 학습 방지 (patience=3) |

### 학습 시간 단축

| 최적화 | 효과 |
|--------|------|
| **Vision Features 캐싱** | **30-40% 속도 향상** |
| Mixed Precision (bfloat16) | 메모리 절약 + 연산 가속 |
| 4-bit 양자화 (NF4) | LLM 메모리 1/4로 절약 |
| Gradient Checkpointing | 큰 모델 학습 가능 |
| DataLoader 최적화 | I/O 병목 해소 |

---

## 상세 문서

| 문서 | 내용 |
|------|------|
| [01_experiment_results.md](01_experiment_results.md) | 실험별 상세 결과 및 비교 |
| [02_optimizations_applied.md](02_optimizations_applied.md) | 적용된 최적화 기법 상세 |
| [03_mode_collapse_analysis.md](03_mode_collapse_analysis.md) | Mode Collapse 분석 |
| [04_next_steps.md](04_next_steps.md) | 다음 단계 계획 |

---

## 노트북 구성

### 기존 노트북 (원본)

| 노트북 | Projector | 실험 |
|--------|-----------|------|
| `01_person_a.ipynb` | Linear, MLP | E1, E3 |
| `02_person_b.ipynb` | C-Abstractor | E5 |
| `03_person_c.ipynb` | Perceiver | E7 |

### 최적화 버전 (v2/v3)

| 노트북 | Projector | 핵심 변경 |
|--------|-----------|----------|
| `01_person_a_v2.ipynb` | **Linear** | 공통 최적화 적용 |
| `01_person_a_v3.ipynb` | **MLP-2L** | 공통 최적화 적용 |
| `02_person_b_v2.ipynb` | C-Abstractor | LR 감소 (1e-4, 2e-5) |
| `02_person_b_v3.ipynb` | C-Abstractor | Epoch 증가 + Early Stopping |

---

## 실험 환경

- **플랫폼**: Google Colab (A100 80GB)
- **Vision Encoder**: CLIP-ViT-L/14-336 (frozen)
- **LLM**: Qwen3-8B (4-bit NF4 양자화)
- **PEFT**: LoRA (r=16, alpha=32)
- **데이터**: AI-Hub 샘플 (865 train, 97 val)
- **프레임**: 8 frames/video
- **결과 저장**: Google Drive `siglip_study/results3/`

---

## 타임라인

| 날짜 | 진행 내용 |
|------|----------|
| 2026-01-20 | E5-v2, E5-v3 실험 완료 |
| 2026-01-20 | Mode Collapse 확인 및 분석 |
| 2026-01-21 | Linear/MLP 최적화 노트북 생성 |
| 다음 | E1-v2, E3-v2 실험 예정 |

---

## 결론 (현재까지)

> **C-Abstractor (206M)는 현재 데이터셋/설정에서 Mode Collapse 발생**
> 
> 단순한 Projector (Linear 4M, MLP 8M)가 더 나은 결과를 보일 가능성 높음

다음 단계: E1-v2 (Linear), E3-v2 (MLP) 실험으로 가설 검증
