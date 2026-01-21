# LLaVA: Visual Instruction Tuning

> **VLM의 시작점**: GPT-4를 활용한 학습 데이터 생성과 2-Stage Training의 기초를 제안

- **저자**: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
- **기관**: University of Wisconsin-Madison, Microsoft Research
- **연도**: 2023
- **링크**: [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)

---

## 핵심 기여

1. **Visual Instruction Tuning**: GPT-4를 활용하여 이미지-텍스트 instruction 데이터 자동 생성
2. **간단한 아키텍처**: Vision Encoder + Linear Projector + LLM 구조
3. **2-Stage Training**: Feature Alignment → End-to-End Fine-tuning
4. **Science QA SOTA**: 멀티모달 추론 벤치마크 최고 성능 달성 (당시)

---

## 아키텍처

![VLM Architecture](../../../model_viz/outputs/vlm_architecture.png)

> Vision Encoder → Projector → LLM의 기본 구조

### 구성 요소

| 컴포넌트 | 모델 | 역할 |
|---------|------|------|
| **Vision Encoder** | CLIP ViT-L/14 | 이미지 → Visual Features |
| **Projector** | Linear (768→4096) | Vision → Language 공간 정렬 |
| **LLM** | Vicuna-7B/13B | 텍스트 생성 |

### 토큰 계산

- **224×224 이미지**: 224/14 = 16 → 16×16 = 256 tokens
- **336×336 이미지**: 336/14 = 24 → 24×24 = 576 tokens
- **+ CLS 토큰**: 총 577 tokens

---

## 학습 데이터 생성

GPT-4를 활용하여 158K 샘플 자동 생성:

| 유형 | 샘플 수 | 설명 |
|------|--------|------|
| Conversation | 58K | 이미지에 대한 다중 턴 대화 |
| Detail Description | 23K | 상세한 이미지 설명 |
| Complex Reasoning | 77K | 이미지 기반 추론 질문 |

---

## 2-Stage Training

![2-Stage Training](../../../model_viz/outputs/two_stage_training.png)

### Stage 1: Feature Alignment

- **데이터**: CC3M 595K 이미지-캡션
- **학습**: Projector만 학습 (LLM, Vision Frozen)
- **설정**: 1 epoch, LR 2e-3, Batch 128
- **목표**: Visual feature를 Language embedding 공간에 정렬

### Stage 2: Visual Instruction Tuning

- **데이터**: LLaVA-Instruct-158K
- **학습**: Projector + LLM (LoRA)
- **설정**: 3 epochs, LR 2e-5, Batch 32
- **목표**: 멀티모달 대화/추론 능력 학습

---

## 우리 프로젝트 적용

### 프롬프트 형식

```python
prompt = """USER: <image>
이 이미지를 한국어로 상세히 설명해주세요.
ASSISTANT:"""
```

### 구현 시 주의점

1. Vision Encoder는 일반적으로 frozen 상태로 사용
2. Projector 초기화: Random, LR 높게 설정 (1e-3)
3. LLM LoRA 설정: r=128, alpha=256, attention layers만
4. Visual tokens가 많아 메모리 관리 필요

---

## 성능

| 벤치마크 | LLaVA-7B | LLaVA-13B |
|---------|----------|-----------|
| Science QA | 90.92% | 91.68% |
| MMBench | 64.3 | 67.7 |
| VQA v2 | 76.3 | 80.0 |

---

## 관련 리소스

- **Hugging Face**: `llava-hf/llava-1.5-7b-hf`
- **GitHub**: [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **데이터셋**: [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
