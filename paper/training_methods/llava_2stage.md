# LLaVA 2-Stage Training Strategy

- **출처**: LLaVA 논문 (Liu et al., 2023)
- **링크**: [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)

## 핵심 아이디어

멀티모달 학습을 두 단계로 분리하여 효율적이고 안정적인 학습

![2-Stage Training](../../../model_viz/outputs/two_stage_training.png)

## Stage 1: Feature Alignment (Pre-training)

```
목표: Vision과 Language 임베딩 공간 정렬

데이터: Image-Caption pairs (CC3M 595K)
학습 대상: Projector만
Frozen: Vision Encoder, LLM

Epochs: 1
Learning Rate: 1e-3
```

## Stage 2: Visual Instruction Tuning

```
목표: 멀티모달 대화/추론 능력 학습

데이터: LLaVA-Instruct (158K)
학습 대상: Projector + LLM (LoRA)
Frozen: Vision Encoder

Epochs: 3
Learning Rate: 2e-5
```

## 우리 프로젝트 적용

### Stage 1 (선택적)
- 한국어 캡션 데이터로 정렬
- Vision Encoder 교체 시 필수

### Stage 2 (필수)
- AI-Hub 비디오 캡셔닝 데이터
- QLoRA로 fine-tuning

## 인용

```bibtex
@inproceedings{liu2023visual,
  title={Visual instruction tuning},
  author={Liu, Haotian and others},
  booktitle={NeurIPS},
  year={2023}
}
```
