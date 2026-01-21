# DoRA: Weight-Decomposed Low-Rank Adaptation

- **저자**: Shih-Yang Liu, et al.
- **연도**: 2024
- **링크**: [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)

## 핵심 기여

![LoRA Architecture](../../../model_viz/outputs/lora_architecture.png)

> DoRA는 LoRA에 Magnitude 분해를 추가

1. **가중치 분해**: Magnitude와 Direction으로 분리
2. **LoRA 대비 성능 향상**: 동일 파라미터로 더 나은 결과
3. **안정적 학습**: Direction만 업데이트하여 안정성 향상

## 핵심 아이디어

```
LoRA:
W' = W + BA

DoRA:
W' = m * (W + BA) / ||W + BA||
   = magnitude * direction

- m: learnable magnitude (scalar)
- direction: normalized (W + BA)
```

## 우리 프로젝트 적용 포인트

1. **LoRA 대안**: 같은 r로 더 나은 성능
2. **PEFT 지원**: `pip install peft` 최신 버전에서 지원
3. **메모리**: LoRA와 거의 동일

## 관련 코드/모델

- **PEFT**: `DoraConfig` 사용
- **GitHub**: [NVlabs/DoRA](https://github.com/NVlabs/DoRA)

## 인용

```bibtex
@article{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and others},
  journal={arXiv preprint arXiv:2402.09353},
  year={2024}
}
```
