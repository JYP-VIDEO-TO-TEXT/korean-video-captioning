# AWQ: Activation-aware Weight Quantization

- **저자**: Ji Lin, Jiaming Tang, Haotian Tang, et al. (MIT)
- **연도**: 2023
- **링크**: [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

## 핵심 기여

![Quantization Comparison](../../../model_viz/outputs/quantization_comparison.png)

1. **Activation-aware**: 중요한 가중치를 식별하여 보호
2. **Zero-shot Quantization**: Calibration 데이터 최소화
3. **INT4 without Retraining**: 재학습 없이 4-bit 양자화
4. **vLLM 호환**: 추론 가속과 함께 사용 가능

## 핵심 아이디어

모든 가중치가 동등하지 않다:
- Activation이 큰 채널의 가중치 = 중요
- Activation이 작은 채널의 가중치 = 덜 중요

Salient Weight를 스케일 조정하여 양자화 오류 최소화

## AWQ vs 다른 양자화

| Method | Calibration | Retraining | 품질 | 속도 |
|--------|-------------|------------|------|------|
| GPTQ | 필요 (많음) | 불필요 | 좋음 | 보통 |
| AWQ | 필요 (적음) | 불필요 | 매우 좋음 | 빠름 |
| QLoRA | 불필요 | 필요 | 좋음 | 학습용 |

## 우리 프로젝트 적용 포인트

1. **추론 시 AWQ 모델 사용**
2. **vLLM + AWQ 조합으로 최대 속도**
3. **A100/H100에서 권장**

## 관련 코드/모델

- **AutoAWQ**: `pip install autoawq`
- **GitHub**: [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)

## 인용

```bibtex
@inproceedings{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and others},
  booktitle={MLSys},
  year={2024}
}
```
