# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

- **저자**: Elias Frantar, et al.
- **연도**: 2022
- **링크**: [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

## 핵심 기여

![Quantization Comparison](../../../model_viz/outputs/quantization_comparison.png)

1. **Post-Training Quantization**: 재학습 없이 양자화
2. **One-shot 양자화**: 레이어별 순차 처리
3. **3-4 bit 양자화**: 극단적 압축 가능
4. **OPT-175B 양자화**: 대형 모델에서도 작동

## GPTQ vs AWQ

| 특성 | GPTQ | AWQ |
|------|------|-----|
| 속도 | 보통 | 빠름 |
| 품질 | 좋음 | 매우 좋음 |
| Calibration | 많이 필요 | 적게 필요 |
| vLLM 지원 | 지원 | 지원 |

## 우리 프로젝트 적용 포인트

1. **AWQ 대안**: AWQ 모델이 없을 때
2. **TheBloke 모델**: GPTQ 버전 다수 제공
3. **AutoGPTQ**: 직접 양자화 가능

## 관련 코드/모델

- **AutoGPTQ**: `pip install auto-gptq`
- **GitHub**: [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)

## 인용

```bibtex
@inproceedings{frantar2023gptq,
  title={GPTQ: Accurate Post-Training Quantization},
  author={Frantar, Elias and others},
  booktitle={ICLR},
  year={2023}
}
```
