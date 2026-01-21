# Qwen: A Comprehensive Language Model Series

- **저자**: Alibaba Cloud
- **연도**: 2023
- **링크**: [arXiv:2309.16609](https://arxiv.org/abs/2309.16609)

## 핵심 기여

![Decoder Layer](../../../model_viz/outputs/decoder_layer.png)

1. **다국어 특화**: 중국어, 한국어 등 아시아 언어 강점
2. **긴 컨텍스트**: 8K~32K 토큰 지원
3. **상업적 사용 가능**: 일부 버전 Apache-2.0
4. **다양한 모달리티**: Qwen-VL, Qwen-Audio 등

## 한국어 성능

| Model | MMLU-Ko | KoBEST |
|-------|---------|--------|
| LLaMA-7B | 34.5% | 낮음 |
| Vicuna-7B | 38.2% | 보통 |
| Qwen-7B | 52.1% | 높음 |

## 우리 프로젝트 적용 포인트

1. **한국어 성능**: LLaMA 계열 대비 우수
2. **Qwen3으로 발전**: 더 나은 버전 사용 권장
3. **VLM 지원**: Qwen-VL 시리즈도 고려 가능

## 관련 코드/모델

- **Hugging Face**: `Qwen/Qwen-7B`
- **GitHub**: [QwenLM/Qwen](https://github.com/QwenLM/Qwen)

## 인용

```bibtex
@article{bai2023qwen,
  title={Qwen Technical Report},
  author={Bai, Jinze and others},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```
