# LLaMA: Open and Efficient Foundation Language Models

- **저자**: Hugo Touvron, et al. (Meta AI)
- **연도**: 2023
- **링크**: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

## 핵심 기여

1. **효율적인 사전학습**: 1T 토큰으로 GPT-3 수준 성능
2. **오픈소스**: 연구 커뮤니티 활성화
3. **다양한 크기**: 7B, 13B, 33B, 65B
4. **LLaVA의 기반**: 대부분의 오픈소스 VLM의 LLM 백본

## 아키텍처 특징

![Decoder Layer](../../../model_viz/outputs/decoder_layer.png)

- **Pre-normalization**: RMSNorm
- **SwiGLU Activation**: FFN에서 사용
- **Rotary Position Embedding (RoPE)**
- **No Bias**: 대부분의 레이어에서 bias 제거

## 우리 프로젝트 적용 포인트

1. **역사적 의의**: LLaVA의 기본 LLM
2. **Vicuna = LLaMA + SFT**: 대화 능력 추가
3. **현재는 Qwen3 권장**: 한국어 성능이 더 좋음

## 관련 코드/모델

- **Hugging Face**: `meta-llama/Llama-2-7b-hf`
- **GitHub**: [facebookresearch/llama](https://github.com/facebookresearch/llama)

## 인용

```bibtex
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```
