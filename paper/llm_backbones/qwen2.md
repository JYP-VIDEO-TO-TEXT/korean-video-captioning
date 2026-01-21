# Qwen2: A Comprehensive Language Model with Enhanced Capabilities

- **저자**: Alibaba Cloud
- **연도**: 2024
- **링크**: [GitHub](https://github.com/QwenLM/Qwen2)

## 핵심 기여

![Decoder Layer](../../../model_viz/outputs/decoder_layer.png)

![GQA Comparison](../../../model_viz/outputs/gqa_comparison.png)

1. **GQA (Grouped Query Attention)**: KV Cache 메모리 절약
2. **확장된 컨텍스트**: 최대 128K 토큰
3. **개선된 다국어**: 29개 언어 지원
4. **VL 모델**: Qwen2-VL로 비전 통합

## Qwen → Qwen2 개선점

| 특성 | Qwen | Qwen2 |
|------|------|-------|
| Attention | MHA | GQA |
| Context | 8K-32K | 32K-128K |
| Languages | ~10 | 29 |
| Vocab Size | 151,936 | 151,936 |

## 우리 프로젝트 적용 포인트

1. **Qwen3 이전 버전**: 안정성이 검증됨
2. **Qwen2-VL**: 비전 태스크 직접 지원
3. **현재는 Qwen3 권장**: 최신 버전 사용

## 관련 코드/모델

- **Hugging Face**: `Qwen/Qwen2-7B-Instruct`
- **VL 버전**: `Qwen/Qwen2-VL-7B-Instruct`

## 인용

```bibtex
@misc{qwen2,
  title={Qwen2 Technical Report},
  author={Alibaba Cloud},
  year={2024}
}
```
