# Speculative Decoding: Fast Inference with Draft Models

- **저자**: Yaniv Leviathan, et al. (Google)
- **연도**: 2023
- **링크**: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

## 핵심 기여

![Speculative Decoding](../../../model_viz/outputs/speculative_decoding.png)

1. **Draft-Verify 패러다임**: 작은 모델로 초안, 큰 모델로 검증
2. **무손실 가속**: 출력 분포 동일 유지
3. **2-3배 속도 향상**: 특히 긴 생성에서 효과적

## 핵심 아이디어

```
기존 Auto-regressive:
Token 1 → Token 2 → Token 3 → ... (순차적)

Speculative Decoding:
Draft Model: [Token 1, 2, 3, 4, 5] (빠르게 생성)
                       ↓
Target Model: [Verify 1, 2, 3] [Reject 4, 5] (병렬 검증)
                       ↓
Output: Token 1, 2, 3 + Resample from 4
```

## 우리 프로젝트 적용 포인트

1. **H100에서 권장**: 충분한 메모리 필요 (두 모델 로드)
2. **Draft 모델**: Qwen3-1.7B
3. **Target 모델**: Qwen3-14B/32B
4. **vLLM 지원**: vLLM에서 speculative decoding 기능 제공

## 관련 코드/모델

- **vLLM**: `--speculative-model` 옵션
- **논문**: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

## 인용

```bibtex
@inproceedings{leviathan2023speculative,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Leviathan, Yaniv and others},
  booktitle={ICML},
  year={2023}
}
```
