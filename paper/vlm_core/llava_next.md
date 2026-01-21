# LLaVA-NeXT: Improved Baselines with Visual Instruction Tuning

- **저자**: Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, et al.
- **연도**: 2024
- **링크**: [Blog](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

## 핵심 기여

![AnyRes Processing](../../../model_viz/outputs/anyres_processing.png)

1. **AnyRes**: 다양한 해상도와 종횡비 지원
2. **고해상도 입력**: 최대 672×672 지원
3. **더 나은 LLM**: Vicuna → Mistral/Nous-Hermes 등
4. **향상된 데이터**: 고품질 instruction 데이터 확대

## AnyRes (Any Resolution)

```
기존 방식:
Image → Resize to 336×336 → Single patch

AnyRes:
Image (다양한 크기) → Split into grids → Multiple patches
                                   ↓
                            Thumbnail + Grid patches
```

### Grid 분할 예시
```
1024×768 이미지:
┌─────┬─────┐
│ 336 │ 336 │  ← 고해상도 패치 (2×2 grid)
├─────┼─────┤
│ 336 │ 336 │
└─────┴─────┘
    +
┌─────┐
│ 336 │  ← Thumbnail (전체 맥락)
└─────┘

총 visual tokens: 576 × (4 + 1) = 2,880
```

## 우리 프로젝트 적용 포인트

1. **LLaVA-NeXT-Video의 기반**: 고해상도 처리 방식 이해
2. **비디오 적용**:
   - 각 프레임을 AnyRes로 처리
   - Spatial pooling으로 토큰 수 조절
3. **메모리 관리**:
   - T4: 단일 해상도 (336) 권장
   - A100+: AnyRes 활용 가능

## 관련 코드/모델

- **Hugging Face**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **GitHub**: [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)

## 인용

```bibtex
@misc{liu2024llavanext,
  title={LLaVA-NeXT: Improved baselines with visual instruction tuning},
  author={Liu, Haotian and others},
  year={2024}
}
```
