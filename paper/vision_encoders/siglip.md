# SigLIP: Sigmoid Loss for Language Image Pre-Training

- **저자**: Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer (Google)
- **연도**: 2023
- **링크**: [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)

## 핵심 기여

1. **Sigmoid Loss**: Softmax 대신 Sigmoid 사용으로 배치 크기 의존성 감소
2. **다국어 지원**: 109개 언어 텍스트로 학습 (한국어 포함!)
3. **배치 독립성**: 큰 배치 없이도 안정적 학습
4. **Fine-grained 이해**: CLIP 대비 세밀한 시각적 이해 향상

## 아키텍처

![ViT Architecture](../../../model_viz/outputs/vit_architecture.png)

## Sigmoid vs Softmax Loss

### CLIP (Softmax Cross-Entropy)
```python
# Softmax-based contrastive loss
logits = image_features @ text_features.T * temperature
loss = F.cross_entropy(logits, labels)  # 배치 내 모든 negative 필요
```

### SigLIP (Sigmoid Binary Cross-Entropy)
```python
# Sigmoid-based pairwise loss
logits = image_features @ text_features.T * temperature
labels = torch.eye(B)  # 대각선만 positive

# Binary cross-entropy per pair
loss = F.binary_cross_entropy_with_logits(logits, labels)
```

**장점**:
- 배치 크기에 덜 민감
- 각 pair를 독립적으로 처리
- 더 안정적인 학습

## 모델 변형

| Model | Parameters | Resolution | 특징 |
|-------|------------|------------|------|
| SigLIP-B/16 | 86M | 224 | 기본 |
| SigLIP-L/16 | 304M | 256 | 대형 |
| SigLIP-So400M | 400M | 384 | 최고 성능 |

## 우리 프로젝트 적용 포인트

1. **다국어 장점**: 한국어 텍스트-이미지 정렬이 CLIP보다 우수
2. **L4 이상 권장**: SigLIP-So400M 사용 시 메모리 여유 필요
3. **Vision Encoder 교체**:
   ```python
   from transformers import SiglipVisionModel
   
   vision_encoder = SiglipVisionModel.from_pretrained(
       "google/siglip-so400m-patch14-384"
   )
   ```
4. **Projector 재학습 필요**: CLIP→SigLIP 교체 시 Stage 1부터 재학습

## 구현 시 주의점

1. **해상도 차이**: 384px 기본 (CLIP 336px 대비 높음)
2. **토큰 수**: (384/14)² = 729 tokens (CLIP 576 대비 많음)
3. **Projector 차원**: hidden_size가 CLIP과 다를 수 있음
4. **전처리**: SigLIP 전용 image processor 사용

## 성능 비교 (ImageNet Zero-shot)

| Model | Top-1 Acc | 한국어 COCO |
|-------|-----------|-------------|
| CLIP ViT-L/14 | 75.3% | 낮음 |
| SigLIP-L/16 | 77.2% | 보통 |
| SigLIP-So400M | 79.1% | 높음 |

## 관련 코드/모델

- **Hugging Face**: `google/siglip-so400m-patch14-384`
- **Variants**:
  - `google/siglip-base-patch16-224`
  - `google/siglip-large-patch16-256`
- **GitHub**: [google-research/big_vision](https://github.com/google-research/big_vision)

## 인용

```bibtex
@inproceedings{zhai2023sigmoid,
  title={Sigmoid loss for language image pre-training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  booktitle={ICCV},
  year={2023}
}
```
