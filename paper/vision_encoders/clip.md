# CLIP: Learning Transferable Visual Models From Natural Language Supervision

- **저자**: Alec Radford, Jong Wook Kim, Chris Hallacy, et al. (OpenAI)
- **연도**: 2021
- **링크**: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

## 핵심 기여

1. **Contrastive Pre-training**: 4억 개 이미지-텍스트 쌍으로 대규모 학습
2. **Zero-shot Transfer**: 새로운 태스크에 fine-tuning 없이 적용 가능
3. **언어-비전 정렬**: 텍스트와 이미지를 동일한 임베딩 공간에 매핑
4. **VLM의 표준**: 대부분의 Vision-Language Model의 Vision Encoder로 채택

## 아키텍처

![ViT Architecture](../../../model_viz/outputs/vit_architecture.png)

![Vision Encoder Comparison](../../../model_viz/outputs/vision_encoder_table.png)

### Vision Encoder 옵션
| Model | Parameters | Input Size | Patch Size |
|-------|------------|------------|------------|
| ViT-B/32 | 86M | 224×224 | 32 |
| ViT-B/16 | 86M | 224×224 | 16 |
| ViT-L/14 | 304M | 224×224 | 14 |
| ViT-L/14@336 | 304M | 336×336 | 14 |

## Contrastive Learning

```python
# Pseudo-code for CLIP training
image_features = vision_encoder(images)  # [B, 512]
text_features = text_encoder(texts)      # [B, 512]

# Normalize
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

# Compute similarity
logits = image_features @ text_features.T * temperature

# Contrastive loss (symmetric)
labels = torch.arange(B)
loss_i2t = F.cross_entropy(logits, labels)
loss_t2i = F.cross_entropy(logits.T, labels)
loss = (loss_i2t + loss_t2i) / 2
```

## 우리 프로젝트 적용 포인트

1. **LLaVA 기본 Vision Encoder**: LLaVA-NeXT-Video는 CLIP ViT-L/14@336 사용
2. **Feature 추출**:
   - 이미지당 576 tokens (336/14 = 24, 24×24 = 576)
   - CLS token + patch tokens
3. **Frozen 사용**: 일반적으로 학습 시 frozen 유지
4. **한계**: 세밀한 공간 정보, 한국어 텍스트 이해 약함

## 구현 시 주의점

1. **해상도 선택**:
   - 224px: 메모리 효율적
   - 336px: 더 나은 품질 (LLaVA 기본)
2. **Patch 수 계산**:
   - 336px, patch 14 → (336/14)² = 576 tokens
3. **정규화**: ImageNet mean/std 사용
4. **텍스트 토큰화**: CLIP tokenizer (77 tokens max)

## 관련 코드/모델

- **Hugging Face**: `openai/clip-vit-large-patch14-336`
- **OpenCLIP**: `laion/CLIP-ViT-L-14-336-laion2B-s32B-b82K`
- **GitHub**: [openai/CLIP](https://github.com/openai/CLIP)

## 인용

```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
```
