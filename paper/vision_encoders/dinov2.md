# DINOv2: Learning Robust Visual Features without Supervision

- **저자**: Maxime Oquab, Timothée Darcet, et al. (Meta AI)
- **연도**: 2023
- **링크**: [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)

## 핵심 기여

1. **Self-supervised Learning**: 레이블 없이 강력한 시각 특징 학습
2. **Teacher-Student Distillation**: 대규모 teacher에서 작은 student로 지식 전달
3. **LVD-142M 데이터셋**: 1.42억 이미지로 사전학습
4. **Dense Prediction**: 세그멘테이션, 깊이 추정 등에 강함

## 아키텍처

![ViT Architecture](../../../model_viz/outputs/vit_architecture.png)

## DINOv2 vs CLIP

| 특성 | CLIP | DINOv2 |
|------|------|--------|
| 학습 방식 | Contrastive (text-image) | Self-distillation |
| 텍스트 정렬 | 강함 | 약함 |
| Dense features | 약함 | 강함 |
| Zero-shot | 좋음 | 제한적 |
| Fine-grained | 보통 | 매우 좋음 |

## 우리 프로젝트 적용 포인트

1. **DINOv3 대안**: 승인 필요 없이 바로 사용 가능
2. **세밀한 시각 이해**: 배경 묘사에 유용
3. **Projector 재학습 필요**: CLIP과 다른 특징 공간

## 모델 변형

| Model | Parameters | 접근성 |
|-------|------------|--------|
| ViT-S/14 | 21M | 공개 |
| ViT-B/14 | 86M | 공개 |
| ViT-L/14 | 300M | 공개 |
| ViT-g/14 | 1.1B | 공개 |

## 관련 코드/모델

- **Hugging Face**: `facebook/dinov2-large`
- **GitHub**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

## 인용

```bibtex
@inproceedings{oquab2024dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and others},
  booktitle={TMLR},
  year={2024}
}
```
