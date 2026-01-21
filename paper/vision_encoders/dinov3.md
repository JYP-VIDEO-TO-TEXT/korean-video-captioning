# DINOv3: Dense Visual Features via Gram Anchoring

- **저자**: Meta AI Research
- **연도**: 2024
- **링크**: [Hugging Face](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

## 핵심 기여

![ViT Architecture](../../../model_viz/outputs/vit_architecture.png)

1. **Gram Anchoring**: Self-supervised learning의 새로운 앵커링 방법
2. **Dense Features**: 픽셀 수준의 세밀한 특징 추출
3. **Semantic Segmentation**: 추가 학습 없이 zero-shot segmentation 가능
4. **다양한 스케일**: ViT-S부터 ViT-H까지 다양한 크기 제공

## DINOv2 vs DINOv3

| 특성 | DINOv2 | DINOv3 |
|------|--------|--------|
| Loss | Self-distillation | Gram Anchoring |
| Features | Semantic | Dense + Semantic |
| Segmentation | 좋음 | 더 좋음 |
| 학습 데이터 | LVD-142M | LVD-1689M / SAT-493M |

## Gram Anchoring

```
기존 Self-distillation:
Teacher → Student (CLS token 정렬)

Gram Anchoring:
Teacher features → Gram Matrix → Anchor
Student features → Gram Matrix → Match Anchor

Gram Matrix: 특징 간의 상관관계를 캡처
→ 로컬 패턴의 스타일/텍스처 정보 보존
→ Dense prediction 성능 향상
```

## 모델 변형

| Model | Parameters | Resolution | 용도 |
|-------|------------|------------|------|
| ViT-S/16 | 21M | 518 | 경량 |
| ViT-B/16 | 86M | 518 | 기본 |
| ViT-L/16 | 300M | 518 | 고품질 |
| ViT-H+/16 | 632M | 518 | 최고 성능 |
| ViT-7B/16 | 7B | 518 | 초대형 |

### Pretrain 변형
- **LVD-1689M**: 대규모 데이터셋 (추천)
- **SAT-493M**: 위성 이미지 특화

## 우리 프로젝트 적용 포인트

1. **모델 접근 (승인 필요)**:
   - Meta 라이선스 동의 필요
   - [Hugging Face 페이지](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)에서 승인

2. **Vision Encoder 교체**:
   ```python
   from transformers import Dinov2Model, AutoImageProcessor
   
   # DINOv3 (dinov2 클래스 사용)
   vision_encoder = Dinov2Model.from_pretrained(
       "facebook/dinov3-vitl16-pretrain-lvd1689m",
       token=HF_TOKEN  # 승인된 토큰 필요
   )
   processor = AutoImageProcessor.from_pretrained(
       "facebook/dinov3-vitl16-pretrain-lvd1689m"
   )
   ```

3. **GPU별 선택**:
   | GPU | 권장 모델 | 메모리 |
   |-----|----------|--------|
   | T4 | SigLIP (대안) | - |
   | L4 | ViT-B/16 | ~8GB |
   | A100 | ViT-L/16 | ~12GB |
   | H100 | ViT-H+/16 | ~20GB |

## 구현 시 주의점

1. **승인 필수**: Gated model, HF 토큰 + 라이선스 동의
2. **해상도**: 518×518 기본 (CLIP 336과 다름)
3. **토큰 수**: (518/14)² ≈ 1369 tokens (많음!)
4. **Projector 재학습**: CLIP→DINOv3 교체 시 필수

## Feature 특성

```python
# DINOv3 출력
outputs = vision_encoder(images)

# CLS token: 전역 의미 특징
cls_token = outputs.last_hidden_state[:, 0]  # [B, 1024]

# Patch tokens: 로컬 dense 특징
patch_tokens = outputs.last_hidden_state[:, 1:]  # [B, 1369, 1024]

# Dense prediction에 유용
# 비디오 내 물체/배경 구분에 강함
```

## 대안 (승인 어려울 경우)

DINOv3 접근이 어려우면:
1. **DINOv2**: 공개 모델, 성능 좋음
2. **SigLIP**: 다국어 지원, 접근 용이
3. **InternViT**: 중국어/한국어 특화

## 관련 코드/모델

- **Hugging Face**: 
  - `facebook/dinov3-vits16-pretrain-lvd1689m`
  - `facebook/dinov3-vitb16-pretrain-lvd1689m`
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`
- **대안**: `facebook/dinov2-large` (승인 불필요)

## 인용

```bibtex
@misc{dinov3,
  title={DINOv3: Dense Visual Features via Gram Anchoring},
  author={Meta AI Research},
  year={2024},
  howpublished={\url{https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m}}
}
```
