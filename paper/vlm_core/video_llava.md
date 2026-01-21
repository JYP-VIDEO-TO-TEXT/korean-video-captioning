# Video-LLaVA: Learning United Visual Representation

- **저자**: Bin Lin, Bin Zhu, Yang Ye, et al.
- **연도**: 2024
- **링크**: [arXiv:2311.10122](https://arxiv.org/abs/2311.10122)

## 핵심 기여

![VLM Architecture](../../../model_viz/outputs/vlm_architecture.png)

1. **통합 비주얼 표현**: 이미지와 비디오를 하나의 표현 공간에서 처리
2. **LanguageBind**: 다중 모달리티 정렬을 위한 인코더
3. **Joint Training**: 이미지와 비디오 데이터 동시 학습
4. **효율적 비디오 인코딩**: 프레임 수에 선형적인 계산량

## 아키텍처

```
Image → [LanguageBind Image Encoder] ─┐
                                      ├→ [Linear Projector] → LLM
Video → [LanguageBind Video Encoder] ─┘

LanguageBind:
- 이미지, 비디오, 오디오 등 다양한 모달리티
- 언어와 정렬된 공통 임베딩 공간
```

## 우리 프로젝트 적용 포인트

1. **대안 모델**: LLaVA-NeXT-Video 대신 사용 가능
2. **장점**: 이미지+비디오 통합 학습으로 일반화 성능 향상
3. **비디오 특화**: 시간적 맥락 이해가 더 나을 수 있음

## 관련 코드/모델

- **Hugging Face**: `LanguageBind/Video-LLaVA-7B-hf`
- **GitHub**: [PKU-YuanGroup/Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)

## 인용

```bibtex
@inproceedings{lin2024videollava,
  title={Video-LLaVA: Learning United Visual Representation},
  author={Lin, Bin and Zhu, Bin and Ye, Yang and others},
  booktitle={CVPR},
  year={2024}
}
```
