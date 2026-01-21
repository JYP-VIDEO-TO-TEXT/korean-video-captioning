# 모델 승인 목록

이 프로젝트에서 사용하는 모델 중 일부는 Hugging Face에서 승인이 필요합니다.

## 승인 필요 모델

### DINOv3 (Meta)

Meta의 DINOv3 비전 인코더는 gated model로, 사용 전 라이선스 동의가 필요합니다.

| 모델 ID | 파라미터 | 용도 | 승인 링크 |
|---------|----------|------|----------|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | 21M | 경량 실험용 | [승인 요청](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | 86M | 중간 규모 | [승인 요청](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | 300M | 고성능 권장 | [승인 요청](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| `facebook/dinov3-vith16-pretrain-lvd1689m` | 840M | 최고 성능 | [승인 요청](https://huggingface.co/facebook/dinov3-vith16-pretrain-lvd1689m) |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | 6.7B | 대규모 실험 | [승인 요청](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) |

**ConvNeXt 백본:**
| 모델 ID | 파라미터 | 승인 링크 |
|---------|----------|----------|
| `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | 28M | [승인 요청](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) |
| `facebook/dinov3-convnext-small-pretrain-lvd1689m` | 50M | [승인 요청](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m) |
| `facebook/dinov3-convnext-base-pretrain-lvd1689m` | 89M | [승인 요청](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m) |
| `facebook/dinov3-convnext-large-pretrain-lvd1689m` | 198M | [승인 요청](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m) |

### 승인 절차

1. Hugging Face 계정으로 로그인
2. 위 링크 중 사용할 모델 페이지 방문
3. **"Access repository"** 버튼 클릭
4. 라이선스(DINOv3 License) 동의 및 정보 입력
5. Meta 측 검토 후 승인 (수 시간 ~ 수일 소요)

> **참고**: 승인은 계정 단위로 이루어지므로, 한 번 승인받으면 해당 계정의 HF_TOKEN으로 모든 DINOv3 모델에 접근할 수 있습니다.

---

## 승인 불필요 모델 (바로 사용 가능)

### Qwen3 시리즈 (Alibaba, Apache-2.0)

| 모델 ID | 파라미터 | 특징 |
|---------|----------|------|
| `Qwen/Qwen3-0.6B-Instruct` | 0.6B | 초경량 |
| `Qwen/Qwen3-1.7B-Instruct` | 1.7B | 경량 |
| `Qwen/Qwen3-4B-Instruct` | 4B | T4 권장 |
| `Qwen/Qwen3-8B-Instruct` | 8B | L4 권장 |
| `Qwen/Qwen3-14B-Instruct` | 14B | A100 권장 |
| `Qwen/Qwen3-32B-Instruct` | 32B | H100 권장 |
| `Qwen/Qwen3-30B-A3B` | 30B (3B active) | MoE, 효율적 |
| `Qwen/Qwen3-235B-A22B` | 235B (22B active) | MoE, 최고 성능 |

**멀티모달 (비디오 지원):**
| 모델 ID | 특징 |
|---------|------|
| `Qwen/Qwen3-Omni` | 텍스트, 이미지, 오디오, 비디오 입력 |

### LLaVA 시리즈 (공개)

| 모델 ID | 파라미터 | 특징 |
|---------|----------|------|
| `llava-hf/LLaVA-NeXT-Video-7B-hf` | 7B | 비디오 캡셔닝 기본 |
| `llava-hf/LLaVA-NeXT-Video-34B-hf` | 34B | 고성능 |
| `llava-hf/llava-v1.6-mistral-7b-hf` | 7B | Mistral 기반 |

### Vision Encoder (공개)

| 모델 ID | 파라미터 | 특징 |
|---------|----------|------|
| `google/siglip-so400m-patch14-384` | 400M | 다국어, 고해상도 |
| `google/siglip-base-patch16-224` | 86M | 경량 |
| `openai/clip-vit-large-patch14-336` | 304M | 범용 |

---

## 접근성 테스트 코드

노트북 `00_setup_and_access_test.ipynb`에서 모델 접근 가능 여부를 테스트할 수 있습니다.

```python
from huggingface_hub import HfApi, login
from google.colab import userdata
import os

# Colab Secrets에서 토큰 로드
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
login(token=os.environ['HF_TOKEN'])

api = HfApi()

# 테스트할 모델 목록
MODELS = {
    # Gated models (승인 필요)
    "facebook/dinov3-vitl16-pretrain-lvd1689m": True,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": True,
    # Public models (승인 불필요)
    "Qwen/Qwen3-8B-Instruct": False,
    "llava-hf/LLaVA-NeXT-Video-7B-hf": False,
    "google/siglip-so400m-patch14-384": False,
}

print("=" * 60)
print("모델 접근성 테스트")
print("=" * 60)

for model_id, needs_approval in MODELS.items():
    try:
        info = api.model_info(model_id)
        print(f"[OK] {model_id}")
    except Exception as e:
        if needs_approval:
            print(f"[NEED APPROVAL] {model_id}")
            print(f"    승인 요청: https://huggingface.co/{model_id}")
        else:
            print(f"[ERROR] {model_id}")
            print(f"    오류: {e}")
```

---

## GPU별 권장 모델 조합

| GPU | VRAM | Vision Encoder | LLM Backbone |
|-----|------|----------------|--------------|
| T4 | 16GB | SigLIP-base (86M) | Qwen3-4B |
| L4 | 24GB | DINOv3-ViT-B (86M) | Qwen3-8B |
| A100 | 40GB | DINOv3-ViT-L (300M) | Qwen3-14B |
| H100 | 80GB | DINOv3-ViT-H (840M) | Qwen3-32B |

---

## 체크리스트

사용 전 아래 항목을 확인하세요:

- [ ] Hugging Face 계정 생성
- [ ] HF_TOKEN 발급 (Settings > Access Tokens)
- [ ] DINOv3 모델 승인 요청 (필요시)
- [ ] Colab Secrets에 HF_TOKEN 등록
- [ ] `00_setup_and_access_test.ipynb`로 접근성 확인
