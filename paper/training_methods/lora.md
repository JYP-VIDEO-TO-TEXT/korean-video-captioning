# LoRA: Low-Rank Adaptation of Large Language Models

- **저자**: Edward J. Hu, Yelong Shen, Phillip Wallis, et al. (Microsoft)
- **연도**: 2021
- **링크**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

## 핵심 기여

1. **파라미터 효율성**: 전체 모델의 0.1~1%만 학습
2. **메모리 효율성**: Optimizer states 대폭 감소
3. **추론 오버헤드 없음**: 학습 후 원본 가중치에 병합 가능
4. **모듈성**: 여러 LoRA를 교체하며 다양한 태스크 수행

## 핵심 아이디어

![LoRA Architecture](../../../model_viz/outputs/lora_architecture.png)

### Low-Rank Decomposition
```
기존 Fine-tuning:
W' = W + ΔW  (ΔW: d×k 행렬, 전체 파라미터)

LoRA:
W' = W + BA  (B: d×r, A: r×k, r << min(d,k))
            (r: rank, 보통 8~64)
```

### 수학적 표현
```python
# Original: h = Wx
# LoRA: h = Wx + BAx = Wx + (scaling * BA)x

class LoRALayer:
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        self.A = nn.Linear(in_features, rank, bias=False)   # Down-projection
        self.B = nn.Linear(rank, out_features, bias=False)  # Up-projection
        self.scaling = alpha / rank
        
        # Initialization
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)  # 초기에는 LoRA 영향 없음
    
    def forward(self, x):
        return self.B(self.A(x)) * self.scaling
```

## 하이퍼파라미터

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `r` (rank) | Low-rank 차원 | 8~64 |
| `alpha` | Scaling factor | 2×r |
| `target_modules` | 적용 대상 | attention layers |
| `dropout` | LoRA dropout | 0.05~0.1 |

### Rank 선택 가이드
- **r=8**: 메모리 제약 (T4), 간단한 태스크
- **r=16**: 균형 (L4), 일반적 사용
- **r=32**: 복잡한 태스크 (A100)
- **r=64**: 최고 품질 (H100)

## 우리 프로젝트 적용 포인트

1. **Target Modules**:
   ```python
   target_modules = [
       "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
       "gate_proj", "up_proj", "down_proj"       # MLP (선택적)
   ]
   ```

2. **GPU별 설정**:
   | GPU | r | alpha | modules |
   |-----|---|-------|---------|
   | T4 | 8 | 16 | attn only |
   | L4 | 16 | 32 | attn only |
   | A100 | 32 | 64 | attn + mlp |
   | H100 | 64 | 128 | attn + mlp |

3. **PEFT 라이브러리 사용**:
   ```python
   from peft import LoraConfig, get_peft_model
   
   lora_config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM"
   )
   model = get_peft_model(model, lora_config)
   ```

## 구현 시 주의점

1. **초기화**: B를 0으로 초기화하여 초기에는 원본 모델과 동일
2. **Learning Rate**: 일반 fine-tuning보다 높게 (1e-4 ~ 2e-4)
3. **병합**: 추론 시 `model.merge_and_unload()`로 오버헤드 제거
4. **여러 LoRA**: 어댑터 교체로 멀티태스크 지원

## 파라미터 수 계산

```python
# 7B 모델, r=16, attention layers만
# Attention: q, k, v, o 각각 (hidden_size × r + r × hidden_size) × num_layers

hidden_size = 4096
num_layers = 32
r = 16

params_per_layer = 4 * (hidden_size * r + r * hidden_size)
total_params = params_per_layer * num_layers
# = 4 * (4096 * 16 + 16 * 4096) * 32
# = 4 * 131072 * 32
# ≈ 16.7M (전체 7B의 0.24%)
```

## 관련 코드/모델

- **PEFT 라이브러리**: `pip install peft`
- **GitHub**: [microsoft/LoRA](https://github.com/microsoft/LoRA)
- **Hugging Face**: [PEFT Documentation](https://huggingface.co/docs/peft)

## 인용

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and others},
  booktitle={ICLR},
  year={2022}
}
```
