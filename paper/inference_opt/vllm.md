# vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention

- **저자**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. (UC Berkeley)
- **연도**: 2023
- **링크**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

## 핵심 기여

1. **PagedAttention**: OS의 가상 메모리 개념을 KV Cache에 적용
2. **2-4배 처리량 향상**: 기존 HuggingFace 대비
3. **메모리 효율성**: KV Cache 파편화 해결
4. **배치 처리**: Continuous batching으로 GPU 활용 극대화

## KV Cache 동작 원리

![KV Cache](../../../model_viz/outputs/kv_cache.png)

> Prefill (한 번에 처리) → Decode (캐시 재사용)

## PagedAttention

### 기존 방식의 문제
```
Request 1: [KV Cache ████████████░░░░░░░░░░░░░░░░░]
Request 2: [KV Cache ██████████████████░░░░░░░░░░░]
Request 3: [KV Cache ████░░░░░░░░░░░░░░░░░░░░░░░░░]
                    ^ 연속 메모리 할당, 파편화 발생

문제:
- 최대 길이만큼 사전 할당
- 짧은 시퀀스도 큰 메모리 점유
- 파편화로 배치 크기 제한
```

### PagedAttention 해결책
```
Physical Blocks: [Block 0][Block 1][Block 2][Block 3]...
                    ↑        ↑        ↑
Request 1: Page Table [0] → Block 0
                      [1] → Block 2
Request 2: Page Table [0] → Block 1
                      [1] → Block 3

장점:
- 필요한 만큼만 블록 할당
- 블록 단위 재사용
- 메모리 파편화 최소화
```

## Continuous Batching

```
기존 Static Batching:
[Req 1 ████████████]
[Req 2 ████████]     ← 짧은 요청도 대기
[Req 3 ██████████████████]
       ↑ 모든 요청 완료까지 대기

Continuous Batching:
[Req 1 ████████████]
[Req 2 ████████][Req 4 ██████]  ← 완료 즉시 새 요청 시작
[Req 3 ██████████████████]
       ↑ GPU 항상 활용
```

## 우리 프로젝트 적용 포인트

1. **추론 서버 구축** (A100/H100):
   ```python
   from vllm import LLM, SamplingParams
   
   llm = LLM(
       model="llava-hf/LLaVA-NeXT-Video-7B-hf",
       tensor_parallel_size=1,
       gpu_memory_utilization=0.9,
   )
   
   sampling_params = SamplingParams(
       temperature=0.7,
       top_p=0.9,
       max_tokens=256,
   )
   
   outputs = llm.generate(prompts, sampling_params)
   ```

2. **배치 추론**:
   ```python
   # 여러 비디오 동시 처리
   prompts = [
       f"USER: <video>{video1}설명 ASSISTANT:",
       f"USER: <video>{video2}설명 ASSISTANT:",
       ...
   ]
   outputs = llm.generate(prompts, sampling_params)  # 배치 처리
   ```

3. **GPU별 설정**:
   | GPU | tensor_parallel | gpu_memory_utilization |
   |-----|-----------------|------------------------|
   | A100 | 1 | 0.9 |
   | H100 | 1 | 0.9 |
   | 2×A100 | 2 | 0.85 |

## 구현 시 주의점

1. **Vision 모델 지원**: vLLM 0.4.0+ 필요
2. **메모리 할당**: 첫 요청 시 메모리 프리로딩
3. **LoRA 어댑터**: vLLM에서 LoRA 로드 지원
4. **양자화**: AWQ/GPTQ 모델 지원

## 성능 비교

| Framework | Throughput (req/s) | Latency (ms) |
|-----------|-------------------|--------------|
| HuggingFace | 1.0x | 1.0x |
| TGI | 2.1x | 0.6x |
| vLLM | **2.4x** | **0.5x** |

## 관련 코드/모델

- **PyPI**: `pip install vllm`
- **GitHub**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Documentation**: [vLLM Docs](https://docs.vllm.ai/)

## 사용 예시 (서버 모드)

```bash
# API 서버 시작
python -m vllm.entrypoints.openai.api_server \
    --model llava-hf/LLaVA-NeXT-Video-7B-hf \
    --port 8000

# 클라이언트에서 호출
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "llava", "prompt": "USER: <video>...", "max_tokens": 256}'
```

## 인용

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and others},
  booktitle={SOSP},
  year={2023}
}
```
