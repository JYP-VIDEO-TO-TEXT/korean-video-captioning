#!/usr/bin/env python3
"""
Qwen3-8B Tokenizer 특성 테스트 스크립트.

테스트 항목:
1. PAD/EOS 토큰 설정 확인
2. 특수 토큰 (chat template, thinking mode) 확인
3. 한국어 캡션 토크나이징 테스트
4. Prompt + Caption 결합 시 토큰 길이/마스킹 검증

실행:
    CUDA_VISIBLE_DEVICES=0 python test_qwen3_tokenizer.py
"""

import sys
sys.path.insert(0, '/home/hj/2026/mutsa-02/korean_video_captioning/siglip_study')

import torch
from transformers import AutoTokenizer
from pathlib import Path
import json


def test_special_tokens(tokenizer):
    """특수 토큰 확인."""
    print("\n" + "="*60)
    print("1. 특수 토큰 확인")
    print("="*60)
    
    # 기본 특수 토큰
    print(f"\n[기본 특수 토큰]")
    print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"  UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    
    # PAD = EOS 문제 확인
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        print(f"\n  ⚠️  [WARNING] PAD token = EOS token!")
        print(f"      - 학습 시 PAD를 -100으로 마스킹하면 EOS도 마스킹됨")
        print(f"      - 모델이 문장 종료를 학습하지 못할 수 있음")
    
    # Special tokens map
    print(f"\n[Special Tokens Map]")
    for k, v in tokenizer.special_tokens_map.items():
        print(f"  {k}: {v}")
    
    # 추가 특수 토큰 (Qwen3 chat template 관련)
    print(f"\n[추가 특수 토큰 (vocab에서 검색)]")
    special_patterns = ['<|im_start|>', '<|im_end|>', '<think>', '</think>', 
                        '<|endoftext|>', '<|assistant|>', '<|user|>', '<|system|>']
    
    for pattern in special_patterns:
        try:
            token_id = tokenizer.convert_tokens_to_ids(pattern)
            if token_id != tokenizer.unk_token_id:
                print(f"  '{pattern}': ID={token_id}")
        except:
            pass
    
    # Vocab에서 thinking 관련 토큰 검색
    print(f"\n[Thinking Mode 관련 토큰]")
    think_tokens = [k for k in tokenizer.get_vocab().keys() if 'think' in k.lower()]
    if think_tokens:
        for t in think_tokens[:10]:  # 최대 10개만 출력
            print(f"  '{t}': ID={tokenizer.convert_tokens_to_ids(t)}")
    else:
        print("  (thinking 관련 토큰 없음)")
    
    return {
        'pad_token': tokenizer.pad_token,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token': tokenizer.eos_token,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_equals_eos': tokenizer.pad_token_id == tokenizer.eos_token_id,
    }


def test_korean_tokenization(tokenizer):
    """한국어 캡션 토크나이징 테스트."""
    print("\n" + "="*60)
    print("2. 한국어 캡션 토크나이징 테스트")
    print("="*60)
    
    # 테스트 캡션들
    test_captions = [
        "남자가 카메라를 향해 서 있습니다.",
        "여자가 책을 읽고 있습니다.",
        "아이들이 공원에서 뛰어놀고 있습니다.",
        "한 남성이 흰색 셔츠를 입고 마이크 앞에서 발표를 하고 있습니다. 뒤에는 프레젠테이션 화면이 보이며, 청중들이 앉아서 듣고 있습니다.",
        "영상에서는 두 사람이 대화를 나누고 있습니다. 첫 번째 사람은 검은색 정장을 입고 있으며, 두 번째 사람은 회색 스웨터를 입고 있습니다. 배경에는 사무실 인테리어가 보입니다.",
    ]
    
    results = []
    
    for i, caption in enumerate(test_captions, 1):
        tokens = tokenizer(caption, add_special_tokens=False)
        token_strs = tokenizer.convert_ids_to_tokens(tokens.input_ids)
        
        print(f"\n[캡션 {i}]")
        print(f"  원문: {caption[:50]}{'...' if len(caption) > 50 else ''}")
        print(f"  토큰 수: {len(tokens.input_ids)}")
        print(f"  문자당 토큰: {len(tokens.input_ids) / len(caption):.2f}")
        print(f"  처음 10개 토큰: {token_strs[:10]}")
        
        results.append({
            'caption': caption,
            'num_tokens': len(tokens.input_ids),
            'chars_per_token': len(caption) / len(tokens.input_ids) if tokens.input_ids else 0,
        })
    
    # 통계
    avg_tokens = sum(r['num_tokens'] for r in results) / len(results)
    print(f"\n[통계]")
    print(f"  평균 토큰 수: {avg_tokens:.1f}")
    print(f"  최소/최대: {min(r['num_tokens'] for r in results)} / {max(r['num_tokens'] for r in results)}")
    
    return results


def test_prompt_caption_combination(tokenizer, max_length=256):
    """Prompt + Caption 결합 테스트."""
    print("\n" + "="*60)
    print("3. Prompt + Caption 결합 테스트")
    print("="*60)
    
    prompt = "이 영상을 자세히 설명해주세요."
    caption = "한 남성이 흰색 셔츠를 입고 마이크 앞에서 발표를 하고 있습니다."
    
    # 개별 토크나이징
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    caption_tokens = tokenizer(caption, add_special_tokens=False)
    
    print(f"\n[개별 토크나이징]")
    print(f"  Prompt: '{prompt}'")
    print(f"    토큰 수: {len(prompt_tokens.input_ids)}")
    print(f"    토큰: {tokenizer.convert_ids_to_tokens(prompt_tokens.input_ids)}")
    print(f"  Caption: '{caption}'")
    print(f"    토큰 수: {len(caption_tokens.input_ids)}")
    
    # 결합 토크나이징
    full_text = f"{prompt} {caption}"
    full_tokens = tokenizer(full_text, add_special_tokens=False)
    
    print(f"\n[결합 토크나이징]")
    print(f"  Full text: '{full_text}'")
    print(f"  토큰 수: {len(full_tokens.input_ids)}")
    print(f"  예상 (prompt + caption): {len(prompt_tokens.input_ids) + len(caption_tokens.input_ids)}")
    print(f"  차이: {len(full_tokens.input_ids) - (len(prompt_tokens.input_ids) + len(caption_tokens.input_ids))}")
    
    # Padding + Truncation 테스트
    padded_tokens = tokenizer(
        full_text, 
        max_length=max_length, 
        padding="max_length", 
        truncation=True,
        return_tensors="pt"
    )
    
    print(f"\n[Padding/Truncation (max_length={max_length})]")
    print(f"  input_ids shape: {padded_tokens.input_ids.shape}")
    print(f"  실제 토큰 수: {(padded_tokens.attention_mask == 1).sum().item()}")
    print(f"  PAD 토큰 수: {(padded_tokens.attention_mask == 0).sum().item()}")
    
    # PAD 위치 확인
    pad_positions = (padded_tokens.input_ids[0] == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
    if len(pad_positions) > 0:
        print(f"  PAD 시작 위치: {pad_positions[0].item()}")
    
    return {
        'prompt_len': len(prompt_tokens.input_ids),
        'caption_len': len(caption_tokens.input_ids),
        'combined_len': len(full_tokens.input_ids),
        'padded_len': padded_tokens.input_ids.shape[1],
        'actual_tokens': (padded_tokens.attention_mask == 1).sum().item(),
    }


def test_pad_masking_for_loss(tokenizer, max_length=256):
    """PAD 토큰 마스킹이 Loss 계산에 미치는 영향 테스트."""
    print("\n" + "="*60)
    print("4. PAD 토큰 마스킹 (Loss 계산용) 테스트")
    print("="*60)
    
    prompt = "이 영상을 자세히 설명해주세요."
    caption = "남자가 카메라를 향해 서 있습니다."
    full_text = f"{prompt} {caption}"
    
    # Tokenize
    tokens = tokenizer(
        full_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = tokens.input_ids[0]
    attention_mask = tokens.attention_mask[0]
    
    # Labels 생성 (PAD 마스킹)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    print(f"\n[마스킹 전]")
    print(f"  input_ids 길이: {len(input_ids)}")
    print(f"  PAD 토큰 수: {(input_ids == tokenizer.pad_token_id).sum().item()}")
    
    print(f"\n[마스킹 후 (PAD -> -100)]")
    print(f"  -100 마스킹된 토큰 수: {(labels == -100).sum().item()}")
    print(f"  학습 대상 토큰 수: {(labels != -100).sum().item()}")
    
    # PAD = EOS 문제 분석
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        eos_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        print(f"\n[⚠️ PAD=EOS 문제 분석]")
        print(f"  EOS 토큰 위치: {eos_positions.tolist()[:10]}...")  # 처음 10개만
        print(f"  총 EOS 토큰 수: {len(eos_positions)}")
        
        # 실제 문장 끝 EOS vs PAD 구분
        actual_tokens = (attention_mask == 1).sum().item()
        print(f"  실제 토큰 영역: 0 ~ {actual_tokens-1}")
        
        # 실제 EOS (문장 끝)가 마스킹되는지 확인
        if actual_tokens > 0 and input_ids[actual_tokens-1] == tokenizer.eos_token_id:
            print(f"  ⚠️ 마지막 실제 토큰이 EOS이고 -100으로 마스킹됨!")
            print(f"     -> 모델이 문장 종료를 학습하지 못할 수 있음")
    
    # Prompt 부분 마스킹 테스트
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens.input_ids)
    
    labels_with_prompt_mask = input_ids.clone()
    labels_with_prompt_mask[labels_with_prompt_mask == tokenizer.pad_token_id] = -100
    labels_with_prompt_mask[:prompt_len] = -100  # Prompt 부분도 마스킹
    
    print(f"\n[Prompt + PAD 마스킹]")
    print(f"  Prompt 길이: {prompt_len}")
    print(f"  총 마스킹 토큰: {(labels_with_prompt_mask == -100).sum().item()}")
    print(f"  학습 대상 토큰 (caption only): {(labels_with_prompt_mask != -100).sum().item()}")
    
    return {
        'total_tokens': len(input_ids),
        'pad_tokens': (input_ids == tokenizer.pad_token_id).sum().item(),
        'masked_tokens': (labels == -100).sum().item(),
        'trainable_tokens': (labels != -100).sum().item(),
        'prompt_len': prompt_len,
    }


def test_chat_template(tokenizer):
    """Qwen3 Chat Template 테스트."""
    print("\n" + "="*60)
    print("5. Chat Template 테스트")
    print("="*60)
    
    # Chat template 존재 확인
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"\n[Chat Template 존재]")
        print(f"  {tokenizer.chat_template[:200]}...")
    else:
        print(f"\n[Chat Template 없음]")
    
    # apply_chat_template 테스트
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "user", "content": "이 영상을 자세히 설명해주세요."}
        ]
        
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"\n[apply_chat_template 결과]")
            print(f"  Input: {messages}")
            print(f"  Output: {formatted}")
            
            # 토큰화
            tokens = tokenizer(formatted, add_special_tokens=False)
            print(f"  토큰 수: {len(tokens.input_ids)}")
            
            return {'chat_template': formatted, 'token_len': len(tokens.input_ids)}
        except Exception as e:
            print(f"\n[apply_chat_template 에러]")
            print(f"  {e}")
            return None
    else:
        print(f"\n[apply_chat_template 메서드 없음]")
        return None


def main():
    print("="*60)
    print("Qwen3-8B Tokenizer 테스트")
    print("="*60)
    
    # Tokenizer 로드
    model_name = "Qwen/Qwen3-8B"
    print(f"\nLoading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # PAD 토큰 설정 (노트북과 동일)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token = eos_token")
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # 테스트 실행
    results = {}
    
    results['special_tokens'] = test_special_tokens(tokenizer)
    results['korean_tokenization'] = test_korean_tokenization(tokenizer)
    results['prompt_caption'] = test_prompt_caption_combination(tokenizer)
    results['pad_masking'] = test_pad_masking_for_loss(tokenizer)
    results['chat_template'] = test_chat_template(tokenizer)
    
    # 결과 저장
    output_path = Path(__file__).parent / "tokenizer_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # Convert non-serializable items
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable_results[k] = {
                    kk: (vv if not isinstance(vv, (torch.Tensor,)) else vv.tolist()) 
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                serializable_results[k] = v
            else:
                serializable_results[k] = str(v)
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")
    
    # 요약
    print("\n" + "="*60)
    print("테스트 요약")
    print("="*60)
    
    if results['special_tokens']['pad_equals_eos']:
        print("\n⚠️ [주의] PAD = EOS 설정됨")
        print("   - 해결책: 별도의 PAD 토큰 추가 또는 EOS 마스킹 제외 로직 필요")
    
    if results['chat_template']:
        print("\n✅ Chat Template 사용 가능")
        print("   - 권장: apply_chat_template 사용으로 프롬프트 일관성 향상")
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    main()
