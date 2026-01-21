#!/usr/bin/env python3
"""
Qwen3-8B Generation 특성 테스트 스크립트.

테스트 항목:
1. Generation Config 확인
2. Thinking Mode 확인 및 비활성화 테스트
3. inputs_embeds 기반 생성 테스트
4. 생성 텍스트 품질 및 길이 분석

실행:
    CUDA_VISIBLE_DEVICES=0 python test_qwen3_generation.py
"""

import sys
sys.path.insert(0, '/home/hj/2026/mutsa-02/korean_video_captioning/siglip_study')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import json
import time


def load_model_and_tokenizer():
    """모델과 토크나이저 로드."""
    model_name = "Qwen/Qwen3-8B"
    print(f"\nLoading model: {model_name}")
    
    # 4-bit 양자화 설정 (노트북과 동일)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    
    return model, tokenizer


def test_generation_config(model, tokenizer):
    """Generation Config 확인."""
    print("\n" + "="*60)
    print("1. Generation Config 확인")
    print("="*60)
    
    gen_config = model.generation_config
    
    print(f"\n[Generation Config 속성]")
    important_attrs = [
        'max_length', 'max_new_tokens', 'min_length',
        'do_sample', 'temperature', 'top_p', 'top_k',
        'repetition_penalty', 'no_repeat_ngram_size',
        'pad_token_id', 'eos_token_id', 'bos_token_id',
    ]
    
    for attr in important_attrs:
        if hasattr(gen_config, attr):
            value = getattr(gen_config, attr)
            print(f"  {attr}: {value}")
    
    # Thinking mode 관련 속성 확인
    print(f"\n[Thinking Mode 관련 속성]")
    thinking_attrs = ['enable_thinking', 'thinking_budget', 'think_start_token_id', 'think_end_token_id']
    for attr in thinking_attrs:
        if hasattr(gen_config, attr):
            print(f"  {attr}: {getattr(gen_config, attr)}")
        else:
            print(f"  {attr}: (속성 없음)")
    
    # 모든 속성 출력
    print(f"\n[모든 Generation Config 속성]")
    for attr in dir(gen_config):
        if not attr.startswith('_'):
            try:
                value = getattr(gen_config, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                pass
    
    return gen_config


def test_thinking_mode(model, tokenizer):
    """Thinking Mode 테스트."""
    print("\n" + "="*60)
    print("2. Thinking Mode 테스트")
    print("="*60)
    
    prompt = "이 영상을 자세히 설명해주세요."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    results = {}
    
    # 1. 기본 생성 (do_sample=False)
    print(f"\n[1] 기본 생성 (do_sample=False)")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"  Input: {prompt}")
    print(f"  Output (raw): {generated[:200]}...")
    print(f"  Output (clean): {generated_clean[:200]}...")
    print(f"  생성 토큰 수: {outputs.shape[1] - inputs.input_ids.shape[1]}")
    
    # Thinking 토큰 확인
    if '<think>' in generated or '</think>' in generated:
        print(f"  ⚠️ Thinking 토큰 발견!")
        results['thinking_detected_basic'] = True
    else:
        print(f"  ✅ Thinking 토큰 없음")
        results['thinking_detected_basic'] = False
    
    # 2. enable_thinking=False 시도 (지원하는 경우)
    print(f"\n[2] enable_thinking=False 테스트")
    try:
        with torch.no_grad():
            outputs_no_think = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                enable_thinking=False,  # Thinking 비활성화 시도
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_no_think = tokenizer.decode(outputs_no_think[0], skip_special_tokens=False)
        print(f"  Output: {generated_no_think[:200]}...")
        results['enable_thinking_supported'] = True
        
        if '<think>' in generated_no_think:
            print(f"  ⚠️ enable_thinking=False에도 Thinking 토큰 발견")
        else:
            print(f"  ✅ Thinking 비활성화 성공")
            
    except (TypeError, ValueError) as e:
        print(f"  enable_thinking 파라미터 미지원: {e}")
        results['enable_thinking_supported'] = False
    
    # 3. Chat template 사용 시 테스트
    print(f"\n[3] Chat Template 사용 테스트")
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # add_generation_prompt=True로 assistant 응답 유도
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"  Formatted prompt: {formatted[:100]}...")
            
            inputs_chat = tokenizer(formatted, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs_chat = model.generate(
                    **inputs_chat,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_chat = tokenizer.decode(outputs_chat[0], skip_special_tokens=False)
            print(f"  Output: {generated_chat[len(formatted):][:200]}...")
            
            if '<think>' in generated_chat:
                print(f"  ⚠️ Chat template 사용 시에도 Thinking 토큰 발견")
                results['thinking_with_chat_template'] = True
            else:
                print(f"  ✅ Chat template 사용 시 Thinking 없음")
                results['thinking_with_chat_template'] = False
                
        except Exception as e:
            print(f"  Chat template 에러: {e}")
            results['chat_template_error'] = str(e)
    
    return results


def test_inputs_embeds_generation(model, tokenizer):
    """inputs_embeds 기반 생성 테스트 (CustomVLM 방식)."""
    print("\n" + "="*60)
    print("3. inputs_embeds 기반 생성 테스트")
    print("="*60)
    
    prompt = "이 영상을 자세히 설명해주세요."
    
    # 텍스트 임베딩 직접 추출
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # 임베딩 레이어에서 직접 추출
        text_embeds = model.get_input_embeddings()(inputs.input_ids)
        print(f"  Text embeddings shape: {text_embeds.shape}")
        
        # 가상의 vision embeddings (노트북에서 CLIP 출력을 projector 통과시킨 것)
        # CLIP ViT-L: 576 patches (24x24) per frame, 4 frames = 2304 tokens
        # 여기서는 간단히 100개 토큰으로 시뮬레이션
        vision_token_count = 100
        vision_embeds = torch.randn(
            1, vision_token_count, text_embeds.shape[-1], 
            device=model.device, dtype=text_embeds.dtype
        )
        print(f"  Vision embeddings shape: {vision_embeds.shape}")
        
        # 결합: [vision_embeds | text_embeds]
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        print(f"  Combined embeddings shape: {combined_embeds.shape}")
    
    # inputs_embeds 기반 생성
    print(f"\n[inputs_embeds 기반 생성]")
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 생성된 토큰 분석
        generated_ids = outputs[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"  생성 토큰 수: {len(generated_ids)}")
        print(f"  생성 텍스트: {generated_text[:300]}...")
        
        # 토큰 분석
        tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
        print(f"  처음 20개 토큰: {tokens[:20]}")
        print(f"  마지막 20개 토큰: {tokens[-20:]}")
        
        # 반복 패턴 확인
        unique_tokens = len(set(tokens))
        print(f"  총 토큰: {len(tokens)}, 고유 토큰: {unique_tokens}, 비율: {unique_tokens/len(tokens):.2%}")
        
        if unique_tokens / len(tokens) < 0.3:
            print(f"  ⚠️ 반복 생성 의심 (고유 토큰 비율 낮음)")
        
        return {
            'success': True,
            'generated_tokens': len(generated_ids),
            'generated_text_len': len(generated_text),
            'unique_token_ratio': unique_tokens / len(tokens),
        }
        
    except Exception as e:
        print(f"  ❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_generation_length(model, tokenizer):
    """생성 길이 제어 테스트."""
    print("\n" + "="*60)
    print("4. 생성 길이 제어 테스트")
    print("="*60)
    
    prompt = "이 영상을 자세히 설명해주세요."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    results = []
    
    for max_new_tokens in [50, 100, 256]:
        print(f"\n[max_new_tokens={max_new_tokens}]")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - start_time
        
        generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"  생성 토큰: {generated_tokens}")
        print(f"  생성 시간: {gen_time:.2f}s")
        print(f"  텍스트 길이 (chars): {len(generated_text)}")
        print(f"  텍스트: {generated_text[:100]}...")
        
        # EOS로 종료되었는지 확인
        if generated_tokens < max_new_tokens:
            print(f"  ✅ EOS로 조기 종료 (max_new_tokens 미달)")
        else:
            print(f"  ⚠️ max_new_tokens까지 생성 (EOS 없이 종료)")
        
        results.append({
            'max_new_tokens': max_new_tokens,
            'generated_tokens': generated_tokens,
            'text_len': len(generated_text),
            'time': gen_time,
            'early_stop': generated_tokens < max_new_tokens,
        })
    
    return results


def test_repetition_penalty(model, tokenizer):
    """반복 방지 파라미터 테스트."""
    print("\n" + "="*60)
    print("5. 반복 방지 파라미터 테스트")
    print("="*60)
    
    prompt = "이 영상을 자세히 설명해주세요."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    configs = [
        {'name': 'baseline', 'kwargs': {}},
        {'name': 'repetition_penalty=1.2', 'kwargs': {'repetition_penalty': 1.2}},
        {'name': 'no_repeat_ngram_size=3', 'kwargs': {'no_repeat_ngram_size': 3}},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n[{config['name']}]")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **config['kwargs']
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(outputs[0].tolist())
        
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
        
        print(f"  텍스트: {generated_text[:150]}...")
        print(f"  고유 토큰 비율: {unique_ratio:.2%}")
        
        results.append({
            'config': config['name'],
            'text': generated_text[:200],
            'unique_ratio': unique_ratio,
        })
    
    return results


def main():
    print("="*60)
    print("Qwen3-8B Generation 테스트")
    print("="*60)
    
    # 모델 로드
    model, tokenizer = load_model_and_tokenizer()
    
    # 테스트 실행
    results = {}
    
    results['generation_config'] = str(test_generation_config(model, tokenizer))
    results['thinking_mode'] = test_thinking_mode(model, tokenizer)
    results['inputs_embeds'] = test_inputs_embeds_generation(model, tokenizer)
    results['generation_length'] = test_generation_length(model, tokenizer)
    results['repetition_penalty'] = test_repetition_penalty(model, tokenizer)
    
    # 결과 저장
    output_path = Path(__file__).parent / "generation_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n결과 저장: {output_path}")
    
    # 요약
    print("\n" + "="*60)
    print("테스트 요약")
    print("="*60)
    
    if results.get('thinking_mode', {}).get('thinking_detected_basic'):
        print("\n⚠️ [주의] Thinking Mode가 기본 활성화되어 있을 수 있음")
        print("   - 생성 결과에 <think>...</think> 토큰이 포함될 수 있음")
        print("   - 이로 인해 생성 길이가 예상보다 길어질 수 있음")
    
    if not results.get('inputs_embeds', {}).get('success'):
        print("\n❌ [에러] inputs_embeds 기반 생성 실패")
        print("   - CustomVLM의 generate() 함수에 문제가 있을 수 있음")
    
    print("\n테스트 완료!")
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
