#!/usr/bin/env python3
"""
Qwen3 모델 테스트 전체 실행 스크립트.

실행:
    CUDA_VISIBLE_DEVICES=0 python run_all_tests.py
"""

import subprocess
import sys
from pathlib import Path


def run_test(script_name: str) -> bool:
    """단일 테스트 스크립트 실행."""
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'#'*60}")
    print(f"# Running: {script_name}")
    print(f"{'#'*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed: {script_name}")
        print(f"   Exit code: {e.returncode}")
        return False


def main():
    print("="*60)
    print("Qwen3 모델 테스트 전체 실행")
    print("="*60)
    
    tests = [
        "test_qwen3_tokenizer.py",
        "test_qwen3_generation.py",
    ]
    
    results = {}
    
    for test in tests:
        results[test] = run_test(test)
    
    # 결과 요약
    print("\n" + "="*60)
    print("전체 테스트 결과")
    print("="*60)
    
    for test, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test}")
    
    # 결과 파일 위치
    print("\n[결과 파일]")
    output_dir = Path(__file__).parent
    for result_file in output_dir.glob("*_results.json"):
        print(f"  {result_file}")
    
    # 전체 성공 여부
    all_passed = all(results.values())
    print(f"\n전체 결과: {'✅ 모든 테스트 통과' if all_passed else '❌ 일부 테스트 실패'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
