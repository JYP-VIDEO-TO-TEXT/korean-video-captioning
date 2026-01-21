# 05. 평가 지표 상세

## 개요

본 문서에서는 대한민국 배경영상 캡셔닝 모델의 평가 지표를 상세히 설명합니다. 주요 지표인 METEOR를 중심으로, 보조 지표인 CIDEr, BERTScore, CLIPScore 등을 다룹니다.

---

## 1. 평가 지표 분류

### 1.1 지표 유형

```
┌─────────────────────────────────────────────────────────────────┐
│                    평가 지표 분류                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. N-gram 기반 (표면적 유사도)                                 │
│  ├── BLEU: 정밀도 기반 n-gram 매칭                              │
│  ├── ROUGE: 재현율 기반 n-gram 매칭                             │
│  └── METEOR: 정밀도 + 재현율 + 동의어/어간                      │
│                                                                  │
│  2. 이미지 캡셔닝 특화                                          │
│  ├── CIDEr: TF-IDF 가중 n-gram (캡셔닝 최적화)                 │
│  └── SPICE: 시맨틱 그래프 기반                                  │
│                                                                  │
│  3. 의미적 유사도 (임베딩 기반)                                 │
│  ├── BERTScore: BERT 임베딩 유사도                              │
│  └── BLEURT: 학습된 메트릭                                      │
│                                                                  │
│  4. 멀티모달 평가                                               │
│  └── CLIPScore: 이미지-텍스트 정렬도                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 지표 선택 기준

| 지표 | 한국어 지원 | 캡셔닝 적합성 | 계산 비용 | 권장 여부 |
|------|------------|--------------|----------|----------|
| **METEOR** | 우수 | 높음 | 중간 | **주요 지표** |
| **CIDEr** | 양호 | 매우 높음 | 중간 | **보조 지표** |
| **BERTScore** | 우수 | 높음 | 높음 | **보조 지표** |
| BLEU | 보통 | 중간 | 낮음 | 참고용 |
| ROUGE | 보통 | 중간 | 낮음 | 참고용 |
| CLIPScore | 양호 | 매우 높음 | 높음 | 선택적 |

---

## 2. METEOR (주요 지표)

### 2.1 METEOR 개요

**METEOR** (Metric for Evaluation of Translation with Explicit ORdering)는 기계번역 평가를 위해 개발되었지만, 캡셔닝 평가에도 널리 사용됩니다.

**AI-Hub 베이스라인:**
- 한국어: **0.3052**
- 영어: **0.33**

### 2.2 METEOR 계산 원리

```
METEOR 계산 과정:

1. 단어 정렬 (Alignment)
   - Exact Match: 정확히 일치하는 단어
   - Stem Match: 어간이 일치하는 단어
   - Synonym Match: 동의어 관계 단어 (WordNet)
   - Paraphrase Match: 구문 수준 의역

2. Precision & Recall 계산
   
   Precision = |matched| / |hypothesis|
   Recall = |matched| / |reference|
   
   예시:
   Reference:  "푸른 바다 위로 하얀 파도가 부서진다"
   Hypothesis: "파란 바다에 하얀 파도가 치고 있다"
   
   Matched: 바다, 하얀, 파도 (+ 푸른≈파란 동의어)
   Precision = 4/7 = 0.57
   Recall = 4/6 = 0.67

3. F-mean 계산 (조화 평균)
   
   F = (P × R) / (α × P + (1-α) × R)
   α = 0.9 (recall 가중)
   
   F = (0.57 × 0.67) / (0.9 × 0.57 + 0.1 × 0.67) = 0.64

4. Fragmentation Penalty
   
   - 정렬된 단어들이 얼마나 연속적인지 측정
   - Chunks: 연속된 정렬 그룹 수
   
   Penalty = γ × (chunks / matched)^β
   γ = 0.5, β = 3
   
   예시: 2 chunks, 4 matched
   Penalty = 0.5 × (2/4)³ = 0.0625

5. 최종 METEOR Score
   
   METEOR = F × (1 - Penalty)
   METEOR = 0.64 × (1 - 0.0625) = 0.60
```

### 2.3 METEOR의 장점 (캡셔닝 태스크)

```
BLEU vs METEOR 비교:

Reference: "아름다운 해변에서 파도가 밀려온다"

Case 1: 동의어 사용
Hypothesis: "예쁜 바닷가에서 파도가 밀려온다"

BLEU-4: 0.32 (정확한 n-gram 매칭 부족)
METEOR: 0.78 (동의어 인정: 아름다운≈예쁜, 해변≈바닷가)

→ METEOR가 의미적 유사성을 더 잘 포착

Case 2: 어순 변화
Hypothesis: "파도가 아름다운 해변에서 밀려온다"

BLEU-4: 0.45 (어순 변화로 n-gram 깨짐)
METEOR: 0.85 (단어 정렬 기반으로 어순 변화 허용)

→ METEOR가 한국어의 자유로운 어순에 강건
```

### 2.4 한국어 METEOR 설정

```python
# 한국어 METEOR 계산

from nltk.translate.meteor_score import meteor_score
from konlpy.tag import Okt

# 한국어 형태소 분석기
okt = Okt()

def tokenize_korean(text):
    """한국어 형태소 분석 기반 토큰화"""
    morphs = okt.morphs(text)
    return morphs

def calculate_meteor_korean(reference, hypothesis):
    """
    한국어 METEOR 계산
    
    Parameters:
    - reference: 정답 캡션 (str)
    - hypothesis: 생성된 캡션 (str)
    
    Returns:
    - METEOR score (float)
    """
    ref_tokens = tokenize_korean(reference)
    hyp_tokens = tokenize_korean(hypothesis)
    
    # METEOR 계산
    score = meteor_score(
        [ref_tokens],  # reference는 리스트로 감싸야 함
        hyp_tokens,
        alpha=0.9,     # recall 가중치
        beta=3.0,      # penalty 강도
        gamma=0.5,     # penalty 계수
    )
    
    return score

# 사용 예시
reference = "푸른 바다 위로 하얀 파도가 부서지며 해변으로 밀려온다"
hypothesis = "파란 바다에 하얀 파도가 치며 해변으로 다가온다"

score = calculate_meteor_korean(reference, hypothesis)
print(f"METEOR Score: {score:.4f}")  # 예: 0.7523
```

### 2.5 METEOR 개선 방향

```python
# 한국어 특화 METEOR 개선

class EnhancedKoreanMETEOR:
    """
    한국어 특화 METEOR 계산기
    
    개선 사항:
    1. KorLex (한국어 워드넷) 기반 동의어 매칭
    2. 조사 제거 정규화
    3. 어간 추출 (stemming) 강화
    """
    
    def __init__(self):
        self.okt = Okt()
        # 한국어 동의어 사전 로드 (KorLex 또는 유사 리소스)
        self.synonyms = self._load_korean_synonyms()
    
    def _load_korean_synonyms(self):
        """한국어 동의어 사전 로드"""
        # 예시 동의어 사전
        return {
            "푸른": ["파란", "청색의", "푸르른"],
            "아름다운": ["예쁜", "고운", "수려한"],
            "바다": ["해양", "바닷가", "해변"],
            "파도": ["물결", "너울"],
            # ... 더 많은 동의어
        }
    
    def normalize_text(self, text):
        """텍스트 정규화"""
        # 조사 제거
        morphs = self.okt.pos(text)
        content_words = [
            word for word, pos in morphs
            if pos not in ['Josa', 'Punctuation']
        ]
        return content_words
    
    def synonym_match(self, word1, word2):
        """동의어 매칭 확인"""
        if word1 == word2:
            return True
        
        # 동의어 사전에서 확인
        if word1 in self.synonyms and word2 in self.synonyms[word1]:
            return True
        if word2 in self.synonyms and word1 in self.synonyms[word2]:
            return True
        
        return False
    
    def calculate(self, reference, hypothesis):
        """개선된 METEOR 계산"""
        ref_tokens = self.normalize_text(reference)
        hyp_tokens = self.normalize_text(hypothesis)
        
        # 매칭 수행 (정확, 어간, 동의어)
        matches = self._find_matches(ref_tokens, hyp_tokens)
        
        # Precision & Recall
        precision = len(matches) / len(hyp_tokens) if hyp_tokens else 0
        recall = len(matches) / len(ref_tokens) if ref_tokens else 0
        
        # F-mean
        alpha = 0.9
        if precision + recall > 0:
            f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        else:
            f_mean = 0
        
        # Fragmentation Penalty
        penalty = self._calculate_penalty(matches, len(matches))
        
        return f_mean * (1 - penalty)
```

---

## 3. CIDEr (캡셔닝 특화 지표)

### 3.1 CIDEr 개요

**CIDEr** (Consensus-based Image Description Evaluation)는 이미지 캡셔닝 평가를 위해 특별히 설계된 지표입니다.

### 3.2 CIDEr 계산 원리

```
CIDEr 계산 과정:

1. TF-IDF 가중치 계산
   
   TF (Term Frequency): n-gram이 캡션에서 등장하는 빈도
   IDF (Inverse Document Frequency): 전체 캡션에서의 희소성
   
   g_k(s) = (h_k(s) / Σ h_l(s)) × log(|I| / Σ min(1, Σ h_k(s_j)))
   
   직관:
   - 모든 캡션에 등장하는 흔한 표현 ("있다", "보인다") → 낮은 가중치
   - 특정 이미지에만 관련된 표현 ("파도", "해변") → 높은 가중치

2. Cosine Similarity 계산
   
   CIDEr_n(c, S) = (1/m) × Σ (g^n(c) · g^n(s_j)) / (||g^n(c)|| × ||g^n(s_j)||)
   
   - c: 생성된 캡션
   - S: 참조 캡션 집합 (여러 개 가능)
   - n: n-gram 크기

3. 최종 CIDEr-D Score
   
   CIDEr-D = (1/N) × Σ CIDEr_n
   
   보통 N=4 (1-gram ~ 4-gram 평균)
```

### 3.3 CIDEr 구현

```python
from pycocoevalcap.cider.cider import Cider

def calculate_cider(references, hypotheses):
    """
    CIDEr 점수 계산
    
    Parameters:
    - references: {image_id: [cap1, cap2, ...]} 형태
    - hypotheses: {image_id: [generated_cap]} 형태
    
    Returns:
    - CIDEr score (float)
    """
    cider = Cider()
    score, scores = cider.compute_score(references, hypotheses)
    return score

# 사용 예시
references = {
    "video_001": [
        "푸른 바다 위로 하얀 파도가 부서진다",
        "해변에 파도가 밀려오고 있다",
    ],
}

hypotheses = {
    "video_001": ["파란 바다에서 하얀 파도가 치고 있다"],
}

score = calculate_cider(references, hypotheses)
print(f"CIDEr Score: {score:.4f}")
```

### 3.4 CIDEr의 장점

| 특징 | 설명 |
|------|------|
| **TF-IDF 가중** | 중요한 단어에 더 높은 가중치 |
| **다중 참조** | 여러 정답 캡션 활용 가능 |
| **인간 평가 상관** | 다른 지표 대비 인간 평가와 높은 상관 |
| **캡셔닝 최적화** | 이미지 캡셔닝 태스크를 위해 설계 |

---

## 4. BERTScore (의미적 유사도)

### 4.1 BERTScore 개요

**BERTScore**는 BERT 임베딩을 사용하여 의미적 유사도를 측정합니다.

### 4.2 BERTScore 계산 원리

```
BERTScore 계산 과정:

1. 토큰 임베딩 추출
   
   Reference:  [r1, r2, r3, ..., rn] (BERT 토큰 임베딩)
   Hypothesis: [h1, h2, h3, ..., hm] (BERT 토큰 임베딩)

2. Cosine Similarity Matrix
   
   ┌─────┬─────┬─────┬─────┐
   │     │ r1  │ r2  │ r3  │
   ├─────┼─────┼─────┼─────┤
   │ h1  │0.9  │0.3  │0.1  │
   │ h2  │0.2  │0.8  │0.4  │
   │ h3  │0.1  │0.5  │0.95 │
   └─────┴─────┴─────┴─────┘

3. Precision (각 hypothesis 토큰의 최대 유사도 평균)
   
   P = (1/m) × Σ max_j cos(h_i, r_j)
   P = (0.9 + 0.8 + 0.95) / 3 = 0.883

4. Recall (각 reference 토큰의 최대 유사도 평균)
   
   R = (1/n) × Σ max_i cos(h_i, r_j)
   R = (0.9 + 0.8 + 0.95) / 3 = 0.883

5. F1 Score
   
   F1 = 2 × (P × R) / (P + R)
   F1 = 2 × 0.883 × 0.883 / 1.766 = 0.883
```

### 4.3 한국어 BERTScore 구현

```python
from bert_score import score

def calculate_bertscore_korean(references, hypotheses):
    """
    한국어 BERTScore 계산
    
    Parameters:
    - references: 정답 캡션 리스트
    - hypotheses: 생성된 캡션 리스트
    
    Returns:
    - P, R, F1 scores
    """
    P, R, F1 = score(
        hypotheses,
        references,
        model_type="bert-base-multilingual-cased",  # 다국어 BERT
        # 또는 한국어 특화 모델:
        # model_type="klue/bert-base",
        lang="ko",
        verbose=True,
    )
    
    return P.mean().item(), R.mean().item(), F1.mean().item()

# 사용 예시
references = [
    "푸른 바다 위로 하얀 파도가 부서진다",
    "높은 산 위에 구름이 걸려 있다",
]

hypotheses = [
    "파란 바다에 하얀 파도가 친다",
    "산꼭대기에 구름이 피어오른다",
]

P, R, F1 = calculate_bertscore_korean(references, hypotheses)
print(f"Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")
```

### 4.4 BERTScore 장점

| 특징 | 설명 |
|------|------|
| **의미적 매칭** | 동의어, 의역 자동 인식 |
| **문맥 고려** | 문맥에 따른 단어 의미 반영 |
| **다국어 지원** | 다국어 BERT로 한국어 지원 |
| **높은 상관도** | 인간 평가와 높은 상관 |

---

## 5. 보조 지표

### 5.1 BLEU (참고용)

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_korean(reference, hypothesis):
    """
    한국어 BLEU 계산
    
    주의: 한국어에서 BLEU는 신뢰도가 낮음
    - 교착어 특성으로 n-gram 매칭 어려움
    - 어순 변화에 민감
    """
    ref_tokens = tokenize_korean(reference)
    hyp_tokens = tokenize_korean(hypothesis)
    
    # Smoothing for short sentences
    smoothing = SmoothingFunction().method1
    
    bleu_scores = {}
    for n in [1, 2, 3, 4]:
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        bleu_scores[f'BLEU-{n}'] = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=weights,
            smoothing_function=smoothing
        )
    
    return bleu_scores

# BLEU 한계:
# - 짧은 문장에서 과대평가
# - 동의어 미인식
# - 한국어 어순 변화에 취약
# → 참고용으로만 사용
```

### 5.2 ROUGE-L (참고용)

```python
from rouge_score import rouge_scorer

def calculate_rouge_korean(reference, hypothesis):
    """
    한국어 ROUGE-L 계산
    
    ROUGE-L: Longest Common Subsequence 기반
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    # 한국어 토큰화 후 공백으로 연결
    ref_text = ' '.join(tokenize_korean(reference))
    hyp_text = ' '.join(tokenize_korean(hypothesis))
    
    scores = scorer.score(ref_text, hyp_text)
    
    return {
        'precision': scores['rougeL'].precision,
        'recall': scores['rougeL'].recall,
        'f1': scores['rougeL'].fmeasure,
    }

# ROUGE-L 장점:
# - 어순 변화에 BLEU보다 강건
# - 긴 문장에서 유용

# ROUGE-L 한계:
# - 동의어 미인식
# - 의미적 유사도 미반영
```

### 5.3 CLIPScore (멀티모달)

```python
import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPScoreCalculator:
    """
    CLIPScore 계산기
    
    이미지-텍스트 정렬도 측정
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    @torch.no_grad()
    def calculate(self, images, texts):
        """
        CLIPScore 계산
        
        Parameters:
        - images: PIL Image 리스트 또는 텐서
        - texts: 캡션 리스트
        
        Returns:
        - CLIPScore (0-100)
        """
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        outputs = self.model(**inputs)
        
        # 이미지-텍스트 유사도
        logits_per_image = outputs.logits_per_image
        similarity = logits_per_image.diagonal()
        
        # CLIPScore (0-100 스케일)
        clip_score = similarity.mean().item() * 2.5  # 스케일 조정
        
        return max(0, min(100, clip_score))

# 사용 예시
calculator = CLIPScoreCalculator()

# 비디오 프레임과 캡션
frames = [frame1, frame2, frame3]  # PIL Images
captions = ["푸른 바다 위로 파도가 친다"]

score = calculator.calculate(frames, captions)
print(f"CLIPScore: {score:.2f}")

# CLIPScore 장점:
# - 이미지-텍스트 직접 비교
# - Reference 필요 없음

# CLIPScore 한계:
# - 한국어 지원 제한
# - 세부 묘사 평가 어려움
```

---

## 6. 종합 평가 프레임워크

### 6.1 평가 파이프라인

```python
class KoreanCaptionEvaluator:
    """
    한국어 영상 캡셔닝 종합 평가기
    """
    
    def __init__(self):
        self.okt = Okt()
        # BERTScore 모델 로드
        self.bertscore_model = "klue/bert-base"
    
    def tokenize(self, text):
        """한국어 토큰화"""
        return self.okt.morphs(text)
    
    def evaluate_single(self, reference, hypothesis):
        """
        단일 샘플 평가
        
        Returns:
        - dict with all metrics
        """
        results = {}
        
        # 1. METEOR (주요 지표)
        results['meteor'] = calculate_meteor_korean(reference, hypothesis)
        
        # 2. CIDEr
        refs = {"0": [reference]}
        hyps = {"0": [hypothesis]}
        results['cider'] = calculate_cider(refs, hyps)
        
        # 3. BERTScore
        P, R, F1 = calculate_bertscore_korean([reference], [hypothesis])
        results['bertscore'] = F1
        
        # 4. BLEU (참고용)
        bleu = calculate_bleu_korean(reference, hypothesis)
        results['bleu-4'] = bleu['BLEU-4']
        
        # 5. ROUGE-L (참고용)
        rouge = calculate_rouge_korean(reference, hypothesis)
        results['rouge-l'] = rouge['f1']
        
        return results
    
    def evaluate_batch(self, references, hypotheses):
        """
        배치 평가
        
        Parameters:
        - references: 정답 캡션 리스트
        - hypotheses: 생성된 캡션 리스트
        
        Returns:
        - dict with averaged metrics
        """
        all_results = []
        
        for ref, hyp in zip(references, hypotheses):
            results = self.evaluate_single(ref, hyp)
            all_results.append(results)
        
        # 평균 계산
        avg_results = {}
        for key in all_results[0].keys():
            avg_results[key] = sum(r[key] for r in all_results) / len(all_results)
        
        return avg_results
    
    def generate_report(self, results, output_path=None):
        """
        평가 리포트 생성
        """
        report = f"""
========================================
한국어 영상 캡셔닝 평가 리포트
========================================

주요 지표:
  METEOR:     {results['meteor']:.4f}
  CIDEr:      {results['cider']:.4f}
  BERTScore:  {results['bertscore']:.4f}

참고 지표:
  BLEU-4:     {results['bleu-4']:.4f}
  ROUGE-L:    {results['rouge-l']:.4f}

========================================
베이스라인 대비:
  METEOR 베이스라인: 0.3052
  현재 METEOR:       {results['meteor']:.4f}
  향상률:            {((results['meteor'] - 0.3052) / 0.3052 * 100):.2f}%
========================================
"""
        print(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
```

### 6.2 평가 실행 예시

```python
# 평가 실행

evaluator = KoreanCaptionEvaluator()

# 테스트 데이터
test_references = [
    "푸른 바다 위로 하얀 파도가 부서지며 해변으로 밀려온다",
    "높은 산 위에 구름이 걸려 있고 나무들이 울창하다",
    "도시의 거리에 사람들이 걸어다니고 있다",
]

test_hypotheses = [
    "파란 바다에 하얀 파도가 치며 해변으로 다가온다",
    "산꼭대기에 구름이 피어오르고 숲이 우거져 있다",
    "도심 거리를 사람들이 지나가고 있다",
]

# 평가 실행
results = evaluator.evaluate_batch(test_references, test_hypotheses)

# 리포트 생성
evaluator.generate_report(results, output_path="evaluation_report.txt")
```

---

## 7. 목표 성능 설정

### 7.1 베이스라인 대비 목표

| 지표 | 베이스라인 | 목표 1 (10% ↑) | 목표 2 (20% ↑) | 최종 목표 |
|------|----------|----------------|----------------|----------|
| **METEOR** | 0.3052 | 0.336 | 0.366 | 0.40+ |
| CIDEr | - | 0.80 | 1.00 | 1.20+ |
| BERTScore | - | 0.75 | 0.80 | 0.85+ |

### 7.2 성능 향상 전략

```
성능 향상 로드맵:

Phase 1: 기본 학습 (METEOR 0.33 목표)
├── LLaVA-NeXT-Video-7B + LoRA
├── 기본 학습 설정
└── 데이터셋 전처리 최적화

Phase 2: 모델 개선 (METEOR 0.36 목표)
├── Qwen3-8B로 LLM 교체
├── 2-Stage Training 적용
└── 적응적 프레임 샘플링

Phase 3: 최적화 (METEOR 0.40+ 목표)
├── SigLIP 2 Vision Encoder
├── LoRA rank 증가 (32 → 64)
├── 데이터 증강
└── 앙상블 (선택적)
```

---

## 8. 평가 실행 가이드

### 8.1 평가 스크립트

```python
# evaluate.py

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="한국어 캡셔닝 평가")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--references", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()
    
    # 데이터 로드
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(args.references, 'r', encoding='utf-8') as f:
        references = json.load(f)
    
    # 평가 실행
    evaluator = KoreanCaptionEvaluator()
    
    refs = [references[k] for k in sorted(references.keys())]
    hyps = [predictions[k] for k in sorted(predictions.keys())]
    
    results = evaluator.evaluate_batch(refs, hyps)
    
    # 결과 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 리포트 출력
    evaluator.generate_report(results)

if __name__ == "__main__":
    main()
```

### 8.2 실행 명령

```bash
# 평가 실행
python evaluate.py \
    --predictions outputs/predictions.json \
    --references data/references.json \
    --output results/evaluation.json
```

---

## 9. 참고 자료

### 논문
- [METEOR: An Automatic Metric for MT Evaluation](https://aclanthology.org/W05-0909.pdf)
- [CIDEr: Consensus-based Image Description Evaluation](https://arxiv.org/abs/1411.5726)
- [BERTScore: Evaluating Text Generation](https://arxiv.org/abs/1904.09675)
- [CLIPScore: A Reference-free Evaluation Metric](https://arxiv.org/abs/2104.08718)

### 코드 레포지토리
- [pycocoevalcap](https://github.com/tylin/coco-caption)
- [bert_score](https://github.com/Tiiiger/bert_score)
- [nltk.translate](https://www.nltk.org/api/nltk.translate.html)
