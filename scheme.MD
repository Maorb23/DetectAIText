# Detect-AI-Text Scoring Scheme

## Overview ✅
This document describes a pragmatic, modular scheme to aggregate the detectors in this repo into a single, **calibrated**, and **explainable** final score for "AI-generated" vs "human-written" text. It summarizes existing components observed in the codebase and recommends additional layers, preprocessing improvements, aggregation math, and evaluation guidance.

---

## 1) What I found in the repo (short) 🔎
- Heuristics (`modules/heuristics.py`) — rule-based signals (e.g., punctuation, repetitions).
- Logit-based analysis (`modules/logits.py`, `scripts/score.py`) — likely using LM logits / ratios.
- Binocular ensemble (`modules/binocular.py`) — combining signals or two-model approach.
- Fast detector wrapper (`modules/fast_detect_gpt.py`) and tokenizer utilities (`utils/utils_tokenizer.py`).
- Trainable classifiers and features (`scripts/features.py`, `scripts/train_classifier.py`, `scripts/models.py`).

These are good building blocks: language model signals, heuristics, and downstream classifiers.

---

## 2) Suggested additional layers (high value) ➕
- **Surprisal / Perplexity** features (normalized by length and token distribution).
- **Log-likelihood ratio** between target model(s) and a background human LM (robust signal).
- **Stylometry & syntactic features** (POS distribution, parse-tree depth, function words, POS n-grams).
- **Semantic coherence / embedding drift** (sentence-level embedding distances, topical divergence).
- **Watermark detection hook** (if enterprise uses watermarking-enabled LM outputs).
- **Adversarial robustness checks** (paraphrase invariance: score on paraphrases / back-translation).
- **Metadata features** (source, timestamp, author behavior when available).
- **Contextual consistency tests** (e.g., named-entity contradictions across the text).

---

## 3) Better preprocessing ideas ✨
- **Canonicalize text**: unicode normalization (NFKC), consistent whitespace, normalize quotes/dashes.
- **Segment & window**: split long documents into overlapping windows (e.g., 200–512 tokens) and score per-window.
- **Remove/mark non-linguistic content**: code blocks, tables, boilerplate, citations, and URLs (or treat them separately).
- **Tokenization parity**: use the *same tokenizer(s)* as the LMs used for scoring (avoid mismatch artifacts).
- **Case & contraction handling**: keep original case but optionally test lowercase-collapsed variant as a feature.
- **Length-binning**: bucket texts by length and calibrate thresholds per bucket (detector behavior often length-dependent).
- **Noise-robust handling**: strip repeated whitespace, normalize newlines, and optionally remove serialization artifacts.

---

## 4) Aggregation pipeline (step-by-step) 🔧
1. **Preprocess** the raw text (canonicalize, segment into windows, tokenization).
2. **Compute base detectors** (per-window and per-document):
   - LM-based scores (perplexity, avg log-prob, logit-ratio)
   - Heuristic detectors (repetition, punctuation, stopword patterns)
   - Stylistic features (POS, syntax metrics)
   - Semantic features (embedding variance / coherence)
   - Any pretrained fast detectors that output a probability
3. **Normalize** each feature to a common scale (e.g., min-max / robust scaler using training distribution) -> s_i ∈ [0,1].
4. **Aggregate per-window -> per-document** using robust statistics (mean, 90th percentile, max) and keep multiple aggregated stats as features.
5. **Stack / Meta-model**: feed aggregated features and base detectors' probabilities into a small classifier (logistic / lightGBM / simple NN) trained to output a calibrated probability p_meta.
6. **Calibration**: calibrate p_meta (Platt scaling or isotonic regression) per domain / length bin for stable probabilities.
7. **Final score**: use p_final = p_meta (or a weighted blend of p_meta and a simple normalized-score fallback if p_meta lacks confidence).

---

## 5) Scoring formula (example) 🧮
- Let s_j be normalized component scores, w_j pre-set or learned weights.
- Fallback normalized ensemble score: S_norm = sum_j w_j * s_j / sum_j w_j
- Meta-model probability p_meta = MetaClassifier(aggregated features)
- Final score recipe (blended):

    p_final = alpha * p_meta + (1 - alpha) * S_norm

- Choose alpha near 1.0 if you have a well-trained meta on representative data. Use lower alpha when distribution shift is expected.
- Thresholding: pick threshold t by desired operating point (e.g., maximize F1 or bound false-positive rate). Report both p_final and a binary label.

---

## 6) Explainability & output format 📣
Return structured output per document containing:
- `p_final` (float 0..1), `label` (0/1), `confidence` (calibrated), and `explanation` (component contributions ordered by impact)
- Per-window breakdown with top contributing windows and component values
- Flags: `low_confidence`, `short_text_warning`, `adversarial_flag`

Example JSON snippet:

```json
{
  "p_final": 0.84,
  "label": 1,
  "confidence": 0.78,
  "components": {"logit_ratio":0.9, "heuristics":0.6, "stylometry":0.2},
  "windows": [{"start":0,"end":200,"p":0.9}, ...]
}
```

---

## 7) Evaluation & monitoring 📊
- Use held-out labeled sets stratified by domain and length; report ROC AUC and PR AUC. Also report FPR@10%TPR and FPR@1%TPR for high-assurance settings.
- Stress tests: paraphrase robustness, adversarial prompts, few-shot priming, and watermark on/off.
- Monitor calibration drift in production and re-calibrate on recent samples periodically.
- Keep per-domain thresholds and domain-specific meta-models if required.

---

## 8) Operational notes 🔩
- Cache expensive LM-based scoring (tokenization & logprobs) using content hash.
- Batch LM scoring to use GPU efficiently.
- Keep deterministic preprocessing for reproducibility.

---

## 9) Short roadmap / experiments to run 📋
- Add LLR logit-ratio feature between target-LM and baseline LM.
- Train a LightGBM stacker on a diverse labeled corpus and test calibration & domain-transfer.
- Implement per-window percentile aggregation and track whether max/90th helps recall on long docs.

---

> If you'd like, I can (a) implement per-window PASSTHROUGH scoring in `scripts/score.py`, (b) add the LLR feature in `modules/logits.py`, or (c) create a notebook that demonstrates training the meta-classifier and calibration. Tell me which to prioritize.

---

*End of scheme.MD*
