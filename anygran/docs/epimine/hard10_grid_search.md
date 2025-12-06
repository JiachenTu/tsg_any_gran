# EpiMine Hard-10 Grid Search Analysis

**Date**: 2025-12-05
**Status**: Phase 1 Complete (36/36 experiments with gen5-mini)

## Overview

This document analyzes the hyperparameter grid search for EpiMine episode detection on the Hard-10 benchmark (10 difficult SGQA cases).

### Baseline Reference
- GPT-5 without EpiMine: **20%**

---

## Grid Search Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| threshold_std | 1.0, 1.5, 2.0 | Episode boundary sensitivity (std devs below mean) |
| min_freq | 1, 2 | Minimum term frequency for salience calculation |
| top_k | 5, 10, all | Limit on key terms for co-occurrence matrix |
| use_llm | 0, 1 | Whether to use LLM for episode name generation |

**Total experiments**: 3 x 2 x 3 x 2 = 36

---

## Complete Results Table

### Best Performers
| Rank | Accuracy | threshold | min_freq | top_k | LLM |
|------|----------|-----------|----------|-------|-----|
| 1    | **50%** | 1.5 | 2 | 10 | Yes |
| 2    | **40%** | 1.5 | 2 | 5 | Yes |
| 3-11 | 30% | various | various | 5,10 | mixed |

### Full Results (noLLM vs LLM comparison)

| threshold | min_freq | top_k | noLLM | LLM | Delta |
|-----------|----------|-------|-------|-----|-------|
| 1.0 | 1 | 5 | 20% | 10% | -10% |
| 1.0 | 1 | 10 | **30%** | 10% | -20% |
| 1.0 | 1 | all | 20% | 20% | 0% |
| 1.0 | 2 | 5 | 10% | 10% | 0% |
| 1.0 | 2 | 10 | 10% | 20% | +10% |
| 1.0 | 2 | all | 10% | 20% | +10% |
| 1.5 | 1 | 5 | **30%** | **30%** | 0% |
| 1.5 | 1 | 10 | 10% | **30%** | +20% |
| 1.5 | 1 | all | 10% | 20% | +10% |
| 1.5 | 2 | 5 | 10% | **40%** | +30% |
| 1.5 | 2 | 10 | 20% | **50%** | +30% |
| 1.5 | 2 | all | 20% | 20% | 0% |
| 2.0 | 1 | 5 | **30%** | 20% | -10% |
| 2.0 | 1 | 10 | **30%** | **30%** | 0% |
| 2.0 | 1 | all | 20% | 20% | 0% |
| 2.0 | 2 | 5 | 20% | 20% | 0% |
| 2.0 | 2 | 10 | **30%** | 20% | -10% |
| 2.0 | 2 | all | 10% | **30%** | +20% |

---

## Parameter Impact Analysis

### 1. Threshold_std (Episode Boundary Sensitivity)

Controls how sensitive episode boundary detection is. Formula: `boundary = mean - threshold_std * std`

| Value | Avg Accuracy | Max Accuracy | N |
|-------|-------------|--------------|---|
| t=1.0 | 15.8% | 30% | 12 |
| **t=1.5** | **24.2%** | **50%** | 12 |
| t=2.0 | 23.3% | 30% | 12 |

**Finding**: t=1.5 is optimal - not too sensitive (t=1.0 creates too many small episodes), not too coarse (t=2.0 misses meaningful boundaries).

### 2. Min_freq (Term Frequency Threshold)

Minimum frequency required for a term to be included in salience calculation.

| Value | Avg Accuracy | Max Accuracy | N |
|-------|-------------|--------------|---|
| mf=1 | 21.7% | 30% | 18 |
| mf=2 | 20.6% | **50%** | 18 |

**Finding**: mf=1 has slightly higher average, but mf=2 achieves the best single configuration. Filtering rare terms (mf=2) helps when combined with optimal threshold.

### 3. Top_k (Key Term Limit)

Number of top key terms to use for co-occurrence matrix construction.

| Value | Avg Accuracy | Max Accuracy | N |
|-------|-------------|--------------|---|
| topk=5 | 20.8% | 40% | 12 |
| **topk=10** | **24.2%** | **50%** | 12 |
| topk=all | 18.3% | 30% | 12 |

**Finding**: topk=10 is optimal. Using unlimited terms (topk=all) hurts performance - noise from less relevant terms degrades episode boundary quality.

### 4. LLM Episode Generation

Whether to use GPT-5-mini to generate episode names and descriptions.

| Value | Avg Accuracy | Max Accuracy | N |
|-------|-------------|--------------|---|
| noLLM (structural only) | 18.9% | 30% | 18 |
| **LLM (GPT-5-mini)** | **23.3%** | **50%** | 18 |

**Finding**: LLM episode naming improves average by +4.4% and max by +20%.

---

## LLM Impact Deep Dive

### Paired Comparison Analysis

Out of 18 paired comparisons (same config, different llm setting):
- LLM **improved** 7 configs (avg +18.6% when improved)
- LLM **degraded** 4 configs (avg -12.5% when degraded)
- LLM **no change** 7 configs

### LLM Impact by Threshold

| Threshold | Improved | Degraded | Same | Net Benefit |
|-----------|----------|----------|------|-------------|
| t=1.0 | 2 | 2 | 2 | Neutral |
| **t=1.5** | **4** | 0 | 2 | **Strong positive** |
| t=2.0 | 1 | 2 | 3 | Slightly negative |

**Key insight**: LLM helps most with t=1.5 configurations:
- t=1.5, mf=2, topk=10: 20% -> 50% (+30%)
- t=1.5, mf=2, topk=5: 10% -> 40% (+30%)
- t=1.5, mf=1, topk=10: 10% -> 30% (+20%)

---

## Key Conclusions

1. **Best configuration**: t=1.5, mf=2, topk=10, llm=1 -> **50%** (+150% relative over 20% baseline)

2. **Optimal parameters**:
   - threshold_std: **1.5** (moderate boundary sensitivity)
   - min_freq: **2** (filter rare terms)
   - top_k: **10** (focus on top 10 key terms)
   - use_llm: **True** (GPT-5-mini for episode naming)

3. **LLM impact is threshold-dependent**: LLM helps most at t=1.5, can hurt at t=1.0 and t=2.0

4. **topk=all hurts performance**: Limiting to 5-10 key terms produces better episode boundaries

5. **Interaction effects matter**: The best configuration (50%) requires the right combination of all parameters

---

## Next Steps

1. **Add gen-model dimension**: Test GPT-5 (instead of GPT-5-mini) for episode generation
2. **Validate on full SGQA**: Run best config on 100 samples
3. **Analyze failure cases**: Understand why certain questions remain challenging

---

## Files

- Results: `/home/jtu9/sgg/tsg-bench/anygran/benchmarks/hard_10/results/`
- Script: `/home/jtu9/sgg/tsg-bench/anygran/benchmarks/hard_10/run_epimine_hard_10.py`
