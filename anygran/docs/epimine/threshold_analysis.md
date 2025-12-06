# EpiMine Threshold Analysis

## Overview

This document analyzes the impact of the `threshold_std` hyperparameter on EpiMine episode detection and downstream SGQA question answering performance.

## Parameter Description

**`threshold_std`**: Controls episode boundary detection sensitivity
- **Formula**: `boundary_threshold = mean_cooccurrence - threshold_std * std_cooccurrence`
- **Effect**: Determines when consecutive actions are split into separate episodes
  - **Lower values (0.5)**: More sensitive boundary detection, creating many small episodes
  - **Higher values (2.0)**: Less sensitive, creating fewer large episodes

## Experimental Setup

- **Benchmark**: SGQA Hard-10 (first 10 hard benchmark cases)
- **Model**: GPT-5
- **Background Dataset**: 2546 action graphs from full sgqa.jsonl
- **Baseline**: GPT-5 with flat scene graph (20% accuracy)

## Results

| threshold_std | Total Episodes | Avg Eps/Sample | Accuracy | Correct/Total | vs Baseline |
|--------------|----------------|----------------|----------|---------------|-------------|
| 0.5 | 78 | 8.67 | 0% | 0/10 | -20% |
| 0.75 | 58 | 6.44 | 20% | 2/10 | +0% |
| 1.0 (default) | 41 | 4.56 | 20% | 2/10 | +0% |
| **1.5** | 19 | 2.11 | **30%** | 3/10 | **+10%** |
| **2.0** | 12 | 1.33 | **30%** | 3/10 | **+10%** |

## Key Findings

### 1. Episode Granularity vs Accuracy Tradeoff
- **Too many episodes (threshold < 1.0)**: Information fragmentation leads to worse performance
- **Optimal granularity (threshold >= 1.5)**: Larger, coherent episodes preserve action context

### 2. Quantitative Observations
- Accuracy increases monotonically as threshold increases from 0.5 to 1.5
- Diminishing returns beyond threshold=1.5 (1.5 and 2.0 both achieve 30%)
- Episode count inversely correlates with accuracy

### 3. Episode Statistics by Threshold
```
threshold=0.5:  78 episodes, avg 8.67 per sample (too fragmented)
threshold=0.75: 58 episodes, avg 6.44 per sample (still too many)
threshold=1.0:  41 episodes, avg 4.56 per sample (default, suboptimal)
threshold=1.5:  19 episodes, avg 2.11 per sample (optimal)
threshold=2.0:  12 episodes, avg 1.33 per sample (optimal, fewer episodes)
```

## Recommendations

1. **Default threshold should be 1.5-2.0** instead of 1.0
2. For action sequences with high coherence, prefer threshold=2.0
3. For diverse action sequences, threshold=1.5 provides more granular segmentation while maintaining accuracy

## Next Steps

- Validate findings on full SGQA dataset (all samples)
- Test additional thresholds between 1.5 and 2.5
- Analyze per-question type performance differences

---

*Generated: 2024-12-05*
*Benchmark: SGQA Hard-10*
*Model: GPT-5*
