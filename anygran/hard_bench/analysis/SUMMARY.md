# SGQA Hard Benchmark Analysis

## Overview

This hard benchmark contains 88 cases where at least one model answered incorrectly.

### Difficulty Distribution
| Difficulty | Count | Description |
|------------|-------|-------------|
| Hard | 32 | All 3 models wrong |
| Medium | 13 | 2 models wrong |
| Easy | 43 | 1 model wrong |

### Category Distribution
| Category | Count | Description |
|----------|-------|-------------|
| temporal_ordering | 57 | before/after/first/last/immediately |
| multi_step | 17 | "all X completed", sequences |
| both_hands | 9 | "both hands" manipulation |
| concurrent_state | 2 | "while/during" queries |
| other | 3 | Other reasoning types |

### Model Performance on Hard Cases
| Model | Errors | Solved in Hard Set |
|-------|--------|-------------------|
| EpiMine v0 | 54 | 34 |
| Baseline-mini | 57 | 31 |
| Baseline-GPT5 | 54 | 34 |

## Key Insights

1. **Temporal reasoning dominates errors**: 57 of 88 errors involve temporal ordering
2. **Hardest cases**: 32 questions failed by all models
3. **EpiMine advantage**: Hierarchical context helps with some multi-step cases

## Files

- `temporal_ordering.md` - Analysis of temporal ordering errors
- `multi_step.md` - Analysis of multi-step reasoning errors
- `both_hands.md` - Analysis of both-hands manipulation errors
- `concurrent_state.md` - Analysis of concurrent state errors
- `all_wrong_cases.md` - Deep dive into hardest cases
