# Error Analysis Summary

## Overview

All 9 unique error cases across Baseline, Hierarchical v0, and Unified v1 approaches.

| # | Question | GT | Baseline | Hier v0 | Unified v1 | Category |
|---|----------|----|---------:|--------:|-----------:|----------|
| 1 | Which location after using cloth? | wall | cabinet ✗ | cabinet ✗ | cabinet ✗ | Temporal + Location |
| 2 | First object after placing laptop? | charger | laptop ✗ | laptop ✗ | laptop ✗ | Temporal |
| 3 | Handled before cooking vessel? | onion | pot ✗ | pot ✗ | pot ✗ | Temporal |
| 4 | Between two water actions? | pot | onion ✗ | onion ✗ | onion ✗ | Temporal + Object |
| 5 | Both hands after tools stored? | wheel | lid ✗ | wheel ✓ | wheel ✓ | Multi-Step |
| 6 | Last tool before final positioning? | screw | screwdriver ✗ | screwdriver ✗ | screwdriver ✗ | Disambiguation |
| 7 | Both hands first manipulation? | metal-board | paper ✗ | paper+cardboard ✗ | paper ✗ | Disambiguation |
| 8 | After first clamp removed? | wood | wood ✓ | clamp ✗ | wood ✓ | Temporal |
| 9 | Held in hand2 during drill? | wood | wood ✓ | wood ✓ | wood+piece ✗ | Concurrent State |

## Error Categories

### Category A: Temporal Ordering (Errors 1, 2, 3, 6, 8)
Questions requiring understanding of action sequence (before/after relationships).
- Common failure: Incorrect identification of temporal boundaries
- All approaches struggle with these

### Category B: Object/Location Confusion (Errors 1, 4)
Questions where model confuses similar concepts or object types.
- Error 1: Predicts "cabinet" (object) instead of "wall" (location)
- Error 4: Swaps "pot" and "onion"

### Category C: Similar Object Disambiguation (Errors 6, 7)
Questions requiring distinction between similar objects.
- Error 6: "screw" vs "screwdriver" - very similar names
- Error 7: "metal-board" vs "paper" - both flat objects

### Category D: Complex Reasoning (Error 5)
Requires multi-step inference.
- "after all tools were stored" requires tracking tool storage completion
- Hierarchical approaches help here (both v0 and v1 correct)

### Category E: Concurrent State Tracking (Error 9)
Requires tracking simultaneous states.
- What is held in hand2 WHILE drill operates
- Unified v1 over-specifies ("wood and wood-piece")

## Key Insights

1. **Temporal reasoning is the biggest challenge**: 5 of 9 errors involve temporal ordering
2. **Hierarchical context helps some cases**: Error 5 solved by v0 and v1 (not baseline)
3. **v1 regression on Error 9**: Over-specification issue with unified format
4. **v0 regression on Error 8**: Separated format caused confusion

## Recommendations

1. **Add explicit temporal markers** in prompts (e.g., "Action 0 happens BEFORE Action 1")
2. **Clarify object vs location** in question understanding
3. **Avoid over-specification** in unified format (simpler output format)
4. **Consider chain-of-thought** for multi-step reasoning
