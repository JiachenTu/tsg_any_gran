# AnyGran: Multi-Granular Hierarchical Scene Graphs for SGQA

## Overview

AnyGran enhances Scene Graph Question Answering (SGQA) by adding hierarchical abstraction layers on top of raw scene graph triplets. The hypothesis is that providing high-level semantic context (overall goals, sub-events) helps models better understand the temporal and causal relationships needed to answer questions accurately.

### Hierarchy Levels

```
Level 3: Overall Goal     "Clean the car interior floor..."
         │
Level 2: Sub-Events       [Phase 1: Prepare] [Phase 2: Stow] [Phase 3: Retrieve]
         │
Level 1: Actions          A0: pick-up, A1: sweep, A2: close, ...
         │
Level 0: Triplets         [person, verb, pick-up] [pick-up, dobj, mop-stick] ...
```

---

## Version History

### v0 (Separated Hierarchy)
- **Format**: Overall goal, sub-events, and scene graphs in separate sections
- **Sub-events format**: Reference action indices that must be looked up
- **Event generation**: GPT-5-mini
- **Prompt structure**:
  ```
  ## Overall Goal
  {overall_goal}

  ## Sub-Events
  1. Phase name: description (Actions: [0, 1, 2])
  2. Phase name: description (Actions: [3, 4, 5])

  ## Scene Graphs
  {raw_scene_graphs}

  ## Question
  {question}
  ```

### v1 (Unified/Interleaved Hierarchy)
- **Format**: Sub-events and their corresponding actions presented together
- **No need to cross-reference action indices**
- **Event generation**: GPT-5 (same events from v0 cache)
- **Prompt structure**:
  ```
  ## Overall Goal
  {overall_goal}

  ## Activity Timeline

  ### Phase 1: Prepare and start sweeping
  Pick up the mop and sweep the car floor...

  Actions in this phase:
  - Action 0 (pick-up): [person, verb, pick-up] [pick-up, dobj, mop-stick]...
  - Action 1 (sweep): [person, verb, sweep] [sweep, dobj, floor]...

  ### Phase 2: Stow mop and place cloth
  ...

  ## Question
  {question}
  ```

---

## Results Comparison (10 Samples, 50 Questions)

| Approach | Correct | EM (%) | Notes |
|----------|---------|--------|-------|
| Baseline (GPT-5) | 43 | 86.0% | No hierarchy, raw scene graphs only |
| Hierarchical v0 (GPT-5) | 43 | 86.0% | Separated format |
| Unified v1 (GPT-5) | 43 | 86.0% | Interleaved format |

All three approaches achieved identical accuracy on the 10-sample test set. However, they differ in **which questions they answer correctly**.

---

## Per-Question Error Analysis

### Errors Across All Three Approaches (Common Hard Cases)

| Question | Ground Truth | All Predictions |
|----------|--------------|-----------------|
| "Which location did the person interact with after using the cloth?" | wall | cabinet |
| "What was the first object the person interacted with after placing the laptop?" | charger | laptop |
| "What object was handled immediately before being transferred to the cooking vessel?" | onion | pot |
| "Which object was interacted with between two water-related actions?" | pot | onion |
| "What was the last tool picked up before the final positioning action?" | screw | screwdriver |
| "Which object required both hands for its first manipulation after being picked up?" | metal-board | paper |

### Version-Specific Differences

| Question | Baseline | Hier v0 | Unified v1 |
|----------|----------|---------|------------|
| "Which object required both hands to manipulate after all tools were stored?" | lid ✗ | wheel ✓ | wheel ✓ |
| "What tool was picked up immediately after the first clamp was removed?" | wood ✓ | clamp ✗ | wood ✓ |
| "Which object was held in hand2 while the drill was being operated?" | wood ✓ | wood ✓ | wood and wood-piece ✗ |

### Key Observations

1. **Persistent Hard Cases**: Some questions are consistently wrong across all approaches, suggesting they require reasoning that current prompting strategies don't address:
   - Temporal ordering confusion (before/after)
   - Object vs location confusion (cloth → wall vs cabinet)
   - Similar object disambiguation (screw vs screwdriver)

2. **v0 Regression**: Hierarchical v0 caused 1 regression on the clamp→wood question, possibly due to the separated format requiring index lookups.

3. **v1 New Error**: Unified v1 introduced a new error on the drill question by over-specifying ("wood and wood-piece" instead of "wood").

4. **Hierarchical Benefits**: Both v0 and v1 correctly answered the "wheel" question that baseline failed, showing hierarchy can help with "after all tools were stored" type reasoning.

---

## Key Changes in v1

1. **Unified prompt structure**: Actions are embedded directly within their sub-event phases, eliminating the need to cross-reference action indices.

2. **Clearer temporal relationships**: Each phase explicitly shows its actions, making phase boundaries and action groupings immediately visible.

3. **Preserved semantic context**: Overall goal and phase descriptions remain, providing high-level understanding.

---

## Files Structure

```
anygran/
├── __init__.py
├── event_generator.py          # Generates Level 2-3 from Level 0-1
├── hierarchical_sgqa.py        # Contains all evaluator classes:
│   ├── HierarchicalSGQAEvaluator  # v0: separated format
│   ├── UnifiedHierarchicalEvaluator  # v1: interleaved format
│   └── BaselineSGQAEvaluator      # No hierarchy
├── run_hierarchical_sgqa.py    # Main runner with --unified flag
├── cache/
│   └── events_limitall.json    # Cached event hierarchies
├── prompts/
│   ├── event_generation.txt    # Prompt for generating events
│   ├── hierarchical_sgqa.txt   # v0 prompt template
│   └── unified_hierarchical.txt # v1 prompt template
├── results/
│   ├── baseline_gpt5_limit10_*.json
│   ├── hierarchical_gpt5_limit10_*.json
│   └── unified_gpt5_limit10_*.json
├── inspect/
│   └── prompt_inspector.py     # Tool for debugging prompts
└── docs/
    ├── ANYGRAN.md              # This file
    ├── HIERARCHY_VISUALIZATION.md
    └── CODE_PIPELINE.md
```

---

## Usage

```bash
# Run baseline + hierarchical v0 comparison
python anygran/run_hierarchical_sgqa.py --limit 10 --model gpt5

# Run with unified v1 format
python anygran/run_hierarchical_sgqa.py --limit 10 --model gpt5 --unified

# Skip baseline, only run hierarchical
python anygran/run_hierarchical_sgqa.py --limit 10 --model gpt5 --skip-baseline

# Use cached events
python anygran/run_hierarchical_sgqa.py --cache-path anygran/cache/events_limitall.json

# Inspect prompts without model calls
python anygran/inspect/prompt_inspector.py --limit 10 --no-call
```

---

## Future Directions

1. **Better temporal reasoning**: Current errors suggest need for explicit temporal markers or chain-of-thought prompting.

2. **Object disambiguation**: Add context about object properties to distinguish similar objects (screw vs screwdriver).

3. **Location vs object clarification**: Prompt engineering to help model distinguish between "where" and "what" in action sequences.

4. **Larger scale evaluation**: Run on full 100-sample dataset to get statistically significant comparisons.
