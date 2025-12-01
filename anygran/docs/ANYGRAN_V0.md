# AnyGran: Any-Granularity Scene Graph Representation

**Version**: v0 (Initial Prototype)
**Date**: December 2024
**Status**: Experimental

---

## Overview

**AnyGran** is a framework for building hierarchical, multi-granular representations on top of raw scene graphs. It uses Large Language Models (LLMs) to generate abstract layers that provide semantic context at multiple levels of granularity.

### Core Idea

Transform flat scene graph sequences into a hierarchical structure where:
- **Higher levels** capture abstract goals and event phases
- **Lower levels** preserve detailed action-object relationships
- **All levels** are available to downstream tasks like Question Answering

### Motivation

Raw scene graphs (triplet sequences) lack high-level semantic structure. By adding abstraction layers, we hypothesize that:
1. Models can better understand the overall context of an activity
2. Temporal reasoning improves with explicit phase boundaries
3. Question answering benefits from multi-level context

---

## Hierarchy Structure

AnyGran v0 implements a 4-level hierarchy:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANYGRAN HIERARCHY (v0)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 3: GOAL LAYER                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  "Clean the car interior using a mop and cloth, then store tools"   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                              LLM Abstraction                                │
│                                     ▼                                       │
│  LEVEL 2: SUB-EVENT LAYER                                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ Retrieve mop │→│ Finish clean │→│ Access cloth │→│ Re-grab mop  │       │
│  │ & clean car  │ │ & place items│ │ from storage │ │ from wall    │       │
│  │ [0,1]        │ │ [2,3,4,5]    │ │ [6,7,8,9]    │ │ [10]         │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
│         │                │                │                │                │
│                              LLM Grouping                                   │
│                                     ▼                                       │
│  LEVEL 1: ACTION LAYER (Original Scene Graphs)                              │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐    │
│  │ G₀ │→│ G₁ │→│ G₂ │→│ G₃ │→│ G₄ │→│ G₅ │→│ G₆ │→│ G₇ │→│ G₈ │→│G₁₀│    │
│  │pick│ │sweep│ │close│ │place│ │open│ │put │ │move│ │open│ │pick│ │pick│    │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘    │
│     │      │      │      │      │      │      │      │      │      │       │
│                                     ▼                                       │
│  LEVEL 0: TRIPLET LAYER (Raw Data)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ [person, verb, pick-up]  [pick-up, dobj, mop-stick]                  │   │
│  │ [pick-up, with, hand1]   [pick-up, with, hand2]                      │   │
│  │ [mop-stick, from, floor]                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Level Descriptions

| Level | Name | Description | Source |
|-------|------|-------------|--------|
| 3 | **Goal Layer** | Overall activity description | LLM-generated |
| 2 | **Sub-Event Layer** | Grouped phases with action indices | LLM-generated |
| 1 | **Action Layer** | Individual scene graphs (Gᵢ) | Original data |
| 0 | **Triplet Layer** | Raw [node, edge, node] triplets | Original data |

---

## Methodology

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ANYGRAN METHODOLOGY (v0)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Raw SGQA Data                                                       │
│  ┌─────────────────────┐                                                    │
│  │ context_graphs:     │                                                    │
│  │   [[triplets], ...]  │                                                    │
│  │ question: "..."     │                                                    │
│  │ answer: "..."       │                                                    │
│  └─────────────────────┘                                                    │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 1: EVENT GENERATION                         │   │
│  │  ┌─────────────┐        ┌──────────────────┐                         │   │
│  │  │ Scene Graphs │ ────▶ │   GPT-5-mini     │ ────▶ Event Hierarchy   │   │
│  │  │ [G₀...Gₙ]    │        │ (EventGenerator) │       {goal, sub_events}│   │
│  │  └─────────────┘        └──────────────────┘                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 2: HIERARCHICAL QA                          │   │
│  │                                                                       │   │
│  │  PROMPT STRUCTURE:                                                    │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │ ## Overall Goal                                                │   │   │
│  │  │ "Clean the car interior..."                                    │   │   │
│  │  │                                                                │   │   │
│  │  │ ## Sub-Events                                                  │   │   │
│  │  │ 1. Retrieve mop and clean car [0,1]                           │   │   │
│  │  │ 2. Finish cleaning... [2,3,4,5]                               │   │   │
│  │  │                                                                │   │   │
│  │  │ ## Detailed Scene Graphs                                       │   │   │
│  │  │ [original triplets...]                                         │   │   │
│  │  │                                                                │   │   │
│  │  │ ## Question                                                    │   │   │
│  │  │ "What object was picked up before sweeping?"                   │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  │            │                                                          │   │
│  │            ▼                                                          │   │
│  │  ┌─────────────┐        ┌──────────────────┐                         │   │
│  │  │ Hier. Prompt │ ────▶ │   GPT-5-mini     │ ────▶ [answer]          │   │
│  │  └─────────────┘        │(HierarchicalSGQA)│                         │   │
│  │                         └──────────────────┘                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ▼                                                                │
│  OUTPUT: Exact Match Evaluation                                             │
│  ┌─────────────────────┐                                                    │
│  │ prediction vs       │                                                    │
│  │ ground_truth        │                                                    │
│  │ (case-insensitive)  │                                                    │
│  └─────────────────────┘                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Event Generation

The `EventGenerator` class uses an LLM to analyze scene graph sequences and produce:

1. **Overall Goal**: A single sentence describing the main activity
2. **Sub-Events**: 2-5 logical phases with:
   - `name`: Phase identifier
   - `description`: Brief explanation
   - `action_indices`: Which actions belong to this phase

**Example Output:**
```json
{
  "overall_goal": "Clean the car interior using a mop and cloth, then store tools",
  "sub_events": [
    {"name": "Retrieve mop and clean car", "description": "Pick up mop and sweep", "action_indices": [0, 1]},
    {"name": "Finish cleaning and place items", "description": "Close car, set mop down", "action_indices": [2, 3, 4, 5]},
    {"name": "Access storage to get cloth", "description": "Open cabinet, get cloth", "action_indices": [6, 7, 8, 9]},
    {"name": "Re-grab mop from wall", "description": "Final tool retrieval", "action_indices": [10]}
  ]
}
```

### Phase 2: Hierarchical QA

The `HierarchicalSGQAEvaluator` constructs prompts with multi-level context:

```
## Overall Goal (What is happening)
{overall_goal}

## Sub-Events (Phases of the activity)
1. Phase name: description (Actions: [indices])
...

## Detailed Scene Graphs (Action-Level)
{original_scene_graphs}

## Question
{question}
```

---

## Implementation

### File Structure

```
anygran/
├── docs/
│   └── ANYGRAN_V0.md              # This documentation
├── prompts/
│   ├── event_generation.txt        # Prompt for Level 2-3 generation
│   └── hierarchical_sgqa.txt       # Prompt for hierarchical QA
├── cache/
│   └── events_*.json               # Cached event hierarchies
├── results/
│   ├── baseline_*.json             # Baseline evaluation results
│   └── hierarchical_*.json         # Hierarchical evaluation results
├── __init__.py
├── event_generator.py              # EventGenerator class
├── hierarchical_sgqa.py            # HierarchicalSGQAEvaluator class
└── run_hierarchical_sgqa.py        # Main CLI runner
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `EventGenerator` | `event_generator.py` | Generates hierarchical event summaries |
| `HierarchicalSGQAEvaluator` | `hierarchical_sgqa.py` | QA with multi-level context |
| `BaselineSGQAEvaluator` | `hierarchical_sgqa.py` | Standard QA for comparison |
| `run_hierarchical_sgqa.py` | Main CLI | Orchestrates evaluation pipeline |

### Usage

```bash
# Test on first 5 samples
python anygran/run_hierarchical_sgqa.py --limit 5

# Run full evaluation
python anygran/run_hierarchical_sgqa.py

# Generate events only (no QA)
python anygran/run_hierarchical_sgqa.py --generate-only --limit 5

# Use cached events
python anygran/run_hierarchical_sgqa.py --cache-path anygran/cache/events.json
```

---

## Results (v0)

### Initial Experiment: First 5 Samples (25 QA Pairs)

| Metric | Baseline | Hierarchical | Δ |
|--------|----------|--------------|---|
| **Exact Match** | 80.0% | 76.0% | -4.0% |
| Correct | 20/25 | 19/25 | -1 |

### Per-Question Analysis

| Status | Question (truncated) | Ground Truth | Baseline | Hierarchical |
|--------|---------------------|--------------|----------|--------------|
| IMPROVED | "What object was first manipulated before water flow..." | tap | pot | **tap** |
| REGRESSED | "Which location did the person interact with after..." | wall | **wall** | cabinet |
| REGRESSED | "What tool was used immediately before the first bolt..." | spanner | **spanner** | wrench |

### Observations

1. **Small sample size**: 5 samples is too small for statistical significance
2. **Mixed results**: Hierarchical helps some questions, hurts others
3. **Semantic confusion**: "wrench" vs "spanner" suggests abstraction may introduce ambiguity
4. **Temporal reasoning**: Some temporal questions improved with phase context

---

## Limitations (v0)

1. **Single LLM for all stages**: Same model generates events and answers questions
2. **No iterative refinement**: Event hierarchy generated in one pass
3. **Fixed granularity**: 2-5 sub-events regardless of sequence complexity
4. **No triplet-level abstraction**: Level 0 unchanged from original data

---

## Future Directions

### v1 Improvements

- [ ] Larger scale evaluation (all 99 samples)
- [ ] Prompt engineering for better event boundaries
- [ ] Different models for event generation vs QA
- [ ] Dynamic number of hierarchy levels

### v2 Possibilities

- [ ] Triplet-level abstraction (Level 0 → Level 0.5)
- [ ] Graph-based event representations
- [ ] Multi-hop reasoning across hierarchy levels
- [ ] Fine-tuning on event generation task

---

## References

- **TSG-Bench**: [arXiv:2505.19510](https://arxiv.org/abs/2505.19510)
- **Base Repository**: `/home/jtu9/sgg/tsg-bench/`

---

## Changelog

### v0 (December 2024)
- Initial prototype implementation
- 4-level hierarchy (Goal → Sub-Events → Actions → Triplets)
- GPT-5-mini for event generation and QA
- First 5 samples evaluation: 76% EM (hierarchical) vs 80% EM (baseline)
