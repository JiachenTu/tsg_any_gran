# EpiMine Implementation for Hierarchical SGQA

This document explains how EpiMine is implemented to create hierarchical episode structures for Scene Graph Question Answering (SGQA).

## Overview

EpiMine (ACL 2025) is an unsupervised episode detection framework originally designed for news events. We adapt it for action scene graphs by detecting episode boundaries through **co-occurrence pattern shifts** rather than relying solely on LLM intuition.

**Key Insight**: Episodes (cohesive clusters of actions) cannot be detected using semantic similarity alone—they require analyzing shifts in term co-occurrence patterns.

## Architecture

The implementation consists of three main classes in `epimine_hierarchical_sgqa.py`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Pipeline Flow                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Actions ──► EpiMineActionAnalyzer ──► Candidate Boundaries     │
│                        │                           │                 │
│                        │                           ▼                 │
│              Background Stats          EpiMineEpisodeGenerator      │
│              Key Terms                          │                    │
│              Co-occurrence Matrix               ▼                    │
│                                        Structured Episodes          │
│                                                 │                    │
│                                                 ▼                    │
│                                    EpiMineHierarchicalEvaluator     │
│                                                 │                    │
│                                                 ▼                    │
│                                           QA Evaluation              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Class 1: EpiMineActionAnalyzer

**Location**: `epimine_hierarchical_sgqa.py:184-398`

**Purpose**: Unsupervised episode boundary detection using co-occurrence pattern shifts.

### Key Methods

#### 1. Salience Computation (lines 217-240)

Computes how discriminative a term is for the foreground (current sample) vs. background (all samples).

```python
def compute_salience(self, term: str, foreground: List) -> float:
    """
    Formula: salience = (1 + log(fg_count)²) × log(bg_total / bg_count)
    """
    fg_count = sum(1 for ag in foreground if term in extract_terms_from_action(ag))
    bg_count = self.term_counts_bg.get(term, 0)

    if bg_count > 0:
        return (1 + np.log(fg_count) ** 2) * np.log(self.num_bg / bg_count)
    else:
        return (1 + np.log(fg_count) ** 2) * np.log(self.num_bg)
```

**Why This Matters**: High salience terms are those that appear frequently in the current activity but rarely across the full dataset—these are the discriminative terms that characterize episodes.

#### 2. Co-occurrence Matrix (lines 272-311)

Builds a Jaccard similarity matrix between action graphs based on shared key terms.

```python
def compute_cooccurrence_matrix(self, action_sequence, key_terms) -> np.ndarray:
    # For each pair of actions (i, j):
    # cooccur[i,j] = |shared_terms| / |union_terms|  (Jaccard similarity)
```

**Key Concept**: High co-occurrence between consecutive actions suggests they belong to the same episode. A drop in co-occurrence signals an episode boundary.

#### 3. Boundary Detection (lines 313-367)

Detects episode boundaries using statistical thresholds.

```python
def detect_episode_boundaries(self, action_sequence, threshold_std=1.0) -> List[List[int]]:
    # 1. Compute consecutive co-occurrence scores
    consecutive_scores = [cooccur_matrix[i, i+1] for i in range(n-1)]

    # 2. Compute threshold: mean - 1σ
    threshold = np.mean(consecutive_scores) - threshold_std * np.std(consecutive_scores)

    # 3. Start new episode when score drops below threshold
    if score < threshold:
        episodes.append([i + 1])  # Start new episode
    else:
        episodes[-1].append(i + 1)  # Continue current episode
```

**Boundary Logic**: When co-occurrence between consecutive actions drops significantly (below mean - 1σ), it indicates a phase transition—start a new episode.

---

## Class 2: EpiMineEpisodeGenerator

**Location**: `epimine_hierarchical_sgqa.py:404-725`

**Purpose**: Refine candidate boundaries with LLM-generated names/descriptions, build structured episodes.

### Key Methods

#### 1. Structured Episode Building (lines 450-559)

Converts action indices into rich episode structures.

```python
def _build_structured_episode(self, episode_id, action_indices, action_sequence, ...) -> Dict:
    """
    Extracts from triplets:
    - agent: from [person, verb, X]
    - primary_actions: from [person, verb, X] → X
    - primary_objects: from [action, dobj, X] → X
    - instruments: from [action, with, X] → X
    - source_locations: from [X, from, Y] → Y
    - target_locations: from [action, to/on/in, X] → X
    """
```

**Output Structure**:
```json
{
  "episode_id": 0,
  "name": "Tool Retrieval",
  "description": "Pick up cleaning tools",
  "core_structure": {
    "agent": "person",
    "primary_actions": ["pick-up"],
    "primary_objects": ["mop-stick"],
    "instruments": ["hand1"],
    "source_locations": ["floor"],
    "target_locations": null
  },
  "time": {
    "action_indices": [0, 1],
    "start_index": 0,
    "end_index": 1,
    "duration": 2
  },
  "temporal_context": {
    "position": "beginning",
    "precedes_episodes": [1, 2],
    "follows_episodes": null
  },
  "discriminative_terms": ["pick-up", "mop-stick"],
  "salience_score": 0.85
}
```

#### 2. LLM Refinement (lines 561-638)

Uses GPT to generate human-readable episode names and descriptions.

```python
def generate_episode_hierarchy(self, action_sequence, episode_boundaries, ...):
    # 1. Format candidate episodes for prompt
    # 2. Call LLM to generate names/descriptions
    # 3. Build structured episodes with LLM output
```

**Prompt Template**: `prompts/epimine_episode_generation.txt`

---

## Class 3: EpiMineHierarchicalEvaluator

**Location**: `epimine_hierarchical_sgqa.py:732-938`

**Purpose**: Run QA evaluation using structured episodes as context.

### Key Methods

#### 1. Timeline Formatting (lines 766-834)

Formats structured episodes into a prompt-friendly representation.

```python
def _format_epimine_timeline(self, episodes, action_sequence, overall_goal) -> str:
    """
    Output format:

    ## Overall Goal
    Clean the car interior

    ## Activity Timeline

    ### Episode 0: Tool Retrieval
    Pick up cleaning tools

    **Structure:**
    - Agent: person
    - Actions: pick-up
    - Objects: mop-stick
    - Instruments: hand1
    - From: floor
    - Time: Actions [0] (duration: 1)
    - Key terms: pick-up, mop-stick

    **Actions in this episode:**
    - Action 0 (pick-up): [person, verb, pick-up] [pick-up, dobj, mop-stick]...
    """
```

#### 2. QA Invocation (lines 836-865)

Processes questions using cached episode hierarchies.

```python
def invoke(self, data_id, context_graphs, question) -> str:
    hierarchy = self.episodes_cache.get(data_id, {})
    unified_timeline = self._format_epimine_timeline(
        hierarchy["episodes"], context_graphs, hierarchy["overall_goal"]
    )
    # Format prompt and get answer
```

---

## Key Algorithms Summary

| Algorithm | Location | Formula/Logic |
|-----------|----------|---------------|
| Salience | lines 217-240 | `(1 + log(fg)²) × log(bg_total / bg)` |
| Co-occurrence | lines 299-309 | Jaccard: `shared / union` |
| Boundary | lines 357-365 | New episode if `score < mean - 1σ` |

---

## Triplet Parsing

**Location**: `epimine_hierarchical_sgqa.py:79-178`

Maps scene graph triplets to structured episode fields:

| Triplet Pattern | Extracted Field | Example |
|-----------------|-----------------|---------|
| `[person, verb, X]` | agent, action | `[person, verb, pick-up]` → agent="person", action="pick-up" |
| `[action, dobj, X]` | objects | `[pick-up, dobj, mop]` → objects=["mop"] |
| `[action, with, X]` | instruments | `[pick-up, with, hand1]` → instruments=["hand1"] |
| `[X, from, Y]` | source_locations | `[mop, from, floor]` → source=["floor"] |
| `[action, to/on/in, X]` | target_locations | `[place, on, table]` → target=["table"] |

---

## Entry Point

**File**: `run_epimine_hierarchical_sgqa.py`

```bash
# Generate episodes only
python run_epimine_hierarchical_sgqa.py --limit 10 --generate-only

# Run full evaluation
python run_epimine_hierarchical_sgqa.py --limit 100

# Use custom threshold
python run_epimine_hierarchical_sgqa.py --cooccur-threshold 0.5
```

---

## Comparison: EpiMine vs Pure LLM Approach

| Aspect | Pure LLM (v1) | EpiMine |
|--------|---------------|---------|
| Episode detection | LLM generates from scratch | Statistical co-occurrence analysis → LLM refines |
| Boundary logic | LLM intuition | `mean - 1σ` threshold |
| Background context | None | Full dataset as background corpus |
| Term importance | All equal | Salience-weighted (discriminative) |
| Reproducibility | Variable (LLM stochastic) | Consistent (statistical) |

---

## Results

On SGQA dataset (100 samples, 500 questions):

| Method | Exact Match |
|--------|-------------|
| Baseline | 86.0% |
| EpiMine-Hierarchical | 88.0% |
| **Improvement** | **+2.0%** |

---

## File References

| File | Purpose |
|------|---------|
| `epimine_hierarchical_sgqa.py` | Main implementation |
| `run_epimine_hierarchical_sgqa.py` | CLI entry point |
| `prompts/epimine_episode_generation.txt` | LLM prompt for episode refinement |
| `prompts/unified_hierarchical.txt` | QA prompt template |
| `cache/epimine_episodes_*.json` | Cached episode hierarchies |
