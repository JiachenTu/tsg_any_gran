# EpiMine Adaptation for Hierarchical Scene Graph QA

## Overview

This document describes how the EpiMine unsupervised episode detection framework is adapted for action scene graph understanding. The adaptation applies EpiMine's core philosophy — detecting episodes via co-occurrence pattern shifts — to the structured triplet format of scene graphs.

---

## Structured Episode Definition

### Original EpiMine (News Domain)

```json
{
  "subject": "Entity performing action",
  "action": "Verb describing activity",
  "object": "Target of action",
  "time": "When it occurred",
  "location": "Where it occurred"
}
```

### Adapted for Scene Graphs

For action scene graphs, we extend this to capture the richer relational structure:

```json
{
  "episode_id": 0,
  "name": "Preparation Phase",
  "description": "Retrieve cleaning tools from storage",

  "core_structure": {
    "agent": "person",
    "primary_actions": ["pick-up", "open"],
    "primary_objects": ["mop-stick", "cabinet"],
    "instruments": ["hand1", "hand2"],
    "source_locations": ["floor", "cabinet"],
    "target_locations": null
  },

  "time": {
    "action_indices": [0, 1, 2],
    "start_index": 0,
    "end_index": 2,
    "duration": 3
  },

  "temporal_context": {
    "position": "beginning",
    "precedes_episodes": [1, 2],
    "follows_episodes": null
  },

  "discriminative_terms": ["pick-up", "mop-stick", "cabinet"],
  "salience_score": 0.85
}
```

---

## Field Mapping

### From EpiMine to Scene Graphs

| EpiMine (News) | Scene Graph Episode | Source Triplet Pattern |
|----------------|---------------------|------------------------|
| `subject` | `agent` | `[person, verb, X]` → "person" |
| `action` | `primary_actions` | `[person, verb, X]` → X |
| `object` | `primary_objects` | `[action, dobj, X]` → X |
| `time` | `time.action_indices` | Scene graph indices [0, 1, 2] |
| `location` | `source_locations` / `target_locations` | `[X, from/to/on/in, Y]` → Y |
| *(new)* | `instruments` | `[action, with, X]` → X |

### Field Definitions

| Field | Source | Type | Description |
|-------|--------|------|-------------|
| `agent` | `[person, verb, X]` triplets | `str` | Who performs the actions (usually "person") |
| `primary_actions` | `[person, verb, X]` → X | `List[str]` | Core verbs defining this episode (can be multiple) |
| `primary_objects` | `[action, dobj, X]` → X | `List[str]` | Main objects being manipulated (can be multiple) |
| `instruments` | `[action, with, X]` → X | `List[str]` | Tools/hands used (can be multiple) |
| `source_locations` | `[X, from, Y]` → Y | `List[str]` | Where objects come from (can be multiple) |
| `target_locations` | `[action, to/on/in, X]` → X | `List[str]` | Where objects go (can be multiple) |
| `time.action_indices` | Scene graph order | `List[int]` | Which action graphs belong to this episode |
| `time.start_index` | First action | `int` | Starting action index |
| `time.end_index` | Last action | `int` | Ending action index |
| `time.duration` | Count | `int` | Number of actions in episode |
| `temporal_context.position` | Relative position | `str` | "beginning", "middle", or "end" |
| `temporal_context.precedes_episodes` | Episode ordering | `List[int]` | Episode IDs that come after |
| `temporal_context.follows_episodes` | Episode ordering | `List[int]` | Episode IDs that come before |
| `discriminative_terms` | Salience computation | `List[str]` | Terms that distinguish this episode |
| `salience_score` | Salience computation | `float` | Overall discriminative score (0-1) |

**Note:** All `List` fields can contain multiple items since:
- An episode may contain multiple actions (e.g., "close" + "place")
- An action may involve multiple objects (e.g., "mop-stick" + "cloth")
- Multiple instruments may be used (e.g., "hand1" + "hand2")
- Multiple locations may be involved (e.g., moving from "floor" to "cabinet")

---

## Example: Cleaning Activity

### Raw Action Sequence (from sgqa.jsonl)

```
Action 0: [pick-up, with, hand1] [pick-up, with, hand2] [mop-stick, from, floor] [person, verb, pick-up] [pick-up, dobj, mop-stick]
Action 1: [sweep, with, hand1] [sweep, with, hand2] [sweep, with, mop-stick] [sweep, dobj, floor] [sweep, in, car] [person, verb, sweep]
Action 2: [close, with, hand1] [close, dobj, door] [person, verb, close]
Action 3: [place, with, hand2] [place, dobj, mop-stick] [place, on, floor] [person, verb, place]
```

### Structured Episodes

```json
[
  {
    "episode_id": 0,
    "name": "Tool Retrieval",
    "description": "Pick up the mop from the floor",
    "core_structure": {
      "agent": "person",
      "primary_actions": ["pick-up"],
      "primary_objects": ["mop-stick"],
      "instruments": ["hand1", "hand2"],
      "source_locations": ["floor"],
      "target_locations": null
    },
    "time": {
      "action_indices": [0],
      "start_index": 0,
      "end_index": 0,
      "duration": 1
    },
    "temporal_context": {
      "position": "beginning",
      "precedes_episodes": [1, 2],
      "follows_episodes": null
    },
    "discriminative_terms": ["pick-up", "mop-stick"],
    "salience_score": 0.82
  },
  {
    "episode_id": 1,
    "name": "Sweeping",
    "description": "Sweep the car floor with the mop",
    "core_structure": {
      "agent": "person",
      "primary_actions": ["sweep"],
      "primary_objects": ["floor"],
      "instruments": ["mop-stick"],
      "source_locations": null,
      "target_locations": ["car"]
    },
    "time": {
      "action_indices": [1],
      "start_index": 1,
      "end_index": 1,
      "duration": 1
    },
    "temporal_context": {
      "position": "middle",
      "precedes_episodes": [2],
      "follows_episodes": [0]
    },
    "discriminative_terms": ["sweep", "floor", "car"],
    "salience_score": 0.91
  },
  {
    "episode_id": 2,
    "name": "Transition & Storage",
    "description": "Close door and store the mop",
    "core_structure": {
      "agent": "person",
      "primary_actions": ["close", "place"],
      "primary_objects": ["door", "mop-stick"],
      "instruments": ["hand1", "hand2"],
      "source_locations": null,
      "target_locations": ["floor"]
    },
    "time": {
      "action_indices": [2, 3],
      "start_index": 2,
      "end_index": 3,
      "duration": 2
    },
    "temporal_context": {
      "position": "end",
      "precedes_episodes": null,
      "follows_episodes": [0, 1]
    },
    "discriminative_terms": ["close", "place", "door"],
    "salience_score": 0.75
  }
]
```

---

## Conceptual Mapping

| EpiMine (News) | SGQA (Scene Graphs) |
|----------------|---------------------|
| Article | Video/Activity sequence |
| Sentence segment | Action graph (single action) |
| Key terms (words) | Verbs + Objects + Relations |
| Episode | Structured sub-event (with core_structure) |
| Background corpus | Full sgqa.jsonl dataset |

---

## Benefits Over Current v1

| Aspect | v1 (Current) | Structured Episodes (EpiMine) |
|--------|--------------|-------------------------------|
| Episode definition | Free-form text description | Structured `{agent, actions, objects, instruments, locations, time}` |
| Boundary logic | LLM intuition | Co-occurrence shifts + statistical thresholds |
| Temporal reasoning | Implicit in text | Explicit `time` and `temporal_context` fields |
| QA grounding | Match question to phase | Match question entities to episode fields |
| Discriminative info | None | `discriminative_terms` + `salience_score` |
| Reproducibility | LLM-dependent | Statistical detection is deterministic |

---

## How Structure Helps QA

**Question:** "What object was picked up before sweeping the floor?"

**Structured Matching Process:**
1. Find episode with `"sweep"` in `primary_actions` → Episode 1
2. Look at `temporal_context.follows_episodes` → [0]
3. Get Episode 0's `primary_objects` → `["mop-stick"]`
4. **Answer:** "mop-stick"

This explicit structure enables precise temporal reasoning that free-form text cannot guarantee.

---

## Why EpiMine Philosophy Fits SGQA

1. **Fixed vocabulary**: Action verbs (`pick-up`, `sweep`, `place`) and objects (`mop-stick`, `floor`) are limited — co-occurrence patterns are clear

2. **Natural episodes**: Activities have phases (prepare → execute → cleanup) that can be detected via co-occurrence shifts

3. **Small dataset**: No need for top-document selection (99 samples in sgqa.jsonl) — use all data

4. **Temporal structure**: Actions are ordered — co-occurrence shifts signal phase transitions

5. **Structured triplets**: Scene graph triplets naturally map to structured episode fields via relation patterns

---

## Implementation Architecture

### Core Classes

1. **`EpiMineActionAnalyzer`**: Unsupervised episode detection
   - Build action vocabulary from background dataset
   - Compute term salience scores
   - Build co-occurrence matrix
   - Detect episode boundaries via statistical thresholds

2. **`EpiMineEpisodeGenerator`**: LLM-based refinement
   - Take candidate boundaries from analyzer
   - Generate structured episode descriptions via GPT
   - Fill in `name`, `description`, and validate boundaries

3. **`EpiMineHierarchicalEvaluator`**: SGQA with structured episodes
   - Extends `UnifiedHierarchicalEvaluator` (v1)
   - Uses EpiMine-detected episodes instead of pure LLM generation
   - Formats structured episodes in unified/interleaved prompt style

### Key Formulas (Adapted from EpiMine)

**Salience (discriminative term importance):**
```
salience(term) = (1 + log(fg_count)²) × log(bg_total / bg_count)
```

**Boundary Detection:**
```
if co_occurrence[current, prev] < (mean - 1σ):
    start_new_episode()
```

---

## Files

| File | Purpose |
|------|---------|
| `epimine_hierarchical_sgqa.py` | Core implementation |
| `run_epimine_hierarchical_sgqa.py` | Entry point with CLI |
| `prompts/epimine_episode_generation.txt` | GPT prompt for episode refinement |
| `docs/EPIMINE_ADAPTATION.md` | This documentation |

---

## Usage

```bash
# Run with default settings
python anygran/run_epimine_hierarchical_sgqa.py --limit 10

# Use cached episodes
python anygran/run_epimine_hierarchical_sgqa.py --cache-path anygran/cache/epimine_episodes.json

# Adjust co-occurrence threshold
python anygran/run_epimine_hierarchical_sgqa.py --cooccur-threshold 0.5

# Skip baseline comparison
python anygran/run_epimine_hierarchical_sgqa.py --skip-baseline
```
