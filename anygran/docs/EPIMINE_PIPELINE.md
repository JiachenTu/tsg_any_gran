# EpiMine Pipeline: Comprehensive Documentation

This document provides a complete analysis of the EpiMine pipeline for Scene Graph Question Answering (SGQA), comparing it with the original EpiMine implementation for news event detection.

---

## 1. Overview

**EpiMine** is an unsupervised episode detection framework that mines episodic structures from sequential data. The core insight: **episodes cannot be detected using semantic similarity alone**—they require analyzing **shifts in term co-occurrence patterns**.

### Core Philosophy (Shared by Both Implementations)

1. **Discriminative Key Terms**: Identify terms that distinguish foreground from background using salience scoring
2. **Co-occurrence as Structure Signal**: Terms that co-occur form "who-did-what" relationships; shifts indicate episode boundaries
3. **Statistical Thresholds**: Use `mean ± 1σ` to adaptively detect boundaries
4. **LLM Refinement**: Use LLM to generate structured episode descriptions from unsupervised boundaries

---

## 2. Pipeline Comparison

### Original EpiMine (News Domain) - 6 Steps

```
Articles → Segmentation → Key Terms → Co-occurrence →
Boundary Detection → Cross-Article Clustering → LLM Refinement
```

### SGQA Adaptation - 3 Classes

```
Action Graphs → EpiMineActionAnalyzer → EpiMineEpisodeGenerator →
EpiMineHierarchicalEvaluator → QA Results
```

### Step-by-Step Mapping

| Step | Original EpiMine (News) | SGQA Adaptation | Code Location |
|------|------------------------|-----------------|---------------|
| **1. Data Loading** | Segment articles into sentences, load background corpus (31 events) | Load action graphs, build background from all sgqa.jsonl (2546 actions) | `build_background_dataset()` line 945 |
| **2. Key Terms** | Word-level salience, min_freq=5, term expansion (cosine ≥0.75) | Triplet term salience, min_freq=2, no expansion | `get_key_terms()` lines 242-270 |
| **3. Co-occurrence** | Word pair frequency in sentences, scaled by IDF | Jaccard similarity between action term sets | `compute_cooccurrence_matrix()` lines 272-311 |
| **4. Boundary Detection** | mean-1σ on 3 factors (salience, co-occur, similarity) | mean-1σ on consecutive co-occurrence only | `detect_episode_boundaries()` lines 313-367 |
| **5. Clustering** | Hierarchical agglomerative across top 25% articles | N/A (single sample processing) | — |
| **6. LLM Refinement** | Claude-2.1 generates {subject,action,object,time,location} | GPT5-mini generates names + structured episodes | `generate_episode_hierarchy()` lines 561-638 |

---

## 3. Hyperparameters & Tunable Values

### 3.1 Common Parameters (Both Implementations)

| Parameter | Original | SGQA | Formula/Description |
|-----------|----------|------|---------------------|
| **Salience formula** | `(1+log(fg)²)×log(bg_total/bg)` | Same | Discriminative term scoring |
| **Boundary threshold** | `mean - 1×std` | `mean - threshold_std×std` | Episode boundary detection |
| **Foreground log exponent** | 2 | 2 | Boost repeated foreground terms |

### 3.2 Original EpiMine Parameters

| Parameter | Default | Code Location | Description |
|-----------|---------|---------------|-------------|
| `min_freq` | **5** | `epimine.py:53` | Minimum term frequency for salience |
| `sim_thresh` (expandTerms) | **0.75** | `epimine.py:68` | Cluster similar terms |
| `sim_thresh` (main call) | **0.85** | `epimine.py:417` | Stricter for main expansion |
| `doc_thresh` | **0.25** | `run.py:148` | Use only top 25% documents |
| `eval_top` | **5** | `run.py:155` | Segments per episode for evaluation |
| `similarity_weight` | **2×** | `epimine.py:301` | Weight similarity vs co-occurrence |
| `clustering_linkage` | `complete` | `epimine.py:304` | Hierarchical clustering method |
| `LLM model` | `claude-2.1` | `epimine.py:320` | Episode description generation |
| `LLM max_tokens` | **2048** | `epimine.py:321` | Context window |
| `LLM temperature` | **1.0** | `epimine.py:323` | Response randomness |
| `LLM trials` | **10** | `run.py:151` | Inference attempts |
| `emb_dim` | **768** | `run.py:149` | MPNET embedding dimension |
| `batch_size` | **32** | `run.py:150` | Encoding batch size |
| `lm_type` | `bbu` | `static_representations.py:182` | BERT uncased |
| `layer` | **12** | `static_representations.py:185` | BERT layer for embeddings |

### 3.3 SGQA Adaptation Parameters

| Parameter | Default | CLI Flag | Code Location | Description |
|-----------|---------|----------|---------------|-------------|
| `min_freq` | **2** | — | `epimine_hierarchical_sgqa.py:196` | Lower threshold (smaller dataset) |
| `threshold_std` | **1.0** | `--cooccur-threshold` | lines 316, 372 | Boundary sensitivity multiplier |
| `top_k` | **None** | — | line 242 | Limit on key terms (None = all) |
| `use_llm` | **True** | `--no-llm` | line 566 | Use GPT for episode names |
| `max_workers` | **5** | `--workers` | line 886 | Parallel QA evaluation threads |
| `temperature` | **0.1** | — | lines 42, 58 | GPT temperature (low for consistency) |
| `model` | `gpt5-mini` | `--model` | CLI argument | GPT5-mini or GPT5 |
| `limit` | **None** | `--limit` | CLI argument | Sample limit for testing |
| `max_discriminative_terms` | **5** | — | line 515 | Terms per episode (hardcoded) |

### 3.4 Relation Type Mappings (SGQA-Specific)

These define how triplets are parsed into semantic roles:

```python
VERB_RELATIONS = {"verb", "verbs"}           # line 84
OBJECT_RELATIONS = {"dobj", "obj", "pobj"}   # line 85
INSTRUMENT_RELATIONS = {"with"}               # line 86
SOURCE_RELATIONS = {"from"}                   # line 87
TARGET_RELATIONS = {"to", "on", "in", "into", "onto", "inside"}  # line 88
```

---

## 4. Key Concepts Explained

### 4.1 What is `min_freq`?

**`min_freq`** is the minimum number of times a term must appear in the foreground (current sample) to be considered as a potential key term.

#### How It Works

```python
def salience(term, foreground, background, min_freq=5):
    fg_count = count_occurrences(term, foreground)

    # FILTERING: If term appears too rarely, reject it
    if fg_count < min_freq:
        return -1  # Filtered out!

    # Only compute salience for frequent enough terms
    return (1 + log(fg_count)²) × log(bg_total / bg_count)
```

#### Why Filter by min_freq?

| Problem | Solution |
|---------|----------|
| Rare terms (appearing 1-2 times) may be noise | Filter them out |
| One-off terms don't indicate episode structure | Require minimum frequency |
| Computational efficiency | Fewer terms to process |

#### Original vs SGQA Values

| Implementation | min_freq | Reason |
|---------------|----------|--------|
| **Original EpiMine (News)** | 5 | Large, open vocabulary (thousands of unique words) |
| **SGQA Adaptation** | 2 | Small, fixed vocabulary (~100 unique terms) |

#### Example

```
Foreground (current sample): 11 actions
Term frequencies:
  "person": 11  → ✓ KEPT (11 > 2)
  "pick-up": 3  → ✓ KEPT (3 > 2)
  "mop-stick": 4 → ✓ KEPT (4 > 2)
  "cabinet": 1  → ✗ FILTERED (1 ≤ 2)
  "wall": 1     → ✗ FILTERED (1 ≤ 2)
```

---

### 4.2 What is Term Expansion (cosine ≥ 0.75)?

**Term Expansion** is a technique used in the **original EpiMine** (news domain) to handle synonyms and semantically similar words. **SGQA does NOT use this** because scene graphs have a fixed vocabulary.

#### The Problem (News Domain)

News articles use varied vocabulary for the same concept:
- "killed", "murdered", "slain", "assassinated" → same meaning
- "attack", "assault", "strike" → same meaning

Without expansion, these are treated as completely separate terms, missing important co-occurrence patterns.

#### The Solution: Two-Stage Expansion

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Expand Key Terms                         │
│                    (threshold = 0.75)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Compute salience for ALL vocabulary words                       │
│  2. Keep high-salience words as initial "key terms"                 │
│  3. For each EXCLUDED (low-salience) word:                          │
│     - Compute cosine similarity to all key terms (using embeddings) │
│     - If max similarity ≥ 0.75, ADD to key terms                    │
│                                                                      │
│  Example:                                                            │
│    Key terms: ["killed", "attack", "explosion"]                     │
│    Excluded "murdered" → cosine("murdered", "killed") = 0.82        │
│    0.82 ≥ 0.75 → ADD "murdered" to key terms ✓                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: Cluster Similar Terms                     │
│                   (threshold = 0.85, stricter)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Compute pairwise cosine similarity within key terms             │
│  2. Group terms with similarity ≥ 0.85 into clusters                │
│  3. Treat each cluster as ONE concept for co-occurrence             │
│                                                                      │
│  Example:                                                            │
│    cosine("killed", "murdered") = 0.88 ≥ 0.85 → CLUSTER             │
│    cosine("killed", "attacked") = 0.72 < 0.85 → SEPARATE            │
│                                                                      │
│  Result:                                                             │
│    Cluster 0: ["killed", "murdered", "slain"]  → treated as ONE     │
│    Cluster 1: ["attack", "assault"]            → treated as ONE     │
│    Cluster 2: ["explosion"]                    → singleton          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Original EpiMine Code

```python
# epimine.py lines 68-102
def expandTerms(all_terms, all_cos, sim_thresh=0.75):
    """
    Group semantically similar terms into clusters.

    Args:
        all_terms: List of key terms
        all_cos: Pairwise cosine similarity matrix
        sim_thresh: Minimum similarity to cluster (default 0.75)

    Returns:
        word_clusters: List of term clusters
    """
    for w_id, term in enumerate(all_terms):
        max_similarity = max(all_cos[w_id])

        if max_similarity < sim_thresh:
            # Term is unique, create singleton cluster
            word_clusters.append([term])
        else:
            # Merge with most similar term's cluster
            merge_clusters(term, most_similar_term)

    return word_clusters
```

#### Two Threshold Values

| Threshold | Value | Purpose | Code Location |
|-----------|-------|---------|---------------|
| **Expansion threshold** | 0.75 | Add excluded terms to key terms | `epimine.py:410` |
| **Clustering threshold** | 0.85 | Group key terms into clusters | `epimine.py:417` |

The stricter 0.85 threshold for clustering ensures only truly synonymous terms are merged.

#### Why SGQA Doesn't Need Term Expansion

| Aspect | News Domain | SGQA Scene Graphs |
|--------|-------------|-------------------|
| **Vocabulary** | Open-ended (any English word) | Fixed (~100 predefined terms) |
| **Synonyms** | Common ("killed"/"murdered") | None (standardized actions) |
| **Variation** | High (different writers) | None (machine-generated) |
| **Embeddings** | Required (BERT/MPNET) | Not needed |

Scene graphs use a **controlled vocabulary** with standardized action verbs (`pick-up`, `place`, `open`) and object names. There's only one way to express each action, so no expansion or clustering is needed.

---

## 5. Key Differences

### 5.1 Structural Differences

| Aspect | Original EpiMine | SGQA Adaptation |
|--------|------------------|-----------------|
| **Input unit** | Article sentence (free text) | Action graph (structured triplets) |
| **Background corpus** | 31 event articles | 2546 action graphs from sgqa.jsonl |
| **Multi-document** | Yes (cluster across articles) | No (per-sample processing) |
| **Document selection** | Top 25% only | Use all samples |
| **Term matching** | Substring in sentence | Exact match in triplet set |
| **Embedding model** | BERT + MPNET | Not used |
| **Similarity in boundaries** | Yes (3 factors) | No (co-occurrence only) |

### 5.2 Episode Structure Differences

**Original EpiMine Episode:**
```json
{
  "title": {
    "subject": "entity",
    "action": "verb",
    "object": "target",
    "time": "when",
    "location": "where"
  },
  "keywords": ["term1", "term2"],
  "example_sentences": ["sent1", "sent2"]
}
```

**SGQA Episode (Richer Structure):**
```json
{
  "episode_id": 0,
  "name": "Tool Retrieval",
  "description": "Pick up cleaning tools",
  "core_structure": {
    "agent": "person",
    "primary_actions": ["pick-up", "grasp"],
    "primary_objects": ["mop-stick", "cloth"],
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
  "discriminative_terms": ["pick-up", "mop-stick"],
  "salience_score": 0.85
}
```

### 5.3 Algorithm Differences

| Algorithm | Original EpiMine | SGQA Adaptation |
|-----------|------------------|-----------------|
| **Salience** | Same formula | Same formula |
| **Co-occurrence** | Word pair frequency matrix, IDF-scaled | Jaccard similarity between term sets |
| **Boundary detection** | 3 conditions (salience + co-occur + similarity) | 1 condition (co-occurrence only) |
| **Threshold** | Fixed `mean - 1×std` | Configurable `mean - threshold_std×std` |
| **Cross-sample clustering** | Hierarchical agglomerative | Not applicable |
| **Term expansion** | Cluster similar terms (cosine ≥0.75) | No expansion |

---

## 6. What's Common (Core Philosophy)

### 6.1 Salience-Based Key Term Selection

Both use the same discriminative salience formula:

```
salience(term) = (1 + log(fg_count)²) × log(bg_total / bg_count)
```

- **Foreground boost**: `(1 + log(fg)²)` amplifies terms appearing repeatedly in current context
- **IDF component**: `log(bg_total / bg)` penalizes common terms

### 6.2 Co-occurrence as Structure Signal

Both detect episode boundaries by analyzing shifts in term co-occurrence patterns:
- High co-occurrence between units → same episode
- Drop in co-occurrence → episode boundary

### 6.3 Statistical Thresholds

Both use `mean ± 1σ` for adaptive boundary detection:
- Boundaries: `score < mean - 1×std`
- This adapts to the data distribution automatically

### 6.4 LLM for Refinement

Both use LLM to refine unsupervised boundaries:
- Unsupervised detection provides candidate boundaries
- LLM generates human-readable names and descriptions
- Hybrid approach: statistical grounding + semantic coherence

### 6.5 Discriminative Terms per Episode

Both track which terms characterize each episode:
- Original: `keywords` field (5-10 terms)
- SGQA: `discriminative_terms` field (up to 5 terms)

---

## 7. What's Different (Domain Adaptation)

### 7.1 No Cross-Document Clustering

**Original**: Merges similar episodes across multiple news articles using hierarchical clustering with complete linkage.

**SGQA**: Processes each sample independently—no cross-sample episode merging. Each activity sequence is self-contained.

**Why**: News events have multiple perspectives across articles; activity sequences are single coherent units.

### 7.2 No Document Selection

**Original**: Uses only top 25% of documents (ranked by episode quality) for clustering.

**SGQA**: Uses all 99 samples in the dataset.

**Why**: Smaller dataset (99 vs potentially thousands of articles) makes filtering unnecessary.

### 7.3 No Embedding Similarity

**Original**: Uses BERT embeddings + cosine similarity as one of three boundary detection factors.

**SGQA**: Uses only co-occurrence (Jaccard similarity).

**Why**: Scene graph triplets are already structured—semantic similarity is captured by shared terms.

### 7.4 Structured Output Format

**Original**: Simple 5-field structure: `{subject, action, object, time, location}`

**SGQA**: Rich nested structure with:
- `core_structure`: Multiple actions, objects, instruments, locations (as lists)
- `time`: Explicit action indices and duration
- `temporal_context`: Explicit episode ordering relationships
- `salience_score`: Normalized importance metric

### 7.5 Lower Minimum Frequency

**Original**: `min_freq=5` (larger vocabulary from news text)

**SGQA**: `min_freq=2` (smaller, fixed vocabulary from scene graphs)

**Why**: Scene graphs have limited vocabulary (predefined verbs, objects)—stricter filtering would eliminate too many terms.

### 7.6 Jaccard vs Count-Based Co-occurrence

**Original**: Counts word pair frequencies, applies IDF scaling

**SGQA**: Jaccard similarity = `|intersection| / |union|` of term sets

**Why**: Action graphs have variable numbers of terms; Jaccard normalizes for this.

---

## 8. Tuning Recommendations

### 8.1 Episode Granularity

| Goal | Parameter | Value | Effect |
|------|-----------|-------|--------|
| **More episodes** | `threshold_std` | 0.5 | Lower threshold → more boundaries |
| **Fewer episodes** | `threshold_std` | 1.5-2.0 | Higher threshold → fewer boundaries |

```bash
# More fine-grained episodes
python run_epimine_hierarchical_sgqa.py --cooccur-threshold 0.5

# Coarser episodes
python run_epimine_hierarchical_sgqa.py --cooccur-threshold 1.5
```

### 8.2 Key Term Strictness

| Goal | Parameter | Value | Effect |
|------|-----------|-------|--------|
| **More terms** | `min_freq` | 1 | Include rare terms |
| **Fewer terms** | `min_freq` | 3-5 | Only frequent terms |

Requires code modification in `EpiMineActionAnalyzer.__init__()`.

### 8.3 LLM Configuration

| Goal | Parameter | Value | Effect |
|------|-----------|-------|--------|
| **Better quality** | `model` | `gpt5` | Higher quality, higher cost |
| **Faster/cheaper** | `model` | `gpt5-mini` | Lower quality, lower cost |
| **No LLM** | `use_llm` | `False` | Use default names only |
| **More creative** | `temperature` | 0.3-0.5 | More varied descriptions |

```bash
# Use larger model
python run_epimine_hierarchical_sgqa.py --model gpt5

# Skip LLM entirely
python run_epimine_hierarchical_sgqa.py --no-llm
```

### 8.4 Performance Tuning

| Goal | Parameter | Value | Effect |
|------|-----------|-------|--------|
| **Faster evaluation** | `max_workers` | 10 | More parallel threads |
| **Lower resource usage** | `max_workers` | 2-3 | Fewer parallel threads |

```bash
python run_epimine_hierarchical_sgqa.py --workers 10
```

---

## 9. Code Reference

### 9.1 File Locations

| Component | File Path |
|-----------|-----------|
| SGQA Implementation | `/home/jtu9/sgg/tsg-bench/anygran/epimine_hierarchical_sgqa.py` |
| SGQA Entry Point | `/home/jtu9/sgg/tsg-bench/anygran/run_epimine_hierarchical_sgqa.py` |
| SGQA Prompt | `/home/jtu9/sgg/tsg-bench/anygran/prompts/epimine_episode_generation.txt` |
| Original EpiMine | `/home/jtu9/sgg/structure_mining/epimine/epimine.py` |
| Original Docs | `/home/jtu9/sgg/structure_mining/epimine/docs/EPIMINE_METHODOLOGY.md` |

### 9.2 Key Functions by Line Number

**SGQA Implementation (`epimine_hierarchical_sgqa.py`):**

| Function | Lines | Purpose |
|----------|-------|---------|
| `compute_salience()` | 217-240 | Discriminative term scoring |
| `get_key_terms()` | 242-270 | Get sorted key terms |
| `compute_cooccurrence_matrix()` | 272-311 | Jaccard similarity matrix |
| `detect_episode_boundaries()` | 313-367 | Boundary detection with threshold |
| `_build_structured_episode()` | 450-559 | Build rich episode structure |
| `generate_episode_hierarchy()` | 561-638 | LLM refinement + assembly |
| `_format_epimine_timeline()` | 766-834 | Format for QA prompt |

**Original EpiMine (`epimine.py`):**

| Function | Lines | Purpose |
|----------|-------|---------|
| `salience()` | 53-66 | Discriminative term scoring |
| `expandTerms()` | 68-102 | Cluster similar terms |
| `co_occurrence()` | 111-133 | Word pair co-occurrence matrix |
| `segment_joint_salience()` | 135-204 | 3-factor segment scoring |
| `episode_segmentation()` | 208-288 | Boundary detection |
| `clusterEpisodes()` | 292-318 | Cross-article clustering |

---

## 10. Summary

### What EpiMine Brings to SGQA

1. **Principled boundaries**: Statistical detection instead of LLM intuition
2. **Discriminative focus**: Salience scoring highlights characteristic terms
3. **Reproducibility**: Same input → same boundaries (deterministic)
4. **Background context**: Distributional grounding across dataset

### What SGQA Adaptation Adds

1. **Structured episodes**: Rich format for QA grounding
2. **Temporal context**: Explicit episode ordering
3. **Per-sample processing**: Simpler pipeline for activity sequences
4. **Configurable threshold**: `--cooccur-threshold` for experimentation

### Performance

| Method | Exact Match (500 questions) |
|--------|----------------------------|
| Baseline | 86.0% |
| EpiMine-Hierarchical | 88.0% |
| **Improvement** | **+2.0%** |
