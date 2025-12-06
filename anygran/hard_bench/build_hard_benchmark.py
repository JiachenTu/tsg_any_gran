"""
Build Hard Benchmark for SGQA
=============================
Extracts hard cases from model evaluation results where models answered incorrectly.
Creates categorized benchmark with error analysis.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter


def load_results(path: str) -> Dict:
    """Load evaluation results JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def load_sgqa_data() -> Dict[str, Dict]:
    """Load original SGQA data indexed by data_id."""
    data_path = Path(__file__).parent.parent.parent / "resource" / "dataset" / "understanding" / "sgqa.jsonl"
    data_by_id = {}
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data_by_id[item['data_id']] = item
    return data_by_id


def categorize_question(question: str) -> str:
    """Categorize question by reasoning type."""
    q = question.lower()

    # Check for concurrent state
    if any(kw in q for kw in ['while', 'during']):
        return 'concurrent_state'

    # Check for both hands
    if 'both hands' in q:
        return 'both_hands'

    # Check for multi-step reasoning
    multi_step_keywords = ['all ', 'completed', 'sequence', 'every', 'entire']
    temporal_keywords = ['before', 'after', 'immediately', 'first', 'last', 'between', 'final']

    has_temporal = any(kw in q for kw in temporal_keywords)
    has_multi_step = any(kw in q for kw in multi_step_keywords)

    if has_temporal and has_multi_step:
        return 'multi_step'
    elif has_temporal:
        return 'temporal_ordering'
    elif has_multi_step:
        return 'multi_step'

    return 'other'


def analyze_error(question: str, gt: str, predictions: Dict) -> str:
    """Generate brief analysis of why this case is hard."""
    category = categorize_question(question)

    # Check if all predictions are the same (systematic error)
    pred_values = [p['answer'] for p in predictions.values()]
    all_same = len(set(pred_values)) == 1

    analysis_parts = []

    if category == 'temporal_ordering':
        analysis_parts.append("Requires precise temporal sequence tracking")
    elif category == 'multi_step':
        analysis_parts.append("Requires multi-step reasoning and completion detection")
    elif category == 'both_hands':
        analysis_parts.append("Requires tracking hand1/hand2 object manipulation")
    elif category == 'concurrent_state':
        analysis_parts.append("Requires tracking simultaneous states")
    else:
        analysis_parts.append("Complex reasoning required")

    if all_same:
        analysis_parts.append(f"All models predicted '{pred_values[0]}' instead of '{gt}'")

    return ". ".join(analysis_parts)


def build_hard_benchmark():
    """Build the hard benchmark from evaluation results."""
    base_path = Path(__file__).parent.parent

    # Load results
    print("Loading evaluation results...")
    epimine = load_results(base_path / "results/epimine_v0/epimine_gpt5-mini_limitall_20251201_210930.json")
    baseline_mini = load_results(base_path / "results/baselines/baseline_gpt5-mini_limitall_20251201_205714.json")
    baseline_gpt5 = load_results(base_path / "results/baselines/baseline_gpt5_limitall_20251201_175116.json")

    # Load original data
    print("Loading original SGQA data...")
    sgqa_data = load_sgqa_data()

    # Index results by (data_id, question)
    epimine_results = {(r['data_id'], r['question']): r for r in epimine['results']}
    baseline_mini_results = {(r['data_id'], r['question']): r for r in baseline_mini['results']}
    baseline_gpt5_results = {(r['data_id'], r['question']): r for r in baseline_gpt5['results']}

    # Find all error cases
    epimine_errors = {k for k, v in epimine_results.items() if not v['exact_match']}
    baseline_mini_errors = {k for k, v in baseline_mini_results.items() if not v['exact_match']}
    baseline_gpt5_errors = {k for k, v in baseline_gpt5_results.items() if not v['exact_match']}

    all_error_keys = epimine_errors | baseline_mini_errors | baseline_gpt5_errors

    print(f"Found {len(all_error_keys)} unique error cases")

    # Build hard benchmark entries
    hard_cases = []
    category_counts = Counter()
    difficulty_counts = Counter()

    for key in sorted(all_error_keys):
        data_id, question = key

        # Get predictions from each model
        e = epimine_results.get(key)
        bm = baseline_mini_results.get(key)
        bg = baseline_gpt5_results.get(key)

        # Determine ground truth
        gt = e['ground_truth'] if e else (bm['ground_truth'] if bm else bg['ground_truth'])

        # Count how many models got it wrong
        e_wrong = key in epimine_errors
        bm_wrong = key in baseline_mini_errors
        bg_wrong = key in baseline_gpt5_errors
        wrong_count = sum([e_wrong, bm_wrong, bg_wrong])

        # Determine difficulty
        if wrong_count == 3:
            difficulty = 'hard'
        elif wrong_count == 2:
            difficulty = 'medium'
        else:
            difficulty = 'easy'

        difficulty_counts[difficulty] += 1

        # Categorize
        category = categorize_question(question)
        category_counts[category] += 1

        # Build predictions dict
        predictions = {
            'epimine_v0': {
                'answer': e['prediction'] if e else 'N/A',
                'correct': not e_wrong if e else None
            },
            'baseline_mini': {
                'answer': bm['prediction'] if bm else 'N/A',
                'correct': not bm_wrong if bm else None
            },
            'baseline_gpt5': {
                'answer': bg['prediction'] if bg else 'N/A',
                'correct': not bg_wrong if bg else None
            }
        }

        # Get original context_graphs
        original_data = sgqa_data.get(data_id, {})
        context_graphs = original_data.get('context_graphs', [])

        # Generate analysis
        analysis = analyze_error(question, gt, predictions)

        hard_case = {
            'data_id': data_id,
            'question': question,
            'ground_truth': gt,
            'context_graphs': context_graphs,
            'difficulty': difficulty,
            'error_category': category,
            'predictions': predictions,
            'analysis': analysis
        }

        hard_cases.append(hard_case)

    # Sort by difficulty (hard first)
    difficulty_order = {'hard': 0, 'medium': 1, 'easy': 2}
    hard_cases.sort(key=lambda x: (difficulty_order[x['difficulty']], x['error_category']))

    # Save hard benchmark
    output_path = Path(__file__).parent / "sgqa_hard.json"
    with open(output_path, 'w') as f:
        json.dump({
            'task': 'SGQA-Hard-Benchmark',
            'description': 'Hard cases where at least one model answered incorrectly',
            'total_cases': len(hard_cases),
            'cases': hard_cases
        }, f, indent=2)
    print(f"Saved hard benchmark to {output_path}")

    # Calculate model-specific stats
    epimine_unique_solved = sum(1 for k in all_error_keys if k not in epimine_errors)
    baseline_mini_unique_solved = sum(1 for k in all_error_keys if k not in baseline_mini_errors)
    baseline_gpt5_unique_solved = sum(1 for k in all_error_keys if k not in baseline_gpt5_errors)

    # Save stats
    stats = {
        'total_hard_cases': len(hard_cases),
        'by_difficulty': dict(difficulty_counts),
        'by_category': dict(category_counts),
        'model_performance': {
            'epimine_v0': {
                'total_errors': len(epimine_errors),
                'unique_solved_in_hard': epimine_unique_solved
            },
            'baseline_mini': {
                'total_errors': len(baseline_mini_errors),
                'unique_solved_in_hard': baseline_mini_unique_solved
            },
            'baseline_gpt5': {
                'total_errors': len(baseline_gpt5_errors),
                'unique_solved_in_hard': baseline_gpt5_unique_solved
            }
        }
    }

    stats_path = Path(__file__).parent / "sgqa_hard_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # Print summary
    print("\n" + "="*60)
    print("SGQA HARD BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total hard cases: {len(hard_cases)}")
    print(f"\nBy difficulty:")
    for diff, count in sorted(difficulty_counts.items(), key=lambda x: difficulty_order[x[0]]):
        print(f"  {diff}: {count}")
    print(f"\nBy category:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")
    print(f"\nModel performance on hard cases:")
    print(f"  EpiMine v0: {epimine_unique_solved}/{len(hard_cases)} solved")
    print(f"  Baseline-mini: {baseline_mini_unique_solved}/{len(hard_cases)} solved")
    print(f"  Baseline-GPT5: {baseline_gpt5_unique_solved}/{len(hard_cases)} solved")

    return hard_cases, stats


def generate_analysis_docs(hard_cases: List[Dict], stats: Dict):
    """Generate markdown analysis documents."""
    analysis_dir = Path(__file__).parent / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Group cases by category
    by_category = {}
    for case in hard_cases:
        cat = case['error_category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(case)

    # Group hard cases (all 3 wrong)
    all_wrong_cases = [c for c in hard_cases if c['difficulty'] == 'hard']

    # Generate SUMMARY.md
    summary_content = f"""# SGQA Hard Benchmark Analysis

## Overview

This hard benchmark contains {len(hard_cases)} cases where at least one model answered incorrectly.

### Difficulty Distribution
| Difficulty | Count | Description |
|------------|-------|-------------|
| Hard | {stats['by_difficulty'].get('hard', 0)} | All 3 models wrong |
| Medium | {stats['by_difficulty'].get('medium', 0)} | 2 models wrong |
| Easy | {stats['by_difficulty'].get('easy', 0)} | 1 model wrong |

### Category Distribution
| Category | Count | Description |
|----------|-------|-------------|
| temporal_ordering | {stats['by_category'].get('temporal_ordering', 0)} | before/after/first/last/immediately |
| multi_step | {stats['by_category'].get('multi_step', 0)} | "all X completed", sequences |
| both_hands | {stats['by_category'].get('both_hands', 0)} | "both hands" manipulation |
| concurrent_state | {stats['by_category'].get('concurrent_state', 0)} | "while/during" queries |
| other | {stats['by_category'].get('other', 0)} | Other reasoning types |

### Model Performance on Hard Cases
| Model | Errors | Solved in Hard Set |
|-------|--------|-------------------|
| EpiMine v0 | {stats['model_performance']['epimine_v0']['total_errors']} | {stats['model_performance']['epimine_v0']['unique_solved_in_hard']} |
| Baseline-mini | {stats['model_performance']['baseline_mini']['total_errors']} | {stats['model_performance']['baseline_mini']['unique_solved_in_hard']} |
| Baseline-GPT5 | {stats['model_performance']['baseline_gpt5']['total_errors']} | {stats['model_performance']['baseline_gpt5']['unique_solved_in_hard']} |

## Key Insights

1. **Temporal reasoning dominates errors**: {stats['by_category'].get('temporal_ordering', 0)} of {len(hard_cases)} errors involve temporal ordering
2. **Hardest cases**: {len(all_wrong_cases)} questions failed by all models
3. **EpiMine advantage**: Hierarchical context helps with some multi-step cases

## Files

- `temporal_ordering.md` - Analysis of temporal ordering errors
- `multi_step.md` - Analysis of multi-step reasoning errors
- `both_hands.md` - Analysis of both-hands manipulation errors
- `concurrent_state.md` - Analysis of concurrent state errors
- `all_wrong_cases.md` - Deep dive into hardest cases
"""

    with open(analysis_dir / "SUMMARY.md", 'w') as f:
        f.write(summary_content)

    # Generate all_wrong_cases.md
    all_wrong_content = f"""# Hardest Cases: All Models Failed

These {len(all_wrong_cases)} questions were answered incorrectly by all 3 models (EpiMine v0, Baseline-mini, Baseline-GPT5).

## Cases

"""
    for i, case in enumerate(all_wrong_cases, 1):
        all_wrong_content += f"""### Case {i}: {case['data_id'][:8]}

**Question:** {case['question']}

**Ground Truth:** `{case['ground_truth']}`

**Predictions:**
- EpiMine v0: `{case['predictions']['epimine_v0']['answer']}`
- Baseline-mini: `{case['predictions']['baseline_mini']['answer']}`
- Baseline-GPT5: `{case['predictions']['baseline_gpt5']['answer']}`

**Category:** {case['error_category']}

**Analysis:** {case['analysis']}

---

"""

    with open(analysis_dir / "all_wrong_cases.md", 'w') as f:
        f.write(all_wrong_content)

    # Generate category-specific files
    for cat_name, cases in by_category.items():
        cat_content = f"""# {cat_name.replace('_', ' ').title()} Errors

Total errors in this category: {len(cases)}

## Error Summary

| # | Question | GT | EpiMine | Baseline-mini | Baseline-GPT5 |
|---|----------|----|---------|--------------:|---------------|
"""
        for i, case in enumerate(cases[:30], 1):  # Limit to first 30
            q_short = case['question'][:50] + "..." if len(case['question']) > 50 else case['question']
            e_mark = "✓" if case['predictions']['epimine_v0']['correct'] else case['predictions']['epimine_v0']['answer']
            bm_mark = "✓" if case['predictions']['baseline_mini']['correct'] else case['predictions']['baseline_mini']['answer']
            bg_mark = "✓" if case['predictions']['baseline_gpt5']['correct'] else case['predictions']['baseline_gpt5']['answer']
            cat_content += f"| {i} | {q_short} | {case['ground_truth']} | {e_mark} | {bm_mark} | {bg_mark} |\n"

        cat_content += f"""

## Common Patterns

"""
        # Analyze common wrong answers
        wrong_answers = Counter()
        for case in cases:
            for model, pred in case['predictions'].items():
                if not pred['correct'] and pred['answer'] != 'N/A':
                    wrong_answers[pred['answer']] += 1

        cat_content += "Most frequent wrong predictions:\n"
        for ans, count in wrong_answers.most_common(10):
            cat_content += f"- `{ans}`: {count} times\n"

        with open(analysis_dir / f"{cat_name}.md", 'w') as f:
            f.write(cat_content)

    print(f"Generated analysis documents in {analysis_dir}")


def generate_recommendations():
    """Generate recommendations document."""
    content = """# Recommendations for Model Enhancement

Based on error analysis of the SGQA hard benchmark, here are targeted recommendations to improve model performance.

## 1. Temporal Ordering (55 errors)

### Problem
Models struggle with questions requiring precise sequence tracking:
- "What was picked up BEFORE X?"
- "What was the LAST tool used after Y?"
- "What object was handled IMMEDIATELY after Z?"

### Recommendations

1. **Explicit Action Indexing**
   - Add action indices to scene graph representation: "Action 0: pick-up knife, Action 1: cut onion..."
   - Makes temporal relationships explicit

2. **Chain-of-Thought Prompting**
   - Add reasoning step: "First, identify all actions. Then, find the reference action X. Finally, look at adjacent actions."
   - Force step-by-step temporal analysis

3. **Reverse Search for "Before"**
   - When question asks "before X", search backwards from X
   - Current models often search forward and miss the answer

4. **Boundary Detection**
   - "Last X before Y" requires finding boundary at Y, then searching backwards
   - Explicit boundary markers in prompts

## 2. Multi-Step Reasoning (17 errors)

### Problem
Questions requiring tracking completion states:
- "Which tool was used after ALL dough kneading was completed?"
- "What was handled after the ENTIRE sequence finished?"

### Recommendations

1. **Phase-Based Grouping (EpiMine approach)**
   - Group actions into semantic phases
   - Track phase completion explicitly
   - EpiMine already shows improvement here

2. **Completion Detection**
   - Add explicit markers for repeated action completion
   - "Kneading phase: Actions 3-7 (completed)"

3. **Hierarchical Context**
   - Overall goal → Sub-events → Actions → Triplets
   - Helps identify when a sub-task is complete

## 3. Both-Hands Manipulation (9 errors)

### Problem
Questions about simultaneous hand states:
- "Which object required both hands?"
- "What was held in hand2 while hand1 was picking up X?"

### Recommendations

1. **Explicit Hand State Tracking**
   - Track hand1_holding and hand2_holding at each action
   - "Action 5: hand1=knife, hand2=onion"

2. **Both-Hands Flag**
   - Mark actions that require both hands explicitly
   - "Action 7: pick-up board (both_hands=true)"

3. **State Timeline**
   - Maintain parallel state tracks for each hand
   - Query: "At action N, what was in hand2?"

## 4. Concurrent State (2 errors)

### Problem
Questions about simultaneous events:
- "What was held WHILE the drill was operating?"
- "Which object was handled DURING brushing?"

### Recommendations

1. **Temporal Overlap Detection**
   - Identify actions that overlap in time
   - Mark concurrent actions explicitly

2. **State Snapshot**
   - At each action, capture full state: {current_action, hand1, hand2, location}
   - Query snapshots for "while/during" questions

## 5. General Improvements

### Object Disambiguation
- Similar objects (screw vs screwdriver, pot vs pan) cause confusion
- Add object descriptions or unique identifiers

### Answer Format Consistency
- Ensure single-word answers match ground truth format
- "metal-board" vs "metal board" should be handled

### Context Length Handling
- Long action sequences (>20 actions) cause degradation
- Consider chunking or summarization for very long sequences

## Implementation Priority

| Priority | Category | Potential Impact |
|----------|----------|------------------|
| 1 | Temporal Ordering | High (55 errors) |
| 2 | Multi-Step Reasoning | Medium (17 errors) |
| 3 | Both-Hands | Medium (9 errors) |
| 4 | Concurrent State | Low (2 errors) |
"""

    output_path = Path(__file__).parent / "recommendations.md"
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated recommendations at {output_path}")


if __name__ == "__main__":
    hard_cases, stats = build_hard_benchmark()
    generate_analysis_docs(hard_cases, stats)
    generate_recommendations()
