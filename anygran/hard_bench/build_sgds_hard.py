"""
Build SGDS Hard Benchmark
=========================
Creates a hard version of the SGDS (Scene Graph Description Selection) dataset
focusing on cases that require complex reasoning.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher


def load_sgds_data() -> List[Dict]:
    """Load original SGDS dataset."""
    data_path = Path(__file__).parent.parent.parent / "resource" / "dataset" / "understanding" / "sgds.jsonl"
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def count_subtle_distractors(target: str, variations: List[str], position: int) -> int:
    """Count how many distractors are semantically similar to target."""
    subtle_count = 0
    threshold = 0.6  # Similarity threshold for "subtle"

    for i, var in enumerate(variations):
        if i == position:
            continue  # Skip the correct answer
        similarity = calculate_text_similarity(target, var)
        if similarity > threshold:
            subtle_count += 1

    return subtle_count


def has_complex_verb(triplet: List[List[str]]) -> bool:
    """Check if the triplet contains a complex verb requiring multi-step reasoning."""
    complex_verbs = [
        'rotate', 'tighten', 'loosen', 'secure', 'adjust', 'position',
        'align', 'attach', 'assemble', 'install', 'connect', 'thread',
        'measure', 'calibrate', 'pour', 'mix', 'knead', 'fold'
    ]

    for t in triplet:
        if len(t) >= 3:
            # Check verb relations
            if t[1] == 'verb' and any(cv in t[2].lower() for cv in complex_verbs):
                return True
            # Check for "to" relations (indicates purpose/consequence)
            if t[1] == 'to':
                return True

    return False


def identify_hard_reason(sample: Dict, context_length: int, subtle_count: int, has_complex: bool) -> str:
    """Determine why this sample is considered hard."""
    reasons = []

    if context_length >= 15:
        reasons.append('long_context')

    if subtle_count >= 2:
        reasons.append('subtle_distractors')

    if has_complex:
        reasons.append('complex_verb')

    # Check for temporal indicators in target sentence
    temporal_words = ['then', 'after', 'before', 'while', 'during', 'finally', 'first']
    if any(tw in sample['target_sentence'].lower() for tw in temporal_words):
        reasons.append('temporal_reasoning')

    return '|'.join(reasons) if reasons else 'moderate_complexity'


def calculate_complexity_score(sample: Dict) -> Tuple[float, Dict]:
    """
    Calculate complexity score for a sample.

    Returns:
        (score, details) where score is 0-1 and details has component scores
    """
    context_length = len(sample.get('context_graphs', []))
    variations = sample.get('variations', [])
    position = sample.get('position', 0)
    target = variations[position] if position < len(variations) else ""
    triplet = sample.get('triplet', [])

    # Length score (0-1): normalize context length, cap at 30
    length_score = min(context_length / 30, 1.0)

    # Distractor similarity score (0-1): count subtle distractors
    subtle_count = count_subtle_distractors(target, variations, position)
    distractor_score = min(subtle_count / 3, 1.0)  # Max 3 similar distractors

    # Verb complexity score (0-1)
    has_complex = has_complex_verb(triplet)
    verb_score = 1.0 if has_complex else 0.5

    # Weighted combination
    complexity_score = 0.4 * length_score + 0.3 * distractor_score + 0.3 * verb_score

    details = {
        'context_length': context_length,
        'length_score': round(length_score, 3),
        'subtle_distractors': subtle_count,
        'distractor_score': round(distractor_score, 3),
        'has_complex_verb': has_complex,
        'verb_score': round(verb_score, 3)
    }

    return complexity_score, details


def build_sgds_hard():
    """Build the SGDS hard benchmark."""
    print("Loading SGDS dataset...")
    data = load_sgds_data()
    print(f"Loaded {len(data)} samples")

    # Calculate complexity for all samples
    scored_samples = []
    for sample in data:
        score, details = calculate_complexity_score(sample)
        scored_samples.append((sample, score, details))

    # Selection criteria for hard cases:
    # 1. Context length >= 15 actions OR
    # 2. Complexity score >= 0.6 OR
    # 3. Has 2+ subtle distractors
    hard_samples = []
    for sample, score, details in scored_samples:
        is_hard = (
            details['context_length'] >= 15 or
            score >= 0.6 or
            details['subtle_distractors'] >= 2
        )

        if is_hard:
            hard_reason = identify_hard_reason(
                sample,
                details['context_length'],
                details['subtle_distractors'],
                details['has_complex_verb']
            )

            # Create hard sample with metadata
            hard_sample = {
                **sample,
                'hard_reason': hard_reason,
                'context_length': details['context_length'],
                'complexity_score': round(score, 3),
                'subtle_distractors': details['subtle_distractors'],
                'has_complex_verb': details['has_complex_verb']
            }
            hard_samples.append(hard_sample)

    # Sort by complexity score (hardest first)
    hard_samples.sort(key=lambda x: -x['complexity_score'])

    print(f"\nSelected {len(hard_samples)} hard samples from {len(data)} total")

    # Save hard benchmark
    output_path = Path(__file__).parent / "sgds_hard.jsonl"
    with open(output_path, 'w') as f:
        for sample in hard_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved hard benchmark to {output_path}")

    # Calculate statistics
    stats = {
        'total_original': len(data),
        'total_hard': len(hard_samples),
        'selection_rate': round(len(hard_samples) / len(data) * 100, 1),
        'complexity_distribution': {
            'very_hard': len([s for s in hard_samples if s['complexity_score'] >= 0.8]),
            'hard': len([s for s in hard_samples if 0.6 <= s['complexity_score'] < 0.8]),
            'moderate': len([s for s in hard_samples if s['complexity_score'] < 0.6])
        },
        'by_reason': {},
        'context_length_stats': {
            'min': min(s['context_length'] for s in hard_samples),
            'max': max(s['context_length'] for s in hard_samples),
            'mean': round(sum(s['context_length'] for s in hard_samples) / len(hard_samples), 1)
        }
    }

    # Count reasons
    from collections import Counter
    reason_counter = Counter()
    for sample in hard_samples:
        for reason in sample['hard_reason'].split('|'):
            reason_counter[reason] += 1
    stats['by_reason'] = dict(reason_counter)

    # Save stats
    stats_path = Path(__file__).parent / "sgds_hard_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # Print summary
    print("\n" + "="*60)
    print("SGDS HARD BENCHMARK SUMMARY")
    print("="*60)
    print(f"Original dataset: {stats['total_original']} samples")
    print(f"Hard benchmark: {stats['total_hard']} samples ({stats['selection_rate']}%)")
    print(f"\nComplexity distribution:")
    print(f"  Very hard (>=0.8): {stats['complexity_distribution']['very_hard']}")
    print(f"  Hard (0.6-0.8): {stats['complexity_distribution']['hard']}")
    print(f"  Moderate (<0.6): {stats['complexity_distribution']['moderate']}")
    print(f"\nBy hard reason:")
    for reason, count in sorted(stats['by_reason'].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nContext length stats:")
    print(f"  Min: {stats['context_length_stats']['min']} actions")
    print(f"  Max: {stats['context_length_stats']['max']} actions")
    print(f"  Mean: {stats['context_length_stats']['mean']} actions")

    return hard_samples, stats


if __name__ == "__main__":
    hard_samples, stats = build_sgds_hard()
