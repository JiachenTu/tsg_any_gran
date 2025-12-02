"""
EpiMine Hierarchical SGQA Evaluation Runner
============================================
Main script to run EpiMine-enhanced multi-granular SGQA evaluation.

Usage:
    # Test on first 5 samples
    python anygran/run_epimine_hierarchical_sgqa.py --limit 5

    # Run full evaluation
    python anygran/run_epimine_hierarchical_sgqa.py

    # Generate episodes only (no QA evaluation)
    python anygran/run_epimine_hierarchical_sgqa.py --generate-only --limit 5

    # Use cached episodes
    python anygran/run_epimine_hierarchical_sgqa.py --cache-path anygran/cache/epimine_episodes.json

    # Adjust co-occurrence threshold
    python anygran/run_epimine_hierarchical_sgqa.py --cooccur-threshold 0.5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from anygran.epimine_hierarchical_sgqa import (
    EpiMineActionAnalyzer,
    EpiMineEpisodeGenerator,
    EpiMineHierarchicalEvaluator,
    GPT5Mini,
    GPT5,
    build_background_dataset,
    load_episodes_cache,
)
from anygran.hierarchical_sgqa import BaselineSGQAEvaluator


# Path to SGQA dataset
SGQA_PATH = Path(__file__).parent.parent / "resource" / "dataset" / "understanding" / "sgqa.jsonl"


def load_sgqa_data(limit: int = None) -> List[Dict]:
    """Load SGQA dataset and flatten QA pairs."""
    qa_data = []
    sample_count = 0

    with open(SGQA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sample_count += 1

                for qa_pair in item["qa_pairs"]:
                    qa_data.append({
                        "data_id": item["data_id"],
                        "doc_index": item["doc_index"],
                        "text_part_index": item["text_part_index"],
                        "context_graphs": item["context_graphs"],
                        "question": qa_pair["Q"],
                        "answer": qa_pair["A"],
                    })

                if limit and sample_count >= limit:
                    break

    return qa_data


def save_results(results: Dict, filename: str):
    """Save results to JSON file."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename

    # Add metadata
    output = {
        "task": "EpiMine-Hierarchical-SGQA",
        "metric": "Exact Match (EM)",
        "timestamp": datetime.now().isoformat(),
        **results
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {filepath}")
    return filepath


def print_comparison(baseline_results: Dict, epimine_results: Dict):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs EpiMine Hierarchical SGQA")
    print("=" * 70)

    baseline_em = baseline_results["exact_match_percent"]
    epimine_em = epimine_results["exact_match_percent"]
    diff = epimine_em - baseline_em

    print(f"\n{'Metric':<30} {'Baseline':<15} {'EpiMine':<15} {'Diff':<10}")
    print("-" * 70)
    print(f"{'Exact Match (%)':<30} {baseline_em:<15.2f} {epimine_em:<15.2f} {diff:+.2f}")
    print(f"{'Correct':<30} {baseline_results['correct']:<15} {epimine_results['correct']:<15}")
    print(f"{'Total Questions':<30} {baseline_results['total_questions']:<15}")

    # Per-question comparison
    print("\n" + "-" * 70)
    print("Per-Question Analysis (first 10 differences):")
    print("-" * 70)

    baseline_by_q = {r["question"]: r for r in baseline_results["results"]}
    epimine_by_q = {r["question"]: r for r in epimine_results["results"]}

    diff_count = 0
    improved = 0
    regressed = 0

    for q, baseline_r in baseline_by_q.items():
        epimine_r = epimine_by_q.get(q)
        if epimine_r and baseline_r["exact_match"] != epimine_r["exact_match"]:
            diff_count += 1
            if epimine_r["exact_match"]:
                improved += 1
                status = "IMPROVED"
            else:
                regressed += 1
                status = "REGRESSED"

            if diff_count <= 10:
                print(f"\n[{status}] Q: {q[:60]}...")
                print(f"  Ground Truth: {baseline_r['ground_truth']}")
                print(f"  Baseline:     {baseline_r['prediction']}")
                print(f"  EpiMine:      {epimine_r['prediction']}")

    print(f"\n\nTotal questions with different outcomes: {diff_count}")
    print(f"  Improved: {improved}")
    print(f"  Regressed: {regressed}")
    print("=" * 70)


def print_episode_stats(episodes_cache: Dict[str, Dict]):
    """Print statistics about detected episodes."""
    print("\n" + "=" * 70)
    print("EpiMine Episode Detection Statistics")
    print("=" * 70)

    total_samples = len(episodes_cache)
    total_episodes = 0
    episode_counts = []

    for data_id, hierarchy in episodes_cache.items():
        num_episodes = len(hierarchy.get("episodes", []))
        total_episodes += num_episodes
        episode_counts.append(num_episodes)

    avg_episodes = total_episodes / total_samples if total_samples > 0 else 0
    min_episodes = min(episode_counts) if episode_counts else 0
    max_episodes = max(episode_counts) if episode_counts else 0

    print(f"\nTotal samples: {total_samples}")
    print(f"Total episodes detected: {total_episodes}")
    print(f"Average episodes per sample: {avg_episodes:.2f}")
    print(f"Min episodes: {min_episodes}")
    print(f"Max episodes: {max_episodes}")

    # Sample episode breakdown
    print("\nSample Episode Breakdown (first 5):")
    for i, (data_id, hierarchy) in enumerate(episodes_cache.items()):
        if i >= 5:
            break
        episodes = hierarchy.get("episodes", [])
        print(f"\n  {data_id[:20]}...")
        print(f"    Goal: {hierarchy.get('overall_goal', 'N/A')[:50]}...")
        for ep in episodes:
            print(f"    Episode {ep['episode_id']}: {ep['name']} - Actions {ep['time']['action_indices']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="EpiMine Hierarchical SGQA Evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (default: all)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to cached EpiMine episode hierarchies JSON"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate episodes, don't run QA evaluation"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (only run EpiMine)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt5-mini", "gpt5"],
        default="gpt5-mini",
        help="Model to use for QA evaluation (default: gpt5-mini)"
    )
    parser.add_argument(
        "--cooccur-threshold",
        type=float,
        default=1.0,
        help="Standard deviations below mean for boundary detection (default: 1.0)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Don't use LLM for episode name/description generation"
    )

    args = parser.parse_args()

    # Load data
    print("Loading SGQA dataset...")
    data = load_sgqa_data(limit=args.limit)
    print(f"Loaded {len(data)} QA pairs from {args.limit or 'all'} samples")

    # Get unique sample count
    unique_samples = len(set(d["data_id"] for d in data))
    print(f"Unique samples: {unique_samples}")

    # Initialize model for QA evaluation
    if args.model == "gpt5":
        model = GPT5()
    else:
        model = GPT5Mini()
    print(f"Using model for QA: {model.__class__.__name__}")

    # Step 1: Build background dataset
    print("\nBuilding background dataset from all SGQA actions...")
    background = build_background_dataset(str(SGQA_PATH))
    print(f"Background dataset: {len(background)} action graphs")

    # Step 2: Initialize EpiMine analyzer
    print("Initializing EpiMine analyzer...")
    analyzer = EpiMineActionAnalyzer(background_dataset=background)

    # Step 3: Generate or load episode hierarchies
    if args.cache_path and Path(args.cache_path).exists():
        print(f"\nLoading cached episodes from: {args.cache_path}")
        episodes_cache = load_episodes_cache(args.cache_path)
    else:
        print("\nGenerating EpiMine episode hierarchies...")
        generator = EpiMineEpisodeGenerator(model=GPT5Mini())
        cache_filename = f"epimine_episodes_limit{args.limit or 'all'}.json"
        cache_path = Path(__file__).parent / "cache" / cache_filename

        episodes_cache = generator.batch_generate(
            data=data,
            analyzer=analyzer,
            output_path=str(cache_path),
            use_llm=not args.no_llm,
            threshold_std=args.cooccur_threshold,
        )

    # Print episode statistics
    print_episode_stats(episodes_cache)

    if args.generate_only:
        print("\n--generate-only flag set. Skipping QA evaluation.")
        return

    # Step 4: Run baseline evaluation
    baseline_results = None
    if not args.skip_baseline:
        baseline_evaluator = BaselineSGQAEvaluator(model=model)
        baseline_results = baseline_evaluator.evaluate(data, max_workers=args.workers)

        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_results(baseline_results, f"baseline_{args.model}_limit{args.limit or 'all'}_{timestamp}.json")

    # Step 5: Run EpiMine hierarchical evaluation
    epimine_evaluator = EpiMineHierarchicalEvaluator(
        model=model,
        episodes_cache=episodes_cache
    )
    epimine_results = epimine_evaluator.evaluate(data, max_workers=args.workers)

    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(epimine_results, f"epimine_{args.model}_limit{args.limit or 'all'}_{timestamp}.json")

    # Step 6: Print comparison
    if baseline_results:
        print_comparison(baseline_results, epimine_results)
    else:
        print(f"\n{'='*60}")
        print(f"EPIMINE HIERARCHICAL SGQA RESULTS")
        print(f"{'='*60}")
        print(f"Model:            {epimine_results['model']}")
        print(f"Total Questions:  {epimine_results['total_questions']}")
        print(f"Correct (EM):     {epimine_results['correct']}")
        print(f"Exact Match (EM): {epimine_results['exact_match_percent']}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
