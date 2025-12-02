"""
Hierarchical SGQA Evaluation Runner
===================================
Main script to run multi-granular SGQA evaluation.

Usage:
    # Test on first 5 samples
    python anygran/run_hierarchical_sgqa.py --limit 5

    # Run full evaluation
    python anygran/run_hierarchical_sgqa.py

    # Generate events only (no QA evaluation)
    python anygran/run_hierarchical_sgqa.py --generate-only --limit 5

    # Use cached events
    python anygran/run_hierarchical_sgqa.py --cache-path anygran/cache/events.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from anygran.event_generator import EventGenerator, load_events_cache
from anygran.hierarchical_sgqa import HierarchicalSGQAEvaluator, BaselineSGQAEvaluator, UnifiedHierarchicalEvaluator

# Import GPT5Mini directly to avoid loading all models
from langchain_openai import ChatOpenAI
from utils.config import load_config, get_config_file_path


class GPT5Mini:
    """GPT-5-mini model wrapper."""
    def __init__(self, temperature=0.1):
        config_path = get_config_file_path()
        config = load_config(config_path)
        self.openai = ChatOpenAI(
            api_key=config["openai"]["key"],
            model_name="gpt-5-mini",
            temperature=temperature,
        )

    def invoke(self, message):
        response = self.openai.invoke(message)
        return response.content.strip()


class GPT5:
    """GPT-5 model wrapper."""
    def __init__(self, temperature=0.1):
        config_path = get_config_file_path()
        config = load_config(config_path)
        self.openai = ChatOpenAI(
            api_key=config["openai"]["key"],
            model_name="gpt-5",
            temperature=temperature,
        )

    def invoke(self, message):
        response = self.openai.invoke(message)
        return response.content.strip()


def load_sgqa_data(limit: int = None) -> List[Dict]:
    """Load SGQA dataset and flatten QA pairs."""
    data_path = Path(__file__).parent.parent / "resource" / "dataset" / "understanding" / "sgqa.jsonl"

    qa_data = []
    sample_count = 0

    with open(data_path, "r", encoding="utf-8") as f:
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
        "task": "Hierarchical-SGQA",
        "metric": "Exact Match (EM)",
        "timestamp": datetime.now().isoformat(),
        **results
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {filepath}")
    return filepath


def print_comparison(baseline_results: Dict, hierarchical_results: Dict):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Hierarchical SGQA")
    print("=" * 70)

    baseline_em = baseline_results["exact_match_percent"]
    hier_em = hierarchical_results["exact_match_percent"]
    diff = hier_em - baseline_em

    print(f"\n{'Metric':<30} {'Baseline':<15} {'Hierarchical':<15} {'Diff':<10}")
    print("-" * 70)
    print(f"{'Exact Match (%)':<30} {baseline_em:<15.2f} {hier_em:<15.2f} {diff:+.2f}")
    print(f"{'Correct':<30} {baseline_results['correct']:<15} {hierarchical_results['correct']:<15}")
    print(f"{'Total Questions':<30} {baseline_results['total_questions']:<15}")

    # Per-question comparison
    print("\n" + "-" * 70)
    print("Per-Question Analysis (first 10 differences):")
    print("-" * 70)

    baseline_by_q = {r["question"]: r for r in baseline_results["results"]}
    hier_by_q = {r["question"]: r for r in hierarchical_results["results"]}

    diff_count = 0
    for q, baseline_r in baseline_by_q.items():
        hier_r = hier_by_q.get(q)
        if hier_r and baseline_r["exact_match"] != hier_r["exact_match"]:
            diff_count += 1
            if diff_count <= 10:
                status = "IMPROVED" if hier_r["exact_match"] else "REGRESSED"
                print(f"\n[{status}] Q: {q[:60]}...")
                print(f"  Ground Truth: {baseline_r['ground_truth']}")
                print(f"  Baseline:     {baseline_r['prediction']}")
                print(f"  Hierarchical: {hier_r['prediction']}")

    print(f"\n\nTotal questions with different outcomes: {diff_count}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Hierarchical SGQA Evaluation")
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
        help="Path to cached event hierarchies JSON"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate events, don't run QA evaluation"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (only run hierarchical)"
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
        "--unified",
        action="store_true",
        help="Use unified/interleaved hierarchical format (v1) instead of separated format (v0)"
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

    # Event generation always uses GPT5Mini (for consistency)
    event_model = GPT5Mini()
    print(f"Using model for event generation: {event_model.__class__.__name__}")

    # Step 1: Generate or load event hierarchies
    if args.cache_path and Path(args.cache_path).exists():
        print(f"\nLoading cached events from: {args.cache_path}")
        events_cache = load_events_cache(args.cache_path)
    else:
        print("\nGenerating event hierarchies...")
        generator = EventGenerator(model=event_model)  # Always use GPT5Mini for events
        cache_filename = f"events_limit{args.limit or 'all'}.json"
        cache_path = Path(__file__).parent / "cache" / cache_filename
        events_cache = generator.batch_generate(data, output_path=str(cache_path), limit=args.limit)

    if args.generate_only:
        print("\n--generate-only flag set. Skipping QA evaluation.")
        return

    # Step 2: Run baseline evaluation
    baseline_results = None
    if not args.skip_baseline:
        baseline_evaluator = BaselineSGQAEvaluator(model=model)
        baseline_results = baseline_evaluator.evaluate(data, max_workers=args.workers)

        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_results(baseline_results, f"baseline_{args.model}_limit{args.limit or 'all'}_{timestamp}.json")

    # Step 3: Run hierarchical evaluation
    if args.unified:
        # Use unified/interleaved format (v1)
        hier_evaluator = UnifiedHierarchicalEvaluator(model=model, events_cache=events_cache)
        result_prefix = "unified"
    else:
        # Use separated format (v0)
        hier_evaluator = HierarchicalSGQAEvaluator(model=model, events_cache=events_cache)
        result_prefix = "hierarchical"

    hier_results = hier_evaluator.evaluate(data, max_workers=args.workers)

    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(hier_results, f"{result_prefix}_{args.model}_limit{args.limit or 'all'}_{timestamp}.json")

    # Step 4: Print comparison
    if baseline_results:
        print_comparison(baseline_results, hier_results)
    else:
        print(f"\n{'='*60}")
        print(f"{'UNIFIED' if args.unified else 'HIERARCHICAL'} SGQA RESULTS")
        print(f"{'='*60}")
        print(f"Model:            {hier_results['model']}")
        print(f"Total Questions:  {hier_results['total_questions']}")
        print(f"Correct (EM):     {hier_results['correct']}")
        print(f"Exact Match (EM): {hier_results['exact_match_percent']}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
