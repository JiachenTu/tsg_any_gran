#!/usr/bin/env python3
"""
Run EpiMine Hierarchical SGQA on Hard Benchmark
================================================
Applies EpiMine episode detection to the hard benchmark cases.

Usage:
    python anygran/hard_bench/run_epimine_hard.py
    python anygran/hard_bench/run_epimine_hard.py --model gpt5
    python anygran/hard_bench/run_epimine_hard.py --skip-baseline
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anygran.epimine_hierarchical_sgqa import (
    EpiMineActionAnalyzer,
    EpiMineEpisodeGenerator,
    EpiMineHierarchicalEvaluator,
    GPT5Mini,
    GPT5,
    build_background_dataset,
)

# Paths
HARD_BENCH_PATH = Path(__file__).parent / "sgqa_hard.json"
SGQA_PATH = Path(__file__).parent.parent.parent / "resource" / "dataset" / "understanding" / "sgqa.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_hard_benchmark() -> list:
    """Load hard benchmark cases."""
    with open(HARD_BENCH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def run_baseline_evaluation(cases: list, model) -> dict:
    """Run baseline evaluation (no EpiMine hierarchy)."""
    from langchain_core.prompts import PromptTemplate
    import re

    prompt_template = PromptTemplate(
        input_variables=["scene_graph", "question"],
        template="""You are a highly advanced language model specialized in answering questions based on a given scene graph and question. Your task is to analyze the scene graph and provide the correct answer in a single word. Your output must strictly follow the format [answer], and nothing else should be printed. Ensure that your answer is concise, accurate, and matches the format exactly.

Scene Graph: {scene_graph}
Question: {question}
""",
    )

    results = []
    correct = 0
    model_name = model.__class__.__name__

    print(f"\n{'='*60}")
    print(f"Running Baseline Evaluation with {model_name}")
    print(f"Total cases: {len(cases)}")
    print(f"{'='*60}\n")

    for i, case in enumerate(cases):
        prompt = prompt_template.format(
            scene_graph=str(case["context_graphs"]),
            question=case["question"]
        )
        response = model.invoke(prompt)
        answer = re.findall(r"\[(.*?)\]", response)
        pred = answer[0] if answer else response.strip()
        is_correct = pred.lower().strip() == case["ground_truth"].lower().strip()

        if is_correct:
            correct += 1

        results.append({
            "data_id": case["data_id"],
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "prediction": pred,
            "exact_match": is_correct,
        })

        print(f"\r[Baseline-{model_name}] {i+1}/{len(cases)} | EM: {correct/(i+1)*100:.1f}%", end="", flush=True)

    print()

    return {
        "model": f"Baseline-{model_name}",
        "total_cases": len(cases),
        "correct": correct,
        "accuracy_percent": round(correct / len(cases) * 100, 2),
        "results": results,
    }


def run_epimine_evaluation(cases: list, model, analyzer, threshold_std: float = 1.0) -> dict:
    """Run EpiMine hierarchical evaluation."""
    # Convert hard benchmark format to expected format
    data = []
    for case in cases:
        data.append({
            "data_id": case["data_id"],
            "context_graphs": case["context_graphs"],
            "question": case["question"],
            "answer": case["ground_truth"],
        })

    # Generate episode hierarchies
    print("\nGenerating EpiMine episode hierarchies...")
    generator = EpiMineEpisodeGenerator(model=GPT5Mini())
    episodes_cache = generator.batch_generate(
        data=data,
        analyzer=analyzer,
        use_llm=True,
        threshold_std=threshold_std,
    )

    # Print episode stats
    total_episodes = sum(len(h.get("episodes", [])) for h in episodes_cache.values())
    avg_episodes = total_episodes / len(episodes_cache) if episodes_cache else 0
    print(f"Generated {total_episodes} episodes across {len(episodes_cache)} samples (avg: {avg_episodes:.2f})")

    # Run evaluation
    evaluator = EpiMineHierarchicalEvaluator(
        model=model,
        episodes_cache=episodes_cache,
    )

    results = []
    correct = 0
    model_name = model.__class__.__name__

    print(f"\n{'='*60}")
    print(f"Running EpiMine Hierarchical Evaluation with {model_name}")
    print(f"Total cases: {len(cases)}")
    print(f"{'='*60}\n")

    for i, item in enumerate(data):
        prediction = evaluator.invoke(
            data_id=item["data_id"],
            context_graphs=item["context_graphs"],
            question=item["question"],
        )
        is_correct = prediction.lower().strip() == item["answer"].lower().strip()

        if is_correct:
            correct += 1

        results.append({
            "data_id": item["data_id"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "prediction": prediction,
            "exact_match": is_correct,
        })

        print(f"\r[EpiMine-{model_name}] {i+1}/{len(data)} | EM: {correct/(i+1)*100:.1f}%", end="", flush=True)

    print()

    return {
        "model": f"EpiMine-{model_name}",
        "total_cases": len(cases),
        "correct": correct,
        "accuracy_percent": round(correct / len(cases) * 100, 2),
        "results": results,
        "episodes_cache": episodes_cache,
    }


def print_comparison(baseline_results: dict, epimine_results: dict):
    """Print comparison between baseline and EpiMine."""
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs EpiMine on SGQA-Hard")
    print("=" * 70)

    baseline_acc = baseline_results["accuracy_percent"]
    epimine_acc = epimine_results["accuracy_percent"]
    diff = epimine_acc - baseline_acc

    print(f"\n{'Method':<30} {'Accuracy':<15} {'Correct':<15} {'Diff':<10}")
    print("-" * 70)
    print(f"{'Baseline':<30} {baseline_acc:<15.2f} {baseline_results['correct']:<15}")
    print(f"{'EpiMine':<30} {epimine_acc:<15.2f} {epimine_results['correct']:<15} {diff:+.2f}")
    print(f"\n{'Total Cases':<30} {baseline_results['total_cases']}")

    # Per-question analysis
    baseline_by_id = {r["data_id"]: r for r in baseline_results["results"]}
    epimine_by_id = {r["data_id"]: r for r in epimine_results["results"]}

    improved = 0
    regressed = 0

    for data_id, baseline_r in baseline_by_id.items():
        epimine_r = epimine_by_id.get(data_id)
        if epimine_r:
            if not baseline_r["exact_match"] and epimine_r["exact_match"]:
                improved += 1
            elif baseline_r["exact_match"] and not epimine_r["exact_match"]:
                regressed += 1

    print(f"\nQuestions improved by EpiMine: {improved}")
    print(f"Questions regressed by EpiMine: {regressed}")
    print("=" * 70)


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / filename

    output = {
        "task": "EpiMine-SGQA-Hard",
        "timestamp": datetime.now().isoformat(),
        **{k: v for k, v in results.items() if k != "episodes_cache"},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="EpiMine SGQA Hard Benchmark Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt5-mini", "gpt5"],
        default="gpt5-mini",
        help="Model to use for QA evaluation (default: gpt5-mini)"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (only run EpiMine)"
    )
    parser.add_argument(
        "--cooccur-threshold",
        type=float,
        default=1.0,
        help="Standard deviations below mean for boundary detection (default: 1.0)"
    )

    args = parser.parse_args()

    # Load hard benchmark
    print("Loading SGQA Hard Benchmark...")
    cases = load_hard_benchmark()
    print(f"Loaded {len(cases)} hard cases")

    # Initialize model
    if args.model == "gpt5":
        model = GPT5()
    else:
        model = GPT5Mini()
    print(f"Using model: {model.__class__.__name__}")

    # Build background dataset
    print("\nBuilding background dataset from all SGQA actions...")
    background = build_background_dataset(str(SGQA_PATH))
    print(f"Background dataset: {len(background)} action graphs")

    # Initialize analyzer
    analyzer = EpiMineActionAnalyzer(background_dataset=background)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run baseline evaluation
    baseline_results = None
    if not args.skip_baseline:
        baseline_results = run_baseline_evaluation(cases, model)
        save_results(baseline_results, f"hard_baseline_{args.model}_{timestamp}.json")

    # Run EpiMine evaluation
    epimine_results = run_epimine_evaluation(
        cases, model, analyzer,
        threshold_std=args.cooccur_threshold
    )
    save_results(epimine_results, f"hard_epimine_{args.model}_{timestamp}.json")

    # Print comparison
    if baseline_results:
        print_comparison(baseline_results, epimine_results)
    else:
        print(f"\n{'='*60}")
        print(f"EPIMINE SGQA-HARD RESULTS")
        print(f"{'='*60}")
        print(f"Model:     {epimine_results['model']}")
        print(f"Total:     {epimine_results['total_cases']}")
        print(f"Correct:   {epimine_results['correct']}")
        print(f"Accuracy:  {epimine_results['accuracy_percent']}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
