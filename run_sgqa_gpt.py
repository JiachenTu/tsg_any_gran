"""
SGQA Evaluation Script for GPT Models
=====================================
Evaluates GPT-4o and GPT-4o-mini on Scene Graph Question Answering (SGQA) task.

Metric: Exact Match (EM) - case-insensitive string comparison

Usage:
    python run_sgqa_gpt.py --model gpt4o
    python run_sgqa_gpt.py --model gpt4o-mini
    python run_sgqa_gpt.py --model all
"""

import argparse
import json
import concurrent.futures
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from langchain_core.prompts import PromptTemplate

from models.models import (
    GPT4o,
    GPT4oMini,
    GPT41,
    GPT41Mini,
    GPT41Nano,
    GPT5,
    GPT5Mini,
    GPT5Nano,
    GPT5Pro,
    O1,
    O1Pro,
    O3,
    O3Mini,
    O4Mini,
)
from utils.path import get_project_path, load_prompt


class SGQAEvaluator:
    """SGQA Evaluator for GPT models with Exact Match metric."""

    def __init__(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

        # Load prompt template
        prompt = load_prompt("sgqa.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["scene_graph", "question"],
            template=prompt,
        )

    def invoke(self, scene_graph: str, question: str) -> str:
        """Invoke model and extract answer from brackets."""
        prompt = self.prompt_template.format(
            scene_graph=scene_graph,
            question=question,
        )
        response = self.model.invoke(prompt)

        # Extract answer from [brackets]
        answer = re.findall(r"\[(.*?)\]", response)
        return answer[0] if answer else response

    def process_single_question(self, data: Dict) -> Dict:
        """Process a single QA pair and return result."""
        prediction = self.invoke(
            scene_graph=str(data["context_graphs"]),
            question=data["question"]
        )

        # Exact Match (EM): case-insensitive comparison
        is_correct = prediction.lower().strip() == data["answer"].lower().strip()

        return {
            "data_id": data["data_id"],
            "question": data["question"],
            "ground_truth": data["answer"],
            "prediction": prediction,
            "exact_match": is_correct,
        }

    def evaluate(self, data: List[Dict], max_workers: int = 5) -> Dict:
        """Run evaluation on all data and return metrics."""
        results = []
        total_correct = 0
        total_questions = len(data)

        print(f"\n{'='*60}")
        print(f"Running SGQA Evaluation with {self.model_name}")
        print(f"Total questions: {total_questions}")
        print(f"{'='*60}\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_single_question, item): idx
                for idx, item in enumerate(data)
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                result = future.result()
                results.append(result)

                if result["exact_match"]:
                    total_correct += 1

                # Progress update
                processed = len(results)
                current_em = (total_correct / processed) * 100
                print(f"\r[{self.model_name}] {processed}/{total_questions} processed | "
                      f"Current EM: {current_em:.1f}%", end="", flush=True)

        print()  # New line after progress

        # Calculate final Exact Match
        em_score = (total_correct / total_questions) * 100 if total_questions > 0 else 0

        return {
            "model": self.model_name,
            "total_questions": total_questions,
            "correct": total_correct,
            "exact_match_percent": round(em_score, 2),
            "results": results,
        }


def load_sgqa_data() -> List[Dict]:
    """Load SGQA dataset and flatten QA pairs."""
    data_path = (
        Path(get_project_path())
        / "resource"
        / "dataset"
        / "understanding"
        / "sgqa.jsonl"
    )

    qa_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                for qa_pair in item["qa_pairs"]:
                    qa_data.append({
                        "data_id": item["data_id"],
                        "doc_index": item["doc_index"],
                        "text_part_index": item["text_part_index"],
                        "context_graphs": item["context_graphs"],
                        "question": qa_pair["Q"],
                        "answer": qa_pair["A"],
                    })

    return qa_data


def save_results(results: Dict, model_name: str):
    """Save results to JSON file."""
    results_dir = Path(get_project_path()) / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sgqa_{model_name.lower()}_{timestamp}.json"
    filepath = results_dir / filename

    # Add metadata
    output = {
        "task": "SGQA",
        "metric": "Exact Match (EM)",
        "timestamp": datetime.now().isoformat(),
        **results
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")
    return filepath


def print_summary(results: Dict):
    """Print evaluation summary."""
    print(f"\n{'='*60}")
    print("SGQA EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model:            {results['model']}")
    print(f"Total Questions:  {results['total_questions']}")
    print(f"Correct (EM):     {results['correct']}")
    print(f"Exact Match (EM): {results['exact_match_percent']}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="SGQA Evaluation for GPT Models")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gpt4o", "gpt4o-mini",
            "gpt41", "gpt41-mini", "gpt41-nano",
            "gpt5", "gpt5-mini", "gpt5-nano", "gpt5-pro",
            "o1", "o1-pro", "o3", "o3-mini", "o4-mini",
            "all"
        ],
        default="gpt4o",
        help="Model to evaluate (default: gpt4o)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    args = parser.parse_args()

    # Load data
    print("Loading SGQA dataset...")
    data = load_sgqa_data()
    print(f"Loaded {len(data)} QA pairs")

    # Model mapping
    model_map = {
        "gpt4o": ("GPT4o", GPT4o),
        "gpt4o-mini": ("GPT4oMini", GPT4oMini),
        "gpt41": ("GPT41", GPT41),
        "gpt41-mini": ("GPT41Mini", GPT41Mini),
        "gpt41-nano": ("GPT41Nano", GPT41Nano),
        "gpt5": ("GPT5", GPT5),
        "gpt5-mini": ("GPT5Mini", GPT5Mini),
        "gpt5-nano": ("GPT5Nano", GPT5Nano),
        "gpt5-pro": ("GPT5Pro", GPT5Pro),
        "o1": ("O1", O1),
        "o1-pro": ("O1Pro", O1Pro),
        "o3": ("O3", O3),
        "o3-mini": ("O3Mini", O3Mini),
        "o4-mini": ("O4Mini", O4Mini),
    }

    # Select models
    models_to_run = []
    if args.model == "all":
        for name, model_class in model_map.values():
            models_to_run.append((name, model_class()))
    else:
        name, model_class = model_map[args.model]
        models_to_run.append((name, model_class()))

    # Run evaluation
    all_results = []
    for model_name, model in models_to_run:
        evaluator = SGQAEvaluator(model)
        results = evaluator.evaluate(data, max_workers=args.workers)

        print_summary(results)

        if not args.no_save:
            save_results(results, model_name)

        all_results.append(results)

    # Print comparison if multiple models
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        for r in all_results:
            print(f"{r['model']}: {r['exact_match_percent']}% EM")
        print("="*60)


if __name__ == "__main__":
    main()
