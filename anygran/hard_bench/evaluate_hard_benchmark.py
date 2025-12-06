"""
Evaluate Hard Benchmark
=======================
Evaluates GPT5 and GPT5-mini on SGQA-Hard and SGDS-Hard benchmarks.
"""

import json
import re
import sys
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils.config import load_config, get_config_file_path


# Model wrappers
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


# SGQA Evaluation
class SGQAHardEvaluator:
    """Evaluator for SGQA Hard benchmark."""

    def __init__(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

        # Baseline SGQA prompt
        self.prompt_template = PromptTemplate(
            input_variables=["scene_graph", "question"],
            template="""You are a highly advanced language model specialized in answering questions based on a given scene graph and question. Your task is to analyze the scene graph and provide the correct answer in a single word. Your output must strictly follow the format [answer], and nothing else should be printed. Ensure that your answer is concise, accurate, and matches the format exactly.

Scene Graph: {scene_graph}
Question: {question}
""",
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
        return answer[0] if answer else response.strip()

    def process_single_case(self, case: Dict) -> Dict:
        """Process a single SGQA case."""
        prediction = self.invoke(
            scene_graph=str(case["context_graphs"]),
            question=case["question"]
        )

        # Exact Match (EM): case-insensitive comparison
        is_correct = prediction.lower().strip() == case["ground_truth"].lower().strip()

        return {
            "data_id": case["data_id"],
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "prediction": prediction,
            "exact_match": is_correct,
            "difficulty": case.get("difficulty", "unknown"),
            "error_category": case.get("error_category", "unknown"),
        }

    def evaluate(self, cases: List[Dict], max_workers: int = 5) -> Dict:
        """Run evaluation on all cases."""
        results = []
        total_correct = 0
        total_cases = len(cases)

        # Track by difficulty and category
        by_difficulty = {}
        by_category = {}

        print(f"\n{'='*60}")
        print(f"Running SGQA-Hard Evaluation with {self.model_name}")
        print(f"Total cases: {total_cases}")
        print(f"{'='*60}\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_single_case, case): idx
                for idx, case in enumerate(cases)
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

                if result["exact_match"]:
                    total_correct += 1

                # Track by difficulty
                diff = result["difficulty"]
                if diff not in by_difficulty:
                    by_difficulty[diff] = {"total": 0, "correct": 0}
                by_difficulty[diff]["total"] += 1
                if result["exact_match"]:
                    by_difficulty[diff]["correct"] += 1

                # Track by category
                cat = result["error_category"]
                if cat not in by_category:
                    by_category[cat] = {"total": 0, "correct": 0}
                by_category[cat]["total"] += 1
                if result["exact_match"]:
                    by_category[cat]["correct"] += 1

                # Progress update
                processed = len(results)
                current_em = (total_correct / processed) * 100
                print(f"\r[{self.model_name}] {processed}/{total_cases} | EM: {current_em:.1f}%", end="", flush=True)

        print()

        # Calculate accuracies
        for d in by_difficulty.values():
            d["accuracy"] = round(d["correct"] / d["total"] * 100, 2) if d["total"] > 0 else 0
        for c in by_category.values():
            c["accuracy"] = round(c["correct"] / c["total"] * 100, 2) if c["total"] > 0 else 0

        accuracy = (total_correct / total_cases) * 100 if total_cases > 0 else 0

        return {
            "task": "SGQA-Hard",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_cases": total_cases,
            "correct": total_correct,
            "accuracy_percent": round(accuracy, 2),
            "by_difficulty": by_difficulty,
            "by_category": by_category,
            "results": results,
        }


# SGDS Evaluation
class SGDSHardEvaluator:
    """Evaluator for SGDS Hard benchmark."""

    def __init__(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

        # Load SGDS prompt
        prompt_path = Path(__file__).parent.parent.parent / "resource" / "prompts" / "sgds.txt"
        with open(prompt_path, "r") as f:
            prompt = f.read()

        self.prompt_template = PromptTemplate(
            input_variables=["sentences", "triplet", "context"],
            template=prompt,
        )

    def parse_prediction(self, response: str) -> int:
        """Parse letter prediction to position index."""
        pattern = r"\[([A-E])\]|\b([A-E])\b"
        match = re.search(pattern, response)
        if match:
            letter = match.group(1) or match.group(2)
            return ord(letter) - 65  # Convert 'A' to 0, 'B' to 1, etc.
        return None

    def process_single_case(self, case: Dict) -> Dict:
        """Process a single SGDS case."""
        # Format variations
        variations_str = "\n".join([
            f"{chr(65 + i)}: {var}"
            for i, var in enumerate(case["variations"])
        ])

        prompt = self.prompt_template.format(
            sentences=variations_str,
            triplet=case["triplet"],
            context=case["context_graphs"],
        )

        response = self.model.invoke(prompt)
        prediction = self.parse_prediction(response)

        is_correct = prediction == case["position"] if prediction is not None else False

        return {
            "target_sentence": case["target_sentence"],
            "position": case["position"],
            "prediction": prediction,
            "prediction_letter": chr(65 + prediction) if prediction is not None else None,
            "is_correct": is_correct,
            "hard_reason": case.get("hard_reason", "unknown"),
            "context_length": case.get("context_length", 0),
            "complexity_score": case.get("complexity_score", 0),
        }

    def evaluate(self, cases: List[Dict], max_workers: int = 5) -> Dict:
        """Run evaluation on all cases."""
        results = []
        total_correct = 0
        total_cases = len(cases)

        # Track by hard reason
        by_reason = {}

        print(f"\n{'='*60}")
        print(f"Running SGDS-Hard Evaluation with {self.model_name}")
        print(f"Total cases: {total_cases}")
        print(f"{'='*60}\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_single_case, case): idx
                for idx, case in enumerate(cases)
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

                if result["is_correct"]:
                    total_correct += 1

                # Track by hard reason
                for reason in result["hard_reason"].split("|"):
                    if reason not in by_reason:
                        by_reason[reason] = {"total": 0, "correct": 0}
                    by_reason[reason]["total"] += 1
                    if result["is_correct"]:
                        by_reason[reason]["correct"] += 1

                # Progress update
                processed = len(results)
                current_acc = (total_correct / processed) * 100
                print(f"\r[{self.model_name}] {processed}/{total_cases} | Acc: {current_acc:.1f}%", end="", flush=True)

        print()

        # Calculate accuracies
        for r in by_reason.values():
            r["accuracy"] = round(r["correct"] / r["total"] * 100, 2) if r["total"] > 0 else 0

        accuracy = (total_correct / total_cases) * 100 if total_cases > 0 else 0

        return {
            "task": "SGDS-Hard",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_cases": total_cases,
            "correct": total_correct,
            "accuracy_percent": round(accuracy, 2),
            "by_hard_reason": by_reason,
            "results": results,
        }


def load_sgqa_hard() -> List[Dict]:
    """Load SGQA hard benchmark."""
    path = Path(__file__).parent / "sgqa_hard.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["cases"]


def load_sgds_hard() -> List[Dict]:
    """Load SGDS hard benchmark."""
    path = Path(__file__).parent / "sgds_hard.jsonl"
    cases = []
    with open(path, "r") as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


def save_results(results: Dict, filename: str):
    """Save results to JSON file."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load benchmarks
    print("Loading hard benchmarks...")
    sgqa_cases = load_sgqa_hard()
    sgds_cases = load_sgds_hard()
    print(f"SGQA Hard: {len(sgqa_cases)} cases")
    print(f"SGDS Hard: {len(sgds_cases)} cases")

    # Initialize models
    print("\nInitializing models...")
    gpt5 = GPT5()
    gpt5_mini = GPT5Mini()

    all_results = {}

    # SGQA Hard - GPT5
    print("\n" + "="*70)
    print("SGQA-Hard with GPT5")
    print("="*70)
    sgqa_evaluator_gpt5 = SGQAHardEvaluator(gpt5)
    sgqa_gpt5_results = sgqa_evaluator_gpt5.evaluate(sgqa_cases)
    save_results(sgqa_gpt5_results, f"sgqa_hard_gpt5_{timestamp}.json")
    all_results["sgqa_gpt5"] = {
        "accuracy": sgqa_gpt5_results["accuracy_percent"],
        "correct": sgqa_gpt5_results["correct"],
        "total": sgqa_gpt5_results["total_cases"]
    }

    # SGQA Hard - GPT5Mini
    print("\n" + "="*70)
    print("SGQA-Hard with GPT5Mini")
    print("="*70)
    sgqa_evaluator_mini = SGQAHardEvaluator(gpt5_mini)
    sgqa_mini_results = sgqa_evaluator_mini.evaluate(sgqa_cases)
    save_results(sgqa_mini_results, f"sgqa_hard_gpt5mini_{timestamp}.json")
    all_results["sgqa_gpt5mini"] = {
        "accuracy": sgqa_mini_results["accuracy_percent"],
        "correct": sgqa_mini_results["correct"],
        "total": sgqa_mini_results["total_cases"]
    }

    # SGDS Hard - GPT5
    print("\n" + "="*70)
    print("SGDS-Hard with GPT5")
    print("="*70)
    sgds_evaluator_gpt5 = SGDSHardEvaluator(gpt5)
    sgds_gpt5_results = sgds_evaluator_gpt5.evaluate(sgds_cases)
    save_results(sgds_gpt5_results, f"sgds_hard_gpt5_{timestamp}.json")
    all_results["sgds_gpt5"] = {
        "accuracy": sgds_gpt5_results["accuracy_percent"],
        "correct": sgds_gpt5_results["correct"],
        "total": sgds_gpt5_results["total_cases"]
    }

    # SGDS Hard - GPT5Mini
    print("\n" + "="*70)
    print("SGDS-Hard with GPT5Mini")
    print("="*70)
    sgds_evaluator_mini = SGDSHardEvaluator(gpt5_mini)
    sgds_mini_results = sgds_evaluator_mini.evaluate(sgds_cases)
    save_results(sgds_mini_results, f"sgds_hard_gpt5mini_{timestamp}.json")
    all_results["sgds_gpt5mini"] = {
        "accuracy": sgds_mini_results["accuracy_percent"],
        "correct": sgds_mini_results["correct"],
        "total": sgds_mini_results["total_cases"]
    }

    # Save summary
    summary = {
        "timestamp": timestamp,
        "results": all_results
    }
    save_results(summary, "summary.json")

    # Print final summary
    print("\n" + "="*70)
    print("HARD BENCHMARK EVALUATION SUMMARY")
    print("="*70)
    print(f"\nSGQA-Hard (88 cases):")
    print(f"  GPT5:      {all_results['sgqa_gpt5']['correct']}/{all_results['sgqa_gpt5']['total']} = {all_results['sgqa_gpt5']['accuracy']}%")
    print(f"  GPT5-mini: {all_results['sgqa_gpt5mini']['correct']}/{all_results['sgqa_gpt5mini']['total']} = {all_results['sgqa_gpt5mini']['accuracy']}%")
    print(f"\nSGDS-Hard (174 cases):")
    print(f"  GPT5:      {all_results['sgds_gpt5']['correct']}/{all_results['sgds_gpt5']['total']} = {all_results['sgds_gpt5']['accuracy']}%")
    print(f"  GPT5-mini: {all_results['sgds_gpt5mini']['correct']}/{all_results['sgds_gpt5mini']['total']} = {all_results['sgds_gpt5mini']['accuracy']}%")


if __name__ == "__main__":
    main()
