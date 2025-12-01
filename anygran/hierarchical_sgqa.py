"""
Hierarchical SGQA Evaluator
===========================
Enhanced SGQA evaluator that uses multi-granular context:
- Level 3: Overall goal
- Level 2: Sub-events
- Level 1: Action graphs
- Level 0: Triplets
"""

import json
import re
import sys
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.prompts import PromptTemplate
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


def load_prompt(prompt_name: str) -> str:
    """Load prompt template from anygran/prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / prompt_name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


class HierarchicalSGQAEvaluator:
    """SGQA Evaluator with multi-granular hierarchical context."""

    def __init__(self, model=None, events_cache: Optional[Dict[str, Dict]] = None):
        """
        Initialize the hierarchical SGQA evaluator.

        Args:
            model: LLM model to use. Defaults to GPT5Mini.
            events_cache: Dict mapping data_id to event hierarchy.
        """
        self.model = model or GPT5Mini()
        self.model_name = self.model.__class__.__name__
        self.events_cache = events_cache or {}

        # Load prompt template
        prompt = load_prompt("hierarchical_sgqa.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["overall_goal", "sub_events", "scene_graph", "question"],
            template=prompt,
        )

    def invoke(self, data_id: str, scene_graph: str, question: str) -> str:
        """
        Invoke model with hierarchical context and extract answer.

        Args:
            data_id: Unique identifier for the sample.
            scene_graph: String representation of the scene graphs.
            question: The question to answer.

        Returns:
            Extracted answer string.
        """
        # Get event hierarchy from cache
        event_data = self.events_cache.get(data_id, {})
        overall_goal = event_data.get("overall_goal", "Activity sequence")
        sub_events = event_data.get("sub_events", [])

        # Format sub-events for the prompt
        sub_events_str = self._format_sub_events(sub_events)

        prompt = self.prompt_template.format(
            overall_goal=overall_goal,
            sub_events=sub_events_str,
            scene_graph=scene_graph,
            question=question,
        )

        response = self.model.invoke(prompt)

        # Extract answer from [brackets]
        answer = re.findall(r"\[(.*?)\]", response)
        return answer[0] if answer else response.strip()

    def _format_sub_events(self, sub_events: List[Dict]) -> str:
        """Format sub-events list for the prompt."""
        if not sub_events:
            return "No sub-events available"

        lines = []
        for i, event in enumerate(sub_events):
            name = event.get("name", f"Phase {i+1}")
            desc = event.get("description", "")
            indices = event.get("action_indices", [])
            lines.append(f"{i+1}. {name}: {desc} (Actions: {indices})")

        return "\n".join(lines)

    def process_single_question(self, data: Dict) -> Dict:
        """Process a single QA pair and return result."""
        prediction = self.invoke(
            data_id=data["data_id"],
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
        """
        Run evaluation on all data and return metrics.

        Args:
            data: List of QA items with data_id, context_graphs, question, answer.
            max_workers: Number of parallel workers.

        Returns:
            Dict with evaluation results and metrics.
        """
        results = []
        total_correct = 0
        total_questions = len(data)

        print(f"\n{'='*60}")
        print(f"Running Hierarchical SGQA Evaluation with {self.model_name}")
        print(f"Total questions: {total_questions}")
        print(f"Event cache size: {len(self.events_cache)} samples")
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
                print(f"\r[Hierarchical-{self.model_name}] {processed}/{total_questions} | "
                      f"EM: {current_em:.1f}%", end="", flush=True)

        print()  # New line after progress

        # Calculate final Exact Match
        em_score = (total_correct / total_questions) * 100 if total_questions > 0 else 0

        return {
            "model": f"Hierarchical-{self.model_name}",
            "total_questions": total_questions,
            "correct": total_correct,
            "exact_match_percent": round(em_score, 2),
            "results": results,
        }


class BaselineSGQAEvaluator:
    """Standard SGQA Evaluator without hierarchical context (for comparison)."""

    def __init__(self, model=None):
        """
        Initialize the baseline SGQA evaluator.

        Args:
            model: LLM model to use. Defaults to GPT5Mini.
        """
        self.model = model or GPT5Mini()
        self.model_name = self.model.__class__.__name__

        # Use the original SGQA prompt
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
        print(f"Running Baseline SGQA Evaluation with {self.model_name}")
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
                print(f"\r[Baseline-{self.model_name}] {processed}/{total_questions} | "
                      f"EM: {current_em:.1f}%", end="", flush=True)

        print()  # New line after progress

        # Calculate final Exact Match
        em_score = (total_correct / total_questions) * 100 if total_questions > 0 else 0

        return {
            "model": f"Baseline-{self.model_name}",
            "total_questions": total_questions,
            "correct": total_correct,
            "exact_match_percent": round(em_score, 2),
            "results": results,
        }
