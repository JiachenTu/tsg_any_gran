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


class UnifiedHierarchicalEvaluator:
    """
    SGQA Evaluator with unified/interleaved hierarchical context (v1).

    Key difference from HierarchicalSGQAEvaluator (v0):
    - v0: Separates overall goal, sub-events, and scene graphs into different sections
    - v1: Interleaves sub-events with their corresponding actions directly
    """

    def __init__(self, model=None, events_cache: Optional[Dict[str, Dict]] = None):
        """
        Initialize the unified hierarchical SGQA evaluator.

        Args:
            model: LLM model to use. Defaults to GPT5Mini.
            events_cache: Dict mapping data_id to event hierarchy.
        """
        self.model = model or GPT5Mini()
        self.model_name = self.model.__class__.__name__
        self.events_cache = events_cache or {}

        # Load unified prompt template
        prompt = load_prompt("unified_hierarchical.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["unified_timeline", "question"],
            template=prompt,
        )

    def _extract_action_verb(self, action_graph: List) -> str:
        """Extract the main action verb from an action graph."""
        for triplet in action_graph:
            if len(triplet) >= 3 and triplet[1] in ["verb", "verbs"]:
                return triplet[2]  # The action verb
        return "unknown"

    def _format_unified_timeline(
        self,
        overall_goal: str,
        sub_events: List[Dict],
        context_graphs: List
    ) -> str:
        """
        Format scene graph with hierarchical context included.

        Output format:
            Overall Goal: {goal}

            Phase 1: {name}
            {description}
            Actions: [Action 0 (verb): triplets, Action 1 (verb): triplets, ...]

            Phase 2: ...
        """
        lines = []

        # Add overall goal
        lines.append(f"Overall Goal: {overall_goal}")
        lines.append("")

        if not sub_events:
            # Fallback: just list all actions without phases
            lines.append("All Actions:")
            for i, action_graph in enumerate(context_graphs):
                verb = self._extract_action_verb(action_graph)
                triplets_str = " ".join(str(t) for t in action_graph)
                lines.append(f"- Action {i} ({verb}): {triplets_str}")
            return "\n".join(lines)

        for phase_idx, event in enumerate(sub_events):
            name = event.get("name", f"Phase {phase_idx+1}")
            desc = event.get("description", "")
            action_indices = event.get("action_indices", [])

            lines.append(f"Phase {phase_idx+1}: {name}")
            lines.append(f"{desc}")
            lines.append("Actions in this phase:")

            for action_idx in action_indices:
                if action_idx < len(context_graphs):
                    action_graph = context_graphs[action_idx]
                    verb = self._extract_action_verb(action_graph)
                    triplets_str = " ".join(str(t) for t in action_graph)
                    lines.append(f"- Action {action_idx} ({verb}): {triplets_str}")

            lines.append("")  # Blank line between phases

        return "\n".join(lines)

    def invoke(self, data_id: str, context_graphs: List, question: str) -> str:
        """
        Invoke model with unified hierarchical context and extract answer.

        Args:
            data_id: Unique identifier for the sample.
            context_graphs: List of action graphs (not string).
            question: The question to answer.

        Returns:
            Extracted answer string.
        """
        # Get event hierarchy from cache
        event_data = self.events_cache.get(data_id, {})
        overall_goal = event_data.get("overall_goal", "Activity sequence")
        sub_events = event_data.get("sub_events", [])

        # Format unified timeline (includes overall goal)
        unified_timeline = self._format_unified_timeline(overall_goal, sub_events, context_graphs)

        prompt = self.prompt_template.format(
            unified_timeline=unified_timeline,
            question=question,
        )

        response = self.model.invoke(prompt)

        # Extract answer from [brackets]
        answer = re.findall(r"\[(.*?)\]", response)
        return answer[0] if answer else response.strip()

    def process_single_question(self, data: Dict) -> Dict:
        """Process a single QA pair and return result."""
        prediction = self.invoke(
            data_id=data["data_id"],
            context_graphs=data["context_graphs"],  # Pass list, not string
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
        print(f"Running Unified Hierarchical SGQA Evaluation (v1) with {self.model_name}")
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
                print(f"\r[Unified-{self.model_name}] {processed}/{total_questions} | "
                      f"EM: {current_em:.1f}%", end="", flush=True)

        print()  # New line after progress

        # Calculate final Exact Match
        em_score = (total_correct / total_questions) * 100 if total_questions > 0 else 0

        return {
            "model": f"Unified-{self.model_name}",
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
