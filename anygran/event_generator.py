"""
Event Generator for Multi-Granular Scene Graphs
================================================
Uses GPT-5-mini to generate hierarchical event summaries from scene graph sequences.

Hierarchy:
  Level 3 (Goal): Overall activity description
  Level 2 (Sub-Events): Grouped phases of the activity
  Level 1 (Actions): Individual scene graphs (existing)
  Level 0 (Triplets): Raw triplets (existing)
"""

import json
import sys
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


class EventGenerator:
    """Generates hierarchical event summaries from scene graph sequences."""

    def __init__(self, model=None):
        """
        Initialize the event generator.

        Args:
            model: LLM model to use. Defaults to GPT5Mini.
        """
        self.model = model or GPT5Mini()
        self.model_name = self.model.__class__.__name__

        # Load prompt template
        prompt = load_prompt("event_generation.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["scene_graphs"],
            template=prompt,
        )

    def generate_event_hierarchy(self, context_graphs: List) -> Dict:
        """
        Generate hierarchical event summary from scene graph sequence.

        Args:
            context_graphs: List of action graphs, each containing triplets.

        Returns:
            Dict with 'overall_goal' and 'sub_events' keys.
        """
        # Format scene graphs for the prompt
        formatted_graphs = self._format_graphs_for_prompt(context_graphs)

        prompt = self.prompt_template.format(scene_graphs=formatted_graphs)
        response = self.model.invoke(prompt)

        # Parse JSON response
        try:
            # Try to extract JSON from the response
            event_data = self._parse_json_response(response)
            return event_data
        except Exception as e:
            print(f"Warning: Failed to parse event response: {e}")
            # Return a fallback structure
            return {
                "overall_goal": "Activity sequence",
                "sub_events": [{"name": "All actions", "description": "Complete sequence", "action_indices": list(range(len(context_graphs)))}],
                "raw_response": response
            }

    def _format_graphs_for_prompt(self, context_graphs: List) -> str:
        """Format scene graphs in a readable way for the prompt."""
        lines = []
        for i, action_graph in enumerate(context_graphs):
            # Extract the main action verb from this graph
            action_verb = self._extract_action_verb(action_graph)
            lines.append(f"Action {i}: {action_verb}")
            for triplet in action_graph:
                lines.append(f"  {triplet}")
        return "\n".join(lines)

    def _extract_action_verb(self, action_graph: List) -> str:
        """Extract the main action verb from an action graph."""
        for triplet in action_graph:
            if len(triplet) >= 3 and triplet[1] == "verb":
                return triplet[2]  # The action verb
        return "unknown"

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from the model response."""
        # Try to find JSON in the response
        response = response.strip()

        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)

        raise ValueError(f"Could not parse JSON from response: {response[:200]}")

    def batch_generate(
        self,
        data: List[Dict],
        output_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Generate event hierarchies for multiple samples.

        Args:
            data: List of SGQA data items with 'data_id' and 'context_graphs'.
            output_path: Optional path to save results as JSON.
            limit: Optional limit on number of samples to process.

        Returns:
            Dict mapping data_id to event hierarchy.
        """
        # Get unique data_ids (each sample may have multiple QA pairs)
        seen_ids = set()
        unique_samples = []
        for item in data:
            if item["data_id"] not in seen_ids:
                seen_ids.add(item["data_id"])
                unique_samples.append(item)

        if limit:
            unique_samples = unique_samples[:limit]

        results = {}
        total = len(unique_samples)

        print(f"\nGenerating event hierarchies for {total} samples...")

        for i, item in enumerate(unique_samples):
            data_id = item["data_id"]
            context_graphs = item["context_graphs"]

            print(f"\r[{i+1}/{total}] Processing {data_id[:20]}...", end="", flush=True)

            event_hierarchy = self.generate_event_hierarchy(context_graphs)
            results[data_id] = event_hierarchy

        print()  # New line after progress

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"Saved event hierarchies to: {output_path}")

        return results


def load_events_cache(cache_path: str) -> Dict[str, Dict]:
    """Load cached event hierarchies from file."""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Test with a sample scene graph sequence
    sample_graphs = [
        [["pick-up", "with", "hand1"], ["person", "verb", "pick-up"], ["pick-up", "dobj", "mop-stick"]],
        [["sweep", "with", "mop-stick"], ["person", "verb", "sweep"], ["sweep", "dobj", "floor"]],
        [["place", "on", "floor"], ["person", "verb", "place"], ["place", "dobj", "mop-stick"]],
    ]

    generator = EventGenerator()
    result = generator.generate_event_hierarchy(sample_graphs)
    print("\nGenerated Event Hierarchy:")
    print(json.dumps(result, indent=2))
