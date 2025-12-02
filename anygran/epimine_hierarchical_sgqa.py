"""
EpiMine-Enhanced Hierarchical SGQA
==================================
Applies EpiMine's unsupervised episode detection philosophy to action scene graphs.

Key adaptations from original EpiMine:
- Key terms = action verbs + objects + relations (from triplets)
- Segments = action graphs (not sentences)
- Background = all actions across sgqa.jsonl
- Episodes have structured format: {agent, actions, objects, instruments, locations, time}

References:
- EpiMine Paper: https://arxiv.org/abs/2408.04873
- Original implementation: /home/jtu9/sgg/structure_mining/epimine/
"""

import json
import re
import sys
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import concurrent.futures

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils.config import load_config, get_config_file_path

from anygran.hierarchical_sgqa import UnifiedHierarchicalEvaluator, BaselineSGQAEvaluator


# =============================================================================
# Model Wrappers
# =============================================================================

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


def load_prompt(prompt_name: str) -> str:
    """Load prompt template from anygran/prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / prompt_name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# =============================================================================
# Triplet Parsing Utilities
# =============================================================================

# Relation types for extracting structured information
VERB_RELATIONS = {"verb", "verbs"}
OBJECT_RELATIONS = {"dobj", "obj", "pobj"}
INSTRUMENT_RELATIONS = {"with"}
SOURCE_RELATIONS = {"from"}
TARGET_RELATIONS = {"to", "on", "in", "into", "onto", "inside"}


def extract_terms_from_triplet(triplet: List[str]) -> Set[str]:
    """
    Extract all terms from a single triplet.

    Args:
        triplet: [node1, relation, node2]

    Returns:
        Set of all terms in the triplet
    """
    if len(triplet) >= 3:
        return {triplet[0], triplet[1], triplet[2]}
    return set(triplet)


def extract_terms_from_action(action_graph: List[List[str]]) -> Set[str]:
    """
    Extract all terms from an action graph.

    Args:
        action_graph: List of triplets for one action

    Returns:
        Set of all unique terms
    """
    terms = set()
    for triplet in action_graph:
        terms.update(extract_terms_from_triplet(triplet))
    return terms


def extract_structured_info(action_graph: List[List[str]]) -> Dict:
    """
    Extract structured information from an action graph.

    Args:
        action_graph: List of triplets for one action

    Returns:
        Dict with agent, action, objects, instruments, source_locations, target_locations
    """
    info = {
        "agent": None,
        "action": None,
        "objects": [],
        "instruments": [],
        "source_locations": [],
        "target_locations": [],
    }

    for triplet in action_graph:
        if len(triplet) < 3:
            continue

        node1, relation, node2 = triplet[0], triplet[1], triplet[2]

        # Extract action verb: [person, verb, X] -> X is the action
        if relation in VERB_RELATIONS:
            if node1 == "person":
                info["agent"] = node1
                info["action"] = node2

        # Extract direct object: [action, dobj, X] -> X is the object
        elif relation in OBJECT_RELATIONS:
            info["objects"].append(node2)

        # Extract instrument: [action, with, X] -> X is the instrument
        elif relation in INSTRUMENT_RELATIONS:
            info["instruments"].append(node2)

        # Extract source location: [X, from, Y] -> Y is source
        elif relation in SOURCE_RELATIONS:
            info["source_locations"].append(node2)

        # Extract target location: [action, to/on/in, X] -> X is target
        elif relation in TARGET_RELATIONS:
            info["target_locations"].append(node2)

    return info


def get_action_verb(action_graph: List[List[str]]) -> str:
    """Extract the main action verb from an action graph."""
    for triplet in action_graph:
        if len(triplet) >= 3 and triplet[1] in VERB_RELATIONS:
            return triplet[2]
    return "unknown"


# =============================================================================
# EpiMine Action Analyzer
# =============================================================================

class EpiMineActionAnalyzer:
    """
    Apply EpiMine philosophy to detect episodes from action sequences.

    Adaptations from original EpiMine:
    - Key terms = action verbs + objects + relations (from triplets)
    - Segments = action graphs (not sentences)
    - Background = all actions across sgqa.jsonl
    - No document selection (use all data)
    - Start with co-occurrence only (no embeddings initially)
    """

    def __init__(self, background_dataset: List[List[List[str]]], min_freq: int = 2):
        """
        Initialize the analyzer.

        Args:
            background_dataset: All action graphs from sgqa.jsonl
            min_freq: Minimum frequency for a term to be considered
        """
        self.background = background_dataset
        self.min_freq = min_freq
        self.term_counts_bg = self._count_terms(background_dataset)
        self.num_bg = len(background_dataset)

    def _count_terms(self, action_graphs: List[List[List[str]]]) -> Counter:
        """Count term occurrences across all action graphs."""
        counts = Counter()
        for action_graph in action_graphs:
            terms = extract_terms_from_action(action_graph)
            counts.update(terms)
        return counts

    def compute_salience(self, term: str, foreground: List[List[List[str]]]) -> float:
        """
        Compute discriminative salience for a term.

        Formula: salience = (1 + log(fg_count)²) × log(bg_total / bg_count)

        Args:
            term: The term to compute salience for
            foreground: The foreground action graphs (current sample)

        Returns:
            Salience score (higher = more discriminative)
        """
        # Count foreground occurrences
        fg_count = sum(1 for ag in foreground if term in extract_terms_from_action(ag))
        bg_count = self.term_counts_bg.get(term, 0)

        if fg_count < self.min_freq:
            return -1.0

        if bg_count > 0:
            return (1 + np.log(fg_count) ** 2) * np.log(self.num_bg / bg_count)
        else:
            return (1 + np.log(fg_count) ** 2) * np.log(self.num_bg)

    def get_key_terms(self, foreground: List[List[List[str]]], top_k: int = None) -> List[Tuple[str, float]]:
        """
        Get key discriminative terms for the foreground.

        Args:
            foreground: The foreground action graphs
            top_k: Optional limit on number of terms

        Returns:
            List of (term, salience_score) tuples, sorted by salience
        """
        # Collect all terms from foreground
        all_terms = set()
        for action_graph in foreground:
            all_terms.update(extract_terms_from_action(action_graph))

        # Compute salience for each term
        term_salience = []
        for term in all_terms:
            salience = self.compute_salience(term, foreground)
            if salience > 0:
                term_salience.append((term, salience))

        # Sort by salience (descending)
        term_salience.sort(key=lambda x: -x[1])

        if top_k:
            return term_salience[:top_k]
        return term_salience

    def compute_cooccurrence_matrix(
        self,
        action_sequence: List[List[List[str]]],
        key_terms: List[str]
    ) -> np.ndarray:
        """
        Build co-occurrence matrix for key terms across action graphs.

        Args:
            action_sequence: List of action graphs
            key_terms: List of key terms to track

        Returns:
            Co-occurrence matrix (num_actions x num_actions)
        """
        num_actions = len(action_sequence)

        # Build term presence matrix: which terms appear in which actions
        term_to_idx = {t: i for i, t in enumerate(key_terms)}
        term_presence = np.zeros((num_actions, len(key_terms)))

        for action_idx, action_graph in enumerate(action_sequence):
            terms = extract_terms_from_action(action_graph)
            for term in terms:
                if term in term_to_idx:
                    term_presence[action_idx, term_to_idx[term]] = 1

        # Compute co-occurrence between consecutive actions
        cooccur_matrix = np.zeros((num_actions, num_actions))

        for i in range(num_actions):
            for j in range(num_actions):
                if i != j:
                    # Count shared terms
                    shared = np.sum(term_presence[i] * term_presence[j])
                    total = np.sum(np.logical_or(term_presence[i], term_presence[j]))
                    if total > 0:
                        cooccur_matrix[i, j] = shared / total  # Jaccard similarity

        return cooccur_matrix

    def detect_episode_boundaries(
        self,
        action_sequence: List[List[List[str]]],
        threshold_std: float = 1.0
    ) -> List[List[int]]:
        """
        Detect episode boundaries based on co-occurrence shifts.

        Uses mean - threshold_std * σ as the boundary threshold.

        Args:
            action_sequence: List of action graphs
            threshold_std: Number of standard deviations below mean for boundary

        Returns:
            List of episode groups, each containing action indices
        """
        if len(action_sequence) <= 1:
            return [[i for i in range(len(action_sequence))]]

        # Get key terms
        key_terms_with_scores = self.get_key_terms(action_sequence)
        key_terms = [t for t, _ in key_terms_with_scores]

        if not key_terms:
            # No discriminative terms, treat as single episode
            return [[i for i in range(len(action_sequence))]]

        # Compute co-occurrence matrix
        cooccur_matrix = self.compute_cooccurrence_matrix(action_sequence, key_terms)

        # Compute consecutive co-occurrence scores
        consecutive_scores = []
        for i in range(len(action_sequence) - 1):
            consecutive_scores.append(cooccur_matrix[i, i + 1])

        if not consecutive_scores:
            return [[i for i in range(len(action_sequence))]]

        # Compute threshold
        mean_score = np.mean(consecutive_scores)
        std_score = np.std(consecutive_scores)
        threshold = mean_score - threshold_std * std_score

        # Detect boundaries
        episodes = [[0]]
        for i, score in enumerate(consecutive_scores):
            if score < threshold:
                # Boundary detected, start new episode
                episodes.append([i + 1])
            else:
                # Continue current episode
                episodes[-1].append(i + 1)

        return episodes

    def analyze_sample(
        self,
        action_sequence: List[List[List[str]]],
        threshold_std: float = 1.0
    ) -> Dict:
        """
        Full analysis of an action sequence.

        Args:
            action_sequence: List of action graphs
            threshold_std: Threshold for boundary detection

        Returns:
            Dict with key_terms, cooccurrence_matrix, and episode_boundaries
        """
        key_terms_with_scores = self.get_key_terms(action_sequence)
        key_terms = [t for t, _ in key_terms_with_scores]

        cooccur_matrix = None
        if key_terms:
            cooccur_matrix = self.compute_cooccurrence_matrix(action_sequence, key_terms)

        episode_boundaries = self.detect_episode_boundaries(action_sequence, threshold_std)

        return {
            "key_terms": key_terms_with_scores,
            "cooccurrence_matrix": cooccur_matrix,
            "episode_boundaries": episode_boundaries,
        }


# =============================================================================
# EpiMine Episode Generator
# =============================================================================

class EpiMineEpisodeGenerator:
    """
    Generate structured episode descriptions using GPT.

    Takes candidate episode boundaries from EpiMineActionAnalyzer
    and generates structured episode descriptions.
    """

    def __init__(self, model=None):
        """
        Initialize the generator.

        Args:
            model: LLM model to use. Defaults to GPT5Mini.
        """
        self.model = model or GPT5Mini()
        self.model_name = self.model.__class__.__name__

        # Load prompt template
        prompt = load_prompt("epimine_episode_generation.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["candidate_episodes", "formatted_actions"],
            template=prompt,
        )

    def _format_actions_for_prompt(self, action_sequence: List[List[List[str]]]) -> str:
        """Format action sequence for the prompt."""
        lines = []
        for i, action_graph in enumerate(action_sequence):
            verb = get_action_verb(action_graph)
            triplets_str = " ".join(str(t) for t in action_graph)
            lines.append(f"Action {i} ({verb}): {triplets_str}")
        return "\n".join(lines)

    def _format_candidate_episodes(
        self,
        episode_boundaries: List[List[int]],
        action_sequence: List[List[List[str]]]
    ) -> str:
        """Format candidate episodes for the prompt."""
        lines = []
        for i, action_indices in enumerate(episode_boundaries):
            verbs = [get_action_verb(action_sequence[idx]) for idx in action_indices]
            lines.append(f"Group {i}: Actions {action_indices} - verbs: {verbs}")
        return "\n".join(lines)

    def _build_structured_episode(
        self,
        episode_id: int,
        action_indices: List[int],
        action_sequence: List[List[List[str]]],
        name: str = None,
        description: str = None,
        key_terms: List[Tuple[str, float]] = None,
        total_episodes: int = 1
    ) -> Dict:
        """
        Build a structured episode from action indices.

        Args:
            episode_id: ID of this episode
            action_indices: List of action indices in this episode
            action_sequence: Full action sequence
            name: Optional name from LLM
            description: Optional description from LLM
            key_terms: Optional key terms with salience scores
            total_episodes: Total number of episodes

        Returns:
            Structured episode dict
        """
        # Extract structured info from all actions in this episode
        agent = "person"
        primary_actions = []
        primary_objects = []
        instruments = []
        source_locations = []
        target_locations = []

        for idx in action_indices:
            info = extract_structured_info(action_sequence[idx])
            if info["agent"]:
                agent = info["agent"]
            if info["action"]:
                primary_actions.append(info["action"])
            primary_objects.extend(info["objects"])
            instruments.extend(info["instruments"])
            source_locations.extend(info["source_locations"])
            target_locations.extend(info["target_locations"])

        # Deduplicate
        primary_actions = list(dict.fromkeys(primary_actions))
        primary_objects = list(dict.fromkeys(primary_objects))
        instruments = list(dict.fromkeys(instruments))
        source_locations = list(dict.fromkeys(source_locations))
        target_locations = list(dict.fromkeys(target_locations))

        # Determine temporal position
        if episode_id == 0:
            position = "beginning"
        elif episode_id == total_episodes - 1:
            position = "end"
        else:
            position = "middle"

        # Compute discriminative terms for this episode
        discriminative_terms = []
        if key_terms:
            episode_terms = set()
            for idx in action_indices:
                episode_terms.update(extract_terms_from_action(action_sequence[idx]))
            discriminative_terms = [t for t, _ in key_terms if t in episode_terms][:5]

        # Compute salience score (average of discriminative term scores)
        salience_score = 0.0
        if key_terms and discriminative_terms:
            term_scores = {t: s for t, s in key_terms}
            scores = [term_scores.get(t, 0) for t in discriminative_terms]
            if scores:
                # Normalize to 0-1 range
                max_score = max(s for _, s in key_terms) if key_terms else 1
                salience_score = np.mean(scores) / max_score if max_score > 0 else 0

        # Build episode structure
        episode = {
            "episode_id": episode_id,
            "name": name or f"Episode {episode_id}",
            "description": description or f"Actions: {', '.join(primary_actions)}",

            "core_structure": {
                "agent": agent,
                "primary_actions": primary_actions,
                "primary_objects": primary_objects,
                "instruments": instruments,
                "source_locations": source_locations if source_locations else None,
                "target_locations": target_locations if target_locations else None,
            },

            "time": {
                "action_indices": action_indices,
                "start_index": min(action_indices),
                "end_index": max(action_indices),
                "duration": len(action_indices),
            },

            "temporal_context": {
                "position": position,
                "precedes_episodes": list(range(episode_id + 1, total_episodes)) if episode_id < total_episodes - 1 else None,
                "follows_episodes": list(range(episode_id)) if episode_id > 0 else None,
            },

            "discriminative_terms": discriminative_terms,
            "salience_score": round(salience_score, 3),
        }

        return episode

    def generate_episode_hierarchy(
        self,
        action_sequence: List[List[List[str]]],
        episode_boundaries: List[List[int]],
        key_terms: List[Tuple[str, float]] = None,
        use_llm: bool = True
    ) -> Dict:
        """
        Generate structured episode hierarchy.

        Args:
            action_sequence: List of action graphs
            episode_boundaries: Candidate episode groupings from analyzer
            key_terms: Key terms with salience scores
            use_llm: Whether to use LLM for name/description generation

        Returns:
            Dict with overall_goal and episodes list
        """
        # Format for LLM prompt
        formatted_actions = self._format_actions_for_prompt(action_sequence)
        candidate_episodes = self._format_candidate_episodes(episode_boundaries, action_sequence)

        # Default values
        overall_goal = "Activity sequence"
        episode_names = {}
        episode_descriptions = {}

        # Optionally use LLM to generate names and descriptions
        if use_llm:
            try:
                prompt = self.prompt_template.format(
                    candidate_episodes=candidate_episodes,
                    formatted_actions=formatted_actions,
                )
                response = self.model.invoke(prompt)

                # Parse JSON response
                llm_data = self._parse_json_response(response)

                if "overall_goal" in llm_data:
                    overall_goal = llm_data["overall_goal"]

                if "episodes" in llm_data:
                    for ep in llm_data["episodes"]:
                        if "action_indices" in ep and isinstance(ep["action_indices"], list):
                            # Match to our episode boundaries
                            for i, boundary in enumerate(episode_boundaries):
                                if set(ep["action_indices"]) == set(boundary):
                                    episode_names[i] = ep.get("name", f"Episode {i}")
                                    episode_descriptions[i] = ep.get("description", "")
                                    break
                        elif "episode_id" in ep:
                            idx = ep["episode_id"]
                            episode_names[idx] = ep.get("name", f"Episode {idx}")
                            episode_descriptions[idx] = ep.get("description", "")

            except Exception as e:
                print(f"Warning: LLM generation failed: {e}")

        # Build structured episodes
        episodes = []
        for i, action_indices in enumerate(episode_boundaries):
            episode = self._build_structured_episode(
                episode_id=i,
                action_indices=action_indices,
                action_sequence=action_sequence,
                name=episode_names.get(i),
                description=episode_descriptions.get(i),
                key_terms=key_terms,
                total_episodes=len(episode_boundaries),
            )
            episodes.append(episode)

        return {
            "overall_goal": overall_goal,
            "episodes": episodes,
        }

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from the model response."""
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
        analyzer: 'EpiMineActionAnalyzer',
        output_path: Optional[str] = None,
        use_llm: bool = True,
        threshold_std: float = 1.0
    ) -> Dict[str, Dict]:
        """
        Generate episode hierarchies for multiple samples.

        Args:
            data: List of SGQA data items with 'data_id' and 'context_graphs'
            analyzer: EpiMineActionAnalyzer instance
            output_path: Optional path to save results as JSON
            use_llm: Whether to use LLM for name/description generation
            threshold_std: Threshold for boundary detection

        Returns:
            Dict mapping data_id to episode hierarchy
        """
        # Get unique data_ids
        seen_ids = set()
        unique_samples = []
        for item in data:
            if item["data_id"] not in seen_ids:
                seen_ids.add(item["data_id"])
                unique_samples.append(item)

        results = {}
        total = len(unique_samples)

        print(f"\nGenerating EpiMine episode hierarchies for {total} samples...")

        for i, item in enumerate(unique_samples):
            data_id = item["data_id"]
            action_sequence = item["context_graphs"]

            print(f"\r[{i+1}/{total}] Processing {data_id[:20]}...", end="", flush=True)

            # Analyze with EpiMine
            analysis = analyzer.analyze_sample(action_sequence, threshold_std)

            # Generate structured episodes
            hierarchy = self.generate_episode_hierarchy(
                action_sequence=action_sequence,
                episode_boundaries=analysis["episode_boundaries"],
                key_terms=analysis["key_terms"],
                use_llm=use_llm,
            )

            results[data_id] = hierarchy

        print()  # New line after progress

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"Saved EpiMine episode hierarchies to: {output_path}")

        return results


# =============================================================================
# EpiMine Hierarchical Evaluator
# =============================================================================

class EpiMineHierarchicalEvaluator:
    """
    SGQA Evaluator using EpiMine-detected structured episodes.

    Combines:
    - Unsupervised episode detection (EpiMine philosophy)
    - Structured episode format {agent, actions, objects, instruments, locations, time}
    - LLM-generated episode descriptions (GPT)
    - Unified/interleaved prompt format (from v1)
    """

    def __init__(
        self,
        model=None,
        episodes_cache: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize the evaluator.

        Args:
            model: LLM model for QA evaluation
            episodes_cache: Dict mapping data_id → EpiMine episode hierarchy
        """
        self.model = model or GPT5Mini()
        self.model_name = self.model.__class__.__name__
        self.episodes_cache = episodes_cache or {}

        # Load unified prompt template
        prompt = load_prompt("unified_hierarchical.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["unified_timeline", "question"],
            template=prompt,
        )

    def _format_epimine_timeline(
        self,
        episodes: List[Dict],
        action_sequence: List[List[List[str]]],
        overall_goal: str = "Activity sequence"
    ) -> str:
        """
        Format EpiMine-detected structured episodes in unified style.

        Augments v1 format with:
        - Overall goal at the top
        - Structured episode information
        - Discriminative terms highlighted
        - Explicit temporal context
        """
        if not episodes:
            # Fallback: list all actions without episodes
            lines = [f"## Overall Goal\n{overall_goal}\n", "### All Actions"]
            for i, action_graph in enumerate(action_sequence):
                verb = get_action_verb(action_graph)
                triplets_str = " ".join(str(t) for t in action_graph)
                lines.append(f"- Action {i} ({verb}): {triplets_str}")
            return "\n".join(lines)

        # Start with overall goal
        lines = [f"## Overall Goal\n{overall_goal}\n\n## Activity Timeline\n"]
        for episode in episodes:
            ep_id = episode["episode_id"]
            name = episode["name"]
            description = episode["description"]
            core = episode["core_structure"]
            time_info = episode["time"]
            disc_terms = episode.get("discriminative_terms", [])

            # Episode header
            lines.append(f"### Episode {ep_id}: {name}")
            lines.append(f"{description}")
            lines.append("")

            # Structured summary
            lines.append(f"**Structure:**")
            lines.append(f"- Agent: {core['agent']}")
            lines.append(f"- Actions: {', '.join(core['primary_actions'])}")
            if core['primary_objects']:
                lines.append(f"- Objects: {', '.join(core['primary_objects'])}")
            if core['instruments']:
                lines.append(f"- Instruments: {', '.join(core['instruments'])}")
            if core['source_locations']:
                lines.append(f"- From: {', '.join(core['source_locations'])}")
            if core['target_locations']:
                lines.append(f"- To: {', '.join(core['target_locations'])}")
            lines.append(f"- Time: Actions {time_info['action_indices']} (duration: {time_info['duration']})")

            if disc_terms:
                lines.append(f"- Key terms: {', '.join(disc_terms)}")
            lines.append("")

            # Actions in this episode
            lines.append("**Actions in this episode:**")
            for action_idx in time_info["action_indices"]:
                if action_idx < len(action_sequence):
                    action_graph = action_sequence[action_idx]
                    verb = get_action_verb(action_graph)
                    triplets_str = " ".join(str(t) for t in action_graph)
                    lines.append(f"- Action {action_idx} ({verb}): {triplets_str}")

            lines.append("")  # Blank line between episodes

        return "\n".join(lines)

    def invoke(self, data_id: str, context_graphs: List, question: str) -> str:
        """
        Invoke model with EpiMine hierarchical context.

        Args:
            data_id: Unique identifier for the sample
            context_graphs: List of action graphs
            question: The question to answer

        Returns:
            Extracted answer string
        """
        # Get episode hierarchy from cache
        hierarchy = self.episodes_cache.get(data_id, {})
        overall_goal = hierarchy.get("overall_goal", "Activity sequence")
        episodes = hierarchy.get("episodes", [])

        # Format unified timeline with EpiMine structure (includes overall goal)
        unified_timeline = self._format_epimine_timeline(episodes, context_graphs, overall_goal)

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
            context_graphs=data["context_graphs"],
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
            data: List of QA items with data_id, context_graphs, question, answer
            max_workers: Number of parallel workers

        Returns:
            Dict with evaluation results and metrics
        """
        results = []
        total_correct = 0
        total_questions = len(data)

        print(f"\n{'='*60}")
        print(f"Running EpiMine Hierarchical SGQA Evaluation with {self.model_name}")
        print(f"Total questions: {total_questions}")
        print(f"Episode cache size: {len(self.episodes_cache)} samples")
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
                print(f"\r[EpiMine-{self.model_name}] {processed}/{total_questions} | "
                      f"EM: {current_em:.1f}%", end="", flush=True)

        print()  # New line after progress

        # Calculate final Exact Match
        em_score = (total_correct / total_questions) * 100 if total_questions > 0 else 0

        return {
            "model": f"EpiMine-{self.model_name}",
            "total_questions": total_questions,
            "correct": total_correct,
            "exact_match_percent": round(em_score, 2),
            "results": results,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def build_background_dataset(sgqa_path: str) -> List[List[List[str]]]:
    """
    Build background corpus from all sgqa.jsonl actions.

    This provides the "general distribution" against which
    foreground actions can be compared for salience computation.

    Args:
        sgqa_path: Path to sgqa.jsonl file

    Returns:
        List of all action graphs
    """
    all_actions = []
    with open(sgqa_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                for action_graph in item["context_graphs"]:
                    all_actions.append(action_graph)
    return all_actions


def load_episodes_cache(cache_path: str) -> Dict[str, Dict]:
    """Load cached episode hierarchies from file."""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
