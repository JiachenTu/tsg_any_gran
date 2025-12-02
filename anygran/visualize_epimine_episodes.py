#!/usr/bin/env python3
"""
EpiMine Episode Visualization
=============================
Visualize the mined episodes/subevents from action scene graphs.
"""

import json
import argparse
from pathlib import Path


def visualize_episodes(cache_path: str, limit: int = None):
    """Visualize episodes from a cache file."""

    with open(cache_path, 'r', encoding='utf-8') as f:
        episodes_cache = json.load(f)

    samples = list(episodes_cache.items())
    if limit:
        samples = samples[:limit]

    print("=" * 80)
    print("EpiMine Hierarchical Episode Visualization")
    print("=" * 80)
    print(f"\nTotal samples: {len(samples)}")

    total_episodes = sum(len(h.get('episodes', [])) for _, h in samples)
    print(f"Total episodes detected: {total_episodes}")
    print(f"Average episodes per sample: {total_episodes / len(samples):.2f}")

    for i, (data_id, hierarchy) in enumerate(samples, 1):
        print("\n" + "=" * 80)
        print(f"Sample {i}: {data_id[:40]}...")
        print("=" * 80)

        # Overall goal
        overall_goal = hierarchy.get('overall_goal', 'N/A')
        print(f"\nOverall Goal: {overall_goal}")

        episodes = hierarchy.get('episodes', [])
        print(f"\nNumber of Episodes: {len(episodes)}")

        for ep in episodes:
            ep_id = ep.get('episode_id', 'N/A')
            name = ep.get('name', f'Episode {ep_id}')
            description = ep.get('description', '')

            print(f"\n{'─' * 60}")
            print(f"Episode {ep_id}: {name}")
            print(f"{'─' * 60}")

            if description:
                print(f"  Description: {description}")

            # Time info
            time_info = ep.get('time', {})
            action_indices = time_info.get('action_indices', [])
            start_idx = time_info.get('start_index', 'N/A')
            end_idx = time_info.get('end_index', 'N/A')
            duration = time_info.get('duration', len(action_indices))

            print(f"\n  Time:")
            print(f"    Action Indices: {action_indices}")
            print(f"    Range: {start_idx} - {end_idx} ({duration} actions)")

            # Temporal context
            temporal = ep.get('temporal_context', {})
            position = temporal.get('position', 'N/A')
            print(f"    Position: {position}")

            # Core structure
            core = ep.get('core_structure', {})
            print(f"\n  Core Structure:")
            print(f"    Agent: {core.get('agent', 'person')}")

            actions = core.get('primary_actions', [])
            if actions:
                print(f"    Primary Actions: {actions}")

            objects = core.get('primary_objects', [])
            if objects:
                print(f"    Primary Objects: {objects}")

            instruments = core.get('instruments', [])
            if instruments:
                print(f"    Instruments: {instruments}")

            source_locs = core.get('source_locations')
            if source_locs:
                print(f"    Source Locations: {source_locs}")

            target_locs = core.get('target_locations')
            if target_locs:
                print(f"    Target Locations: {target_locs}")

            # Discriminative terms and salience
            disc_terms = ep.get('discriminative_terms', [])
            if disc_terms:
                print(f"\n  Discriminative Terms: {disc_terms}")

            salience = ep.get('salience_score', 0)
            if salience:
                print(f"  Salience Score: {salience:.3f}")

    print("\n" + "=" * 80)
    print("End of Visualization")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize EpiMine episodes")
    parser.add_argument(
        "--cache-path",
        type=str,
        default=str(Path(__file__).parent / "cache" / "epimine_episodes_limit10.json"),
        help="Path to episode cache JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to visualize"
    )

    args = parser.parse_args()

    if not Path(args.cache_path).exists():
        print(f"Error: Cache file not found at {args.cache_path}")
        print("Run: python anygran/run_epimine_hierarchical_sgqa.py --limit 10 --generate-only")
        return

    visualize_episodes(args.cache_path, args.limit)


if __name__ == "__main__":
    main()
