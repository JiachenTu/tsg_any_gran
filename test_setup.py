"""
Light test script to verify the setup works with minimal API usage.
Tests with GPT-4o-mini on just 2 samples from the SA-SGG dataset.
"""

import json
import sys
import importlib.util

from models.models import GPT4oMini

# Import from file with hyphen in name
spec = importlib.util.spec_from_file_location("sa_sgg", "evaluation/generation/sa-sgg.py")
sa_sgg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa_sgg)

GraphGeneration = sa_sgg.GraphGeneration
GraphScorer = sa_sgg.GraphScorer

def test_setup():
    """Test the setup with minimal API calls"""

    print("=" * 60)
    print("TSG-Bench Setup Test")
    print("=" * 60)
    print("\n1. Testing model initialization...")

    try:
        model = GPT4oMini()
        print("✓ GPT-4o-mini model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return

    print("\n2. Testing scene graph generation on 2 samples...")

    generation = GraphGeneration(model)
    data_path = "resource/dataset/generation/sa-sgg.jsonl"

    # Read only first 2 lines
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [f.readline() for _ in range(2)]

    results = []
    for i, line in enumerate(lines, 1):
        data = json.loads(line)

        print(f"\n   Sample {i}:")
        print(f"   - Sentence: {data['target_sentence'][:80]}...")

        context = data["context"]
        target_sentence = data["target_sentence"]
        available_nodes = ", ".join(data["mandatory_space"]["object"])
        available_edges = ", ".join(data["mandatory_space"]["relationship"])
        verbs = data["mandatory_space"]["verb"]

        try:
            response = generation.invoke(
                context=context,
                target_sentence=target_sentence,
                available_nodes=available_nodes,
                available_edges=available_edges,
                verbs=verbs,
            )

            pred_scene_graphs = GraphScorer.parse_response(response)

            if len(pred_scene_graphs) > 0 and len(data["graphs"]) > 0:
                scores = GraphScorer.calculate_scores(
                    data["graphs"][0]["triplets"],
                    pred_scene_graphs[0]
                )

                print(f"   - Generated {len(pred_scene_graphs[0])} triplets")
                print(f"   - Precision: {scores['precision']:.2f}")
                print(f"   - Recall: {scores['recall']:.2f}")
                print(f"   - F1: {scores['f1']:.2f}")

                results.append(scores)
            else:
                print(f"   ✗ Failed to generate valid scene graph")

        except Exception as e:
            print(f"   ✗ Error during generation: {e}")
            continue

    if results:
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)

        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"  Samples processed: {len(results)}/2")
        print(f"  Average Precision: {avg_precision:.2f}")
        print(f"  Average Recall: {avg_recall:.2f}")
        print(f"  Average F1: {avg_f1:.2f}")
        print("=" * 60)
        print("\n✓ Setup test completed successfully!")
        print("\nYou can now run full evaluations with:")
        print("  python evaluation/generation/sa-sgg.py")
        print("  python evaluation/generation/ma-sgg.py")
        print("  python evaluation/understanding/sgqa.py")
        print("  python evaluation/understanding/sgds.py")
    else:
        print("\n✗ Test failed - no successful samples processed")

if __name__ == "__main__":
    test_setup()
