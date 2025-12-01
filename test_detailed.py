"""
Detailed test to show inputs and outputs for 2 samples
"""

import json
import importlib.util

from models.models import GPT4oMini

# Import from file with hyphen in name
spec = importlib.util.spec_from_file_location("sa_sgg", "evaluation/generation/sa-sgg.py")
sa_sgg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa_sgg)

GraphGeneration = sa_sgg.GraphGeneration
GraphScorer = sa_sgg.GraphScorer

def print_separator(char="=", length=80):
    print(char * length)

def test_detailed():
    """Test with detailed input/output display"""

    print_separator()
    print("TSG-Bench Detailed Test - Showing Inputs and Outputs")
    print_separator()

    # Initialize model
    print("\n[1] Initializing GPT-4o-mini model...")
    model = GPT4oMini()
    generation = GraphGeneration(model)
    print("✓ Model initialized")

    # Load 2 samples
    data_path = "resource/dataset/generation/sa-sgg.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(f.readline()) for _ in range(2)]

    # Process each sample
    for idx, data in enumerate(samples, 1):
        print_separator()
        print(f"SAMPLE {idx}")
        print_separator()

        # Show INPUT
        print("\n📥 INPUT:")
        print("-" * 80)
        print(f"Context (previous actions):\n{data['context']}\n")
        print(f"Target Sentence:\n{data['target_sentence']}\n")
        print(f"Available Objects: {', '.join(data['mandatory_space']['object'])}")
        print(f"Available Verbs: {', '.join(data['mandatory_space']['verb'])}")
        print(f"Available Relationships: {', '.join(data['mandatory_space']['relationship'])}")

        # Show GROUND TRUTH
        print("\n🎯 GROUND TRUTH (Expected Output):")
        print("-" * 80)
        if data["graphs"]:
            print("Scene Graph Triplets:")
            for triplet in data["graphs"][0]["triplets"]:
                print(f"  {triplet[0]} -> {triplet[1]} -> {triplet[2]}")

        # Prepare and send to model
        context = data["context"]
        target_sentence = data["target_sentence"]
        available_nodes = ", ".join(data["mandatory_space"]["object"])
        available_edges = ", ".join(data["mandatory_space"]["relationship"])
        verbs = data["mandatory_space"]["verb"]

        print("\n⚙️  Sending to GPT-4o-mini...")

        response = generation.invoke(
            context=context,
            target_sentence=target_sentence,
            available_nodes=available_nodes,
            available_edges=available_edges,
            verbs=verbs,
        )

        # Show MODEL OUTPUT
        print("\n📤 MODEL OUTPUT (Raw Response):")
        print("-" * 80)
        print(response)

        # Parse and show results
        pred_scene_graphs = GraphScorer.parse_response(response)

        print("\n📊 PARSED SCENE GRAPH:")
        print("-" * 80)
        if pred_scene_graphs and len(pred_scene_graphs[0]) > 0:
            print("Predicted Triplets:")
            for triplet in pred_scene_graphs[0]:
                print(f"  {triplet[0]} -> {triplet[1]} -> {triplet[2]}")

            # Calculate scores
            if data["graphs"]:
                scores = GraphScorer.calculate_scores(
                    data["graphs"][0]["triplets"],
                    pred_scene_graphs[0]
                )

                print("\n📈 EVALUATION METRICS:")
                print("-" * 80)
                print(f"Precision: {scores['precision']:.2%}")
                print(f"Recall: {scores['recall']:.2%}")
                print(f"F1 Score: {scores['f1']:.2%}")

                # Show what was missed/wrong
                true_triplet_set = set(tuple(t) for t in data["graphs"][0]["triplets"])
                pred_triplet_set = set(tuple(t) for t in pred_scene_graphs[0])

                missing = true_triplet_set - pred_triplet_set
                incorrect = pred_triplet_set - true_triplet_set

                if missing:
                    print("\n❌ Missing Triplets (should have been included):")
                    for triplet in missing:
                        print(f"  {triplet[0]} -> {triplet[1]} -> {triplet[2]}")

                if incorrect:
                    print("\n⚠️  Incorrect Triplets (shouldn't be there):")
                    for triplet in incorrect:
                        print(f"  {triplet[0]} -> {triplet[1]} -> {triplet[2]}")
        else:
            print("⚠️  No valid scene graph generated")

        print()

    print_separator()
    print("✓ Test completed!")
    print_separator()

if __name__ == "__main__":
    test_detailed()
