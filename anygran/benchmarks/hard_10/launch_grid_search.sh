#!/bin/bash
# Grid Search for EpiMine Hard-10 Hyperparameter Tuning
# Parameters: threshold_std × min_freq × top_k × use_llm
# Grid: 3 × 2 × 3 × 2 = 36 experiments

cd /home/jtu9/sgg/tsg-bench

# Define parameter values
THRESHOLDS=(1.0 1.5 2.0)
MIN_FREQS=(1 2)
TOP_KS=(5 10)  # None handled separately
LLMS=(0 1)    # 0 = --no-llm, 1 = with llm

IDX=1

# Loop through all combinations
for t in "${THRESHOLDS[@]}"; do
    for mf in "${MIN_FREQS[@]}"; do
        for topk in "${TOP_KS[@]}"; do
            for llm in "${LLMS[@]}"; do
                SESSION="grid_${IDX}"

                if [ "$llm" -eq 0 ]; then
                    LLM_FLAG="--no-llm"
                else
                    LLM_FLAG=""
                fi

                CMD="cd /home/jtu9/sgg/tsg-bench && python anygran/benchmarks/hard_10/run_epimine_hard_10.py --model gpt5 --skip-baseline --cooccur-threshold $t --min-freq $mf --top-k $topk $LLM_FLAG"

                echo "[$IDX/36] Creating session $SESSION: t=$t mf=$mf topk=$topk llm=$llm"
                tmux new-session -d -s "$SESSION" "$CMD"

                ((IDX++))
            done
        done

        # Also run with top_k=None (all terms)
        for llm in "${LLMS[@]}"; do
            SESSION="grid_${IDX}"

            if [ "$llm" -eq 0 ]; then
                LLM_FLAG="--no-llm"
            else
                LLM_FLAG=""
            fi

            CMD="cd /home/jtu9/sgg/tsg-bench && python anygran/benchmarks/hard_10/run_epimine_hard_10.py --model gpt5 --skip-baseline --cooccur-threshold $t --min-freq $mf $LLM_FLAG"

            echo "[$IDX/36] Creating session $SESSION: t=$t mf=$mf topk=all llm=$llm"
            tmux new-session -d -s "$SESSION" "$CMD"

            ((IDX++))
        done
    done
done

echo ""
echo "Launched $((IDX-1)) grid search sessions!"
echo "Use 'tmux list-sessions' to see them"
echo "Use 'tmux attach -t grid_N' to attach to session N"
