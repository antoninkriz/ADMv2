(for f in TRAIN_TEST_results/*; do
    if [ -f "$f/mrr.txt" ]; then
        MRR=$(cat "$f/mrr.txt")
        PARAMS=$(cat "$f/params.json" | jq -c)
        echo "$MRR $PARAMS"
    fi
done) | sort

