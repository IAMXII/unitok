#!/bin/bash
INPUT_PAIRS=/data/bench2drive
CACHE_ROOT=/data/vqcache
OUTPUT_PATH=/data/tempdata
NUM_CHUNKS=8

for IDX in $(seq 0 7); do
    python build_waypoint_samples.py \
        --input_pairs $INPUT_PAIRS \
        --cache_root $CACHE_ROOT \
        --temp_path $OUTPUT_PATH \
        --chunk_idx $IDX \
        --num_chunks $NUM_CHUNKS > ./logs/build_log_${IDX}.txt 2>&1 &
done

wait
echo "All VQA sample building done."
