#!/bin/bash
# run_multi_gpu.sh

INPUT_PAIRS=/path/to/dataset
UNITOK_PATH=/path/to/unitok.ckpt
CACHE_ROOT=/path/to/cache
NUM_CHUNKS=8  # 和 GPU 数一致
BATCH_SIZE=32
NUM_PROCESSES=4

for IDX in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$IDX \
    python vq_encode_batch.py \
        --input_pairs $INPUT_PAIRS \
        --unitok_path $UNITOK_PATH \
        --cache_root $CACHE_ROOT \
        --chunk_idx $IDX \
        --num_chunks $NUM_CHUNKS \
        --batch_size $BATCH_SIZE \
        --num_processes $NUM_PROCESSES \
        --gpu_id 0 > ./logs/log_gpu${IDX}.txt 2>&1 &
done

wait
echo "All 8 GPU processes finished."
