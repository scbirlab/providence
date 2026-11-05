#!/usr/bin/env bash

set -euox pipefail

DATASET="hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/downloaded"

HF_HOME="$CACHE" providence get "$DATASET" \
    --checkpoint "$OUTPUT_DIR"/got/out.pv \
    --output "$OUTPUT_DIR"/got/out.csv

HF_HOME="$CACHE" providence get "$DATASET" \
    --start 100 \
    --end 200 \
    --checkpoint "$OUTPUT_DIR"/got/out_100-200.pv \
    --output "$OUTPUT_DIR"/got/out_100-200.csv

if [ "$(diff "$OUTPUT_DIR"/got/out.csv "$OUTPUT_DIR"/got/out_100-200.csv | wc -l)" -eq 0 ]
then
    exit 1
fi

if [ ! "$(wc -l < "$OUTPUT_DIR"/got/out_100-200.csv)" -eq 101 ]
then
    exit 1
fi

HF_HOME="$CACHE" providence get "$DATASET" \
    --start 100 \
    --end 200 \
    --force-write-data \
    --checkpoint "$OUTPUT_DIR"/got/out_100-200_fs.pv

HF_HOME="$CACHE" providence get "$DATASET" \
    --start 200 \
    --end 300 \
    --checkpoint "$OUTPUT_DIR"/got/out_200-300.pv \
    --output "$OUTPUT_DIR"/got/out_200-300.csv

if [ "$(diff "$OUTPUT_DIR"/got/out_100-200.csv "$OUTPUT_DIR"/got/out_200-300.csv | wc -l)" -eq 0 ]
then
    exit 1
fi

