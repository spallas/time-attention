#!/usr/bin/env bash
for fname in gen_confs/*.json; do
    echo "Processing $fname"
    python train.py --config $fname
done