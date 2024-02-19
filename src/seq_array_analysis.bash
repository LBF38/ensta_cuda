#!/bin/bash

make seq_array

for i in {4..11}; do
    size=$((2 ** i))
    for _ in {1..5}; do
        ./bin/seq_array $size >>output.txt
    done
done

python3 src/seq_array_analysis.py output.txt seq_array.png
rm output.txt
