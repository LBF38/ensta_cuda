#!/bin/bash

# Compile the program
make seq_array

# Run the program with different array sizes
for i in {4..11}; do
    size=$((2 ** i))
    for _ in {1..5}; do
        ./bin/seq_array $size
    done
done
