#!/bin/bash

make

bin_path="./bin/"
py_prog="src/analysis.py"

for prog in seq_array add_matrix mult_array; do
    echo "Running $prog"
    for i in {4..11}; do
        size=$((2 ** i))
        for _ in {1..5}; do
            ${bin_path}$prog $size >>output_$prog.txt
        done
    done
    echo "Plotting $prog"
    python3 $py_prog output_$prog.txt ${prog}_analysis.png
    echo "Cleaning up"
    rm output_$prog.txt
    printf "Done with %s\n" $prog
done
