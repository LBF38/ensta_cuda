#!/bin/bash

make
program="./bin/seq_array_cuda" # TODO: change it for your program
# nvprof="/usr/local/cuda/bin/nvprof"
# TODO: change those for your GPU config
# it should be the corresponding power of 2 for the number of blocks and threads
MaxNumOfBlocksPerGrid=10  # 2^10 = 1024
MaxNumOfThreadPerBlock=10 # 2^10 = 1024

mkdir -p output_cuda/logs
mkdir -p output_cuda/plots

for expNum in {1..5}; do
    for inputSize in 16 32 64 128 256 512 1024 2048; do
        for gridSize in $(seq 0 $MaxNumOfBlocksPerGrid); do
            curr_grid=$((2 ** gridSize))
            for blkSize in $(seq 0 $MaxNumOfThreadPerBlock); do
                curr_blk=$((2 ** blkSize))
                echo "Running $program with input size $inputSize, grid size $curr_grid, block size $curr_blk, expNum $expNum"
                logfile="array_add_${inputSize}_${curr_grid}_${curr_blk}_${expNum}.csv"
                nvprof --log-file "output_cuda/logs/$logfile" --csv $program $inputSize "$curr_grid" "$curr_blk"
                # python3 src/cuda_analysis.py "output_cuda/logs/$logfile" "output_cuda/plots/${logfile%.csv}".png
            done
        done
    done
done
