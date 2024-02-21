#!/bin/bash

make
program="./bin/seq_array_cuda" # TODO: change it for your program
# nvprof="/usr/local/cuda/bin/nvprof"
MaxNumOfBlocksPerGrid=1024  # TODO: change it for your GPU config
MaxNumOfThreadPerBlock=1024 # TODO: change it for your GPU config

mkdir -p output_cuda/logs
mkdir -p output_cuda/plots

for expNum in {1..5}; do
    for inputSize in 16 32 64 128 256 512 1024 2048; do
        for gridSize in $(seq 1 $MaxNumOfBlocksPerGrid); do
            for blkSize in $(seq 1 $MaxNumOfThreadPerBlock); do
                echo "Running $program with input size $inputSize, grid size $gridSize, block size $blkSize, expNum $expNum"
                logfile="array_add_${inputSize}_${gridSize}_${blkSize}_${expNum}.csv"
                nvprof --log-file "output_cuda/logs/$logfile" --csv $program $inputSize "$gridSize" "$blkSize"
                # python3 src/cuda_analysis.py "output_cuda/logs/$logfile" "output_cuda/plots/${logfile%.csv}".png
            done
        done
    done
done
