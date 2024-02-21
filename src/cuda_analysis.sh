#!/bin/bash

make
bin_path="./bin/"
# TODO: change those for your GPU config
# it should be the corresponding power of 2 for the number of blocks and threads
MaxNumOfBlocksPerGrid=10  # 2^10 = 1024
MaxNumOfThreadPerBlock=11 # 2^11 = 2048

for program in seq_array_cuda add_matrix_cuda mult_array_cuda; do
    echo "Running $program"
    log_path="output_cuda/$program/logs"
    plot_path="output_cuda/$program/plots"

    mkdir -p $log_path
    mkdir -p $plot_path

    for expNum in {1..5}; do
        for inputSize in 16 32 64 128 256 512 1024 2048; do
            for gridSize in $(seq 0 $MaxNumOfBlocksPerGrid); do
                curr_grid=$((2 ** gridSize))
                for blkSize in $(seq 0 $MaxNumOfThreadPerBlock); do
                    curr_blk=$((2 ** blkSize))
                    echo "Running $program with input size $inputSize, grid size $curr_grid, block size $curr_blk, expNum $expNum"
                    logfile="${program}_${inputSize}_${curr_grid}_${curr_blk}_${expNum}.csv"
                    nvprof --log-file ${log_path}/"$logfile" --csv ${bin_path}$program $inputSize "$curr_grid" "$curr_blk"
                    # python3 src/cuda_analysis.py "output_cuda/logs/$logfile" "output_cuda/plots/${logfile%.csv}".png
                done
            done
        done
    done
done
