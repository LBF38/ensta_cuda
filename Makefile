# Variables
CC = gcc
NVCC = nvcc
SRC_DIR = src
BIN_DIR = bin
CFLAGS = -I/usr/local/include
CUDAFLAGS = -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/lib
CUDALDFLAGS = -L/usr/local/cuda/lib64 -lcudart

# Find all .c and .cu files in the src directory
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Get the corresponding object files
C_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BIN_DIR)/%,$(C_SOURCES))
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(CUDA_SOURCES))

# Default rule
all: $(BIN_DIR) $(C_OBJECTS) $(CUDA_OBJECTS)

# Rule to make the bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to compile .c files
$(BIN_DIR)/%: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Rule to compile seq_array.c
seq_array: $(BIN_DIR)/seq_array

$(BIN_DIR)/seq_array: $(SRC_DIR)/seq_array.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Rule to compile .cu files
$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@ $(CUDALDFLAGS)

# Clean rule
clean:
	rm -rf $(BIN_DIR)
