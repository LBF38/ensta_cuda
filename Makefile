# Variables
NVCC = nvcc
SRC_DIR = src
BIN_DIR = bin
CFLAGS = -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

# Find all .cu files in the src directory
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
# Get the corresponding object files
OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))

# Default rule
all: $(BIN_DIR) $(OBJECTS)

# Rule to make the bin directory
$(BIN_DIR):
    mkdir -p $(BIN_DIR)

# Rule to compile .cu files
$(BIN_DIR)/%: $(SRC_DIR)/%.cu
    $(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean rule
clean:
    rm -rf $(BIN_DIR)
