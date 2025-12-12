# ================================
# Makefile for MPI Reduce_scatter
# OpenMPI version (working setup)
# ================================

# Explicit OpenMPI tools (as tested)
MPICC   := /usr/bin/mpicc.openmpi
MPIRUN  := /usr/bin/mpirun.openmpi

# Build options
CFLAGS  := -O2 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS :=

# Program
TARGET  := reduce_scatter_8x8
SRC     := reduce_scatter_8x8.c

# Run configuration
NP      := 64
RUNOPTS := --oversubscribe

.PHONY: all run clean info

# Default target
all: $(TARGET)

# Build
$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run with 64 MPI processes
run: $(TARGET)
	$(MPIRUN) $(RUNOPTS) -np $(NP) ./$(TARGET)

# Print configuration (useful for debugging)
info:
	@echo "MPICC   = $(MPICC)"
	@echo "MPIRUN  = $(MPIRUN)"
	@echo "CFLAGS  = $(CFLAGS)"
	@echo "NP      = $(NP)"
	@echo "RUNOPTS = $(RUNOPTS)"

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o
