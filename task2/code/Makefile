# ======================================================
# Makefile: Gaussian Elimination with MPI-IO + ULFM
# Scenario (b): spawn new MPI processes after failure
# OpenMPI (built with --with-ft=ulfm)
# ======================================================

# ---- OpenMPI ULFM installation ----
OMPI_PREFIX := $(HOME)/opt/openmpi-ulfm-api
MPICC       := $(OMPI_PREFIX)/bin/mpicc
MPIRUN      := $(OMPI_PREFIX)/bin/mpirun

# ---- Compilation flags ----
CFLAGS  := -O2 -std=c11 -Wall -Wextra -Wpedantic -DUSE_ULFM
LDFLAGS :=

# ---- Target ----
TARGET := gauss_ft_mpiio
SRC    := gauss_ft_mpiio.c

# ---- Runtime parameters ----
NP := 6

# ВАЖНО:
#  - --with-ft ulfm  : включаем ULFM
#  - --mca io ^ompio : отключаем OMPIO (MPI-IO стабильнее с ROMIO)
RUNOPTS := --oversubscribe --with-ft ulfm --mca io ompio


# ---- Phony targets ----
.PHONY: all run fail info clean

# ---- Build ----
all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ---- Normal run ----
run: $(TARGET)
	$(MPIRUN) $(RUNOPTS) -np $(NP) ./$(TARGET)

# ---- Run with simulated failure ----
# FAIL_RANK  – номер MPI-процесса, который "падает"
# FAIL_PHASE – 0: elimination, 1: back substitution
# FAIL_STEP  – номер шага алгоритма
fail: $(TARGET)
	FAIL_RANK=2 FAIL_PHASE=0 FAIL_STEP=3 \
	$(MPIRUN) $(RUNOPTS) -np $(NP) ./$(TARGET)

# ---- Info ----
info:
	@echo "OMPI_PREFIX = $(OMPI_PREFIX)"
	@echo "MPICC       = $(MPICC)"
	@echo "MPIRUN      = $(MPIRUN)"
	@echo "TARGET      = $(TARGET)"
	@echo "NP          = $(NP)"
	@echo "RUNOPTS     = $(RUNOPTS)"

# ---- Clean ----
clean:
	rm -f $(TARGET) *.o gauss.ckpt
