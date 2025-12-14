# ======================================================
# Makefile: Gauss + MPI-IO checkpoint + ULFM + warm spares (scenario a)
# ======================================================

OMPI_PREFIX := $(HOME)/opt/openmpi-ulfm-api
MPICC       := $(OMPI_PREFIX)/bin/mpicc
MPIRUN      := $(OMPI_PREFIX)/bin/mpirun

CFLAGS  := -O2 -std=c11 -Wall -Wextra -Wpedantic -DUSE_ULFM
LDFLAGS :=

TARGET := gauss_spares_ft
SRC    := gauss_spares_ft.c

WORK   := 6
SPARES := 2
NP     := $(shell echo $$(( $(WORK) + $(SPARES) )) )

CKPT := /tmp/gauss_spares.ckpt

# normal run (no FT framework)
RUNOPTS := --oversubscribe --mca io ompio

# FT run:
# - enable ULFM
# - keep ompio (works in your build)
# - force coll modules away from tuned/ucc/han (often needed for ULFM)
# - disable help aggregation for clearer logs
RUNOPTS_FT := --oversubscribe --with-ft ulfm --mca io ompio \
             --mca coll ^han,ucc,tuned \
             --mca orte_base_help_aggregate 0

.PHONY: all run run_ft fail_ft info clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	$(MPIRUN) $(RUNOPTS) -np $(NP) ./$(TARGET) data.in $(CKPT) $(WORK)

run_ft: $(TARGET)
	$(MPIRUN) $(RUNOPTS_FT) -np $(NP) ./$(TARGET) data.in $(CKPT) $(WORK)

fail_ft: $(TARGET)
	FAIL_RANK=2 FAIL_PHASE=0 FAIL_STEP=3 \
	$(MPIRUN) $(RUNOPTS_FT) -np $(NP) ./$(TARGET) data.in $(CKPT) $(WORK)

info:
	@echo "WORK=$(WORK) SPARES=$(SPARES) NP=$(NP)"
	@echo "CKPT=$(CKPT)"
	@echo "RUNOPTS=$(RUNOPTS)"
	@echo "RUNOPTS_FT=$(RUNOPTS_FT)"

clean:
	rm -f $(TARGET) *.o
	rm -f $(CKPT)
