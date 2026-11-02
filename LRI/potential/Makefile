
# ============================================================
# LRI/potential/Makefile
# ============================================================

CXX := g++

TARGET := build_vlri_lut
SRC := build_vlri_lut.cpp

# nuSQuIDS
NUSQ_CFLAGS := $(shell pkg-config --cflags nusquids)
NUSQ_LIBS := $(shell pkg-config --libs nusquids)

# Compiler flags
CPPFLAGS := $(NUSQ_CFLAGS)

CXXFLAGS := -O3 -std=c++17 -fopenmp \
            -Wall -Wextra -Wpedantic

# Libraries
LDLIBS := $(NUSQ_LIBS) -lhdf5_hl -lgsl -lgslcblas

# Output path relative to repository root
OUTPUT := results/potential/vlri_unit.h5

# Optional custom Earth model
EARTH_MODEL ?=

.PHONY: all run clean info

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

run: $(TARGET)
	cd ../.. && ./LRI/potential/$(TARGET) $(OUTPUT) $(EARTH_MODEL)

info:
	@echo "Compiler: $(CXX)"
	@echo "nuSQuIDS version:"
	@pkg-config --modversion nusquids
	@echo "Output: $(OUTPUT)"
	@echo "OpenMP threads: $${OMP_NUM_THREADS:-default}"

clean:
	rm -f $(TARGET)
