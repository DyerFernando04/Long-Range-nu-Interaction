# =============================================================================
# STD_FLUX Makefile
# =============================================================================
#
# Builds:
#
#   generate_flux_std
#
# Sources:
#
#   main.cpp
#   EarthDepth.h
#
# Physics:
#
#   Standard nuSQuIDS propagation only:
#
#       H = H_vac + H_standard_matter
#
# No LRI potential is included.
#
# OpenMP is used to parallelize over zenith-angle propagations.
# =============================================================================


CXX := g++

TARGET := generate_flux_std

SRC := main.cpp

HEADERS := \
	EarthDepth.h


# -----------------------------------------------------------------------------
# nuSQuIDS configuration
# -----------------------------------------------------------------------------

NUSQ_CFLAGS := $(shell pkg-config --cflags nusquids)
NUSQ_LIBS   := $(shell pkg-config --libs nusquids)


# -----------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------

CPPFLAGS := $(NUSQ_CFLAGS)

CXXFLAGS := \
	-O3 \
	-std=c++17 \
	-fopenmp \
	-Wall \
	-Wextra \
	-Wpedantic


# -----------------------------------------------------------------------------
# Linker flags
# -----------------------------------------------------------------------------

LDFLAGS := \
	-fopenmp


# HDF5 High-Level is required by libnuSQuIDS itself.
LDLIBS := \
	$(NUSQ_LIBS) \
	-lhdf5_hl


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

.PHONY: all clean info

all: $(TARGET)


$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
		$(SRC) \
		$(LDFLAGS) $(LDLIBS) \
		-o $(TARGET)


# -----------------------------------------------------------------------------
# Show configuration
# -----------------------------------------------------------------------------

info:
	@echo "============================================================"
	@echo " STD_FLUX build configuration"
	@echo "============================================================"
	@echo "Compiler:"
	@echo "  $(CXX)"
	@echo
	@echo "nuSQuIDS version:"
	@pkg-config --modversion nusquids
	@echo
	@echo "nuSQuIDS CFLAGS:"
	@echo "  $(NUSQ_CFLAGS)"
	@echo
	@echo "nuSQuIDS LIBS:"
	@echo "  $(NUSQ_LIBS)"
	@echo
	@echo "CXXFLAGS:"
	@echo "  $(CXXFLAGS)"
	@echo
	@echo "Executable:"
	@echo "  $(TARGET)"
	@echo "============================================================"


# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------

clean:
	rm -f $(TARGET)
