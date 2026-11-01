# =============================================================================
# LUT_FLUX Makefile
# =============================================================================
#
# Builds:
#
#   generate_flux_lut
#       -> full LUT2 grid
#
#   generate_flux_lut_smoke
#       -> small 2x2 smoke-test grid
#
# The smoke executable is compiled with:
#
#   -DLRI_SMOKE_TEST
#
# so main.cpp selects the reduced parameter grid.
# =============================================================================


CXX := g++

TARGET := generate_flux_lut
SMOKE_TARGET := generate_flux_lut_smoke

SRC := main.cpp

HEADERS := \
	LRI.h \
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
# Default build
# -----------------------------------------------------------------------------
#
# Running simply:
#
#   make
#
# builds BOTH executables.
# -----------------------------------------------------------------------------

.PHONY: all full smoke clean info

all: $(TARGET) $(SMOKE_TARGET)


# -----------------------------------------------------------------------------
# Full LUT2 executable
# -----------------------------------------------------------------------------

full: $(TARGET)


$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
		$(SRC) \
		$(LDFLAGS) $(LDLIBS) \
		-o $(TARGET)


# -----------------------------------------------------------------------------
# Smoke-test executable
# -----------------------------------------------------------------------------
#
# The only difference is:
#
#   -DLRI_SMOKE_TEST
#
# main.cpp must contain:
#
#   #ifdef LRI_SMOKE_TEST
#       ... small grid ...
#   #else
#       ... full grid ...
#   #endif
#
# -----------------------------------------------------------------------------

smoke: $(SMOKE_TARGET)


$(SMOKE_TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
		-DLRI_SMOKE_TEST \
		$(SRC) \
		$(LDFLAGS) $(LDLIBS) \
		-o $(SMOKE_TARGET)


# -----------------------------------------------------------------------------
# Show configuration
# -----------------------------------------------------------------------------

info:
	@echo "============================================================"
	@echo " LUT_FLUX build configuration"
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
	@echo "Executables:"
	@echo "  Full  : $(TARGET)"
	@echo "  Smoke : $(SMOKE_TARGET)"
	@echo "============================================================"


# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------

clean:
	rm -f $(TARGET) $(SMOKE_TARGET)
