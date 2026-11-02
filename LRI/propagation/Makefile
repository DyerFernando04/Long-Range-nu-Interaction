# ============================================================================
# LRI/propagation/Makefile
# ============================================================================
# Compila un UNICO propagador con tres modos independientes:
#   standard : solo oscilaciones estandar (g' = 0)
#   lri      : solo el modelo LRI elegido (NO calcula standard)
#   both     : ejecuta ambos modos y reutiliza checkpoints existentes
#
# No usa YAML. Ejecutar desde la raiz del repositorio con:
#   make -C LRI/propagation
#   make -C LRI/propagation smoke
#   make -C LRI/propagation standard THREADS=32
#   make -C LRI/propagation lri MODEL=Le-Lmu THREADS=32
#
# Opciones adicionales: EXTRA_ARGS="--mass-begin 0 --mass-end 1 ..."
# ============================================================================

CXX            ?= g++
PKG_CONFIG     ?= pkg-config

TARGET         := generate_flux_lut
SOURCE         := main.cpp
HEADERS        := LRI.h EarthDepth.h AtmosphericSource.h ../potential/PotentialLibrary.h
REPO_ROOT      := ../..

# Includes y bibliotecas de nuSQuIDS y HDF5 instalados en el sistema.
NUSQ_CFLAGS    := $(shell $(PKG_CONFIG) --cflags nusquids 2>/dev/null)
NUSQ_LIBS      := $(shell $(PKG_CONFIG) --libs nusquids 2>/dev/null)
HDF5_CFLAGS    := $(shell $(PKG_CONFIG) --cflags hdf5 2>/dev/null)
HDF5_LIBS      := $(shell $(PKG_CONFIG) --libs hdf5 2>/dev/null)

CPPFLAGS       += $(NUSQ_CFLAGS) $(HDF5_CFLAGS)
CXXFLAGS       += -O3 -std=c++17 -fopenmp -Wall -Wextra -Wpedantic
LDLIBS         += $(NUSQ_LIBS) -lhdf5_hl $(HDF5_LIBS) -lgsl -lgslcblas

# Opciones de ejecucion. Las rutas son relativas a REPO_ROOT.
MODEL          ?= Le-Lmu
THREADS        ?= 4
MCEQ           ?= data/atmospheric/mceq_gsf2019_ddm_height.h5
POTENTIAL      ?= results/potential/vlri_unit.h5
EXTRA_ARGS     ?=

OVERWRITE      ?= 0
OVERWRITE_FLAG := $(if $(filter 1 yes true on,$(OVERWRITE)),--overwrite,)

COMMON_ARGS    := --mceq $(MCEQ) --threads $(THREADS) $(OVERWRITE_FLAG) $(EXTRA_ARGS)
LRI_ARGS       := --potential $(POTENTIAL) --model $(MODEL)

.PHONY: all check-deps help info smoke smoke-standard standard lri both run clean

all: $(TARGET)

check-deps:
	@$(PKG_CONFIG) --exists nusquids || { \
	    echo "ERROR: pkg-config no encuentra nuSQuIDS (nusquids.pc)."; \
	    echo "Comprueba PKG_CONFIG_PATH y la instalacion de nuSQuIDS."; \
	    exit 1; \
	}
	@$(PKG_CONFIG) --exists hdf5 || { \
	    echo "ERROR: pkg-config no encuentra HDF5 (hdf5.pc)."; \
	    exit 1; \
	}

$(TARGET): $(SOURCE) $(HEADERS)
	@$(MAKE) --no-print-directory check-deps
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDLIBS)

# Tests pequenos (el smoke LRI no ejecuta el caso estandar).
smoke: $(TARGET)
	cd $(REPO_ROOT) && ./LRI/propagation/$(TARGET) --mode lri $(LRI_ARGS) --smoke $(COMMON_ARGS)

smoke-standard: $(TARGET)
	cd $(REPO_ROOT) && ./LRI/propagation/$(TARGET) --mode standard --smoke $(COMMON_ARGS)

# Cada modo es independiente. Los resultados se guardan/reanudan en HDF5.
standard: $(TARGET)
	cd $(REPO_ROOT) && ./LRI/propagation/$(TARGET) --mode standard $(COMMON_ARGS)

lri: $(TARGET)
	cd $(REPO_ROOT) && ./LRI/propagation/$(TARGET) --mode lri $(LRI_ARGS) $(COMMON_ARGS)

both: $(TARGET)
	cd $(REPO_ROOT) && ./LRI/propagation/$(TARGET) --mode both $(LRI_ARGS) $(COMMON_ARGS)

# Alias del modo predeterminado: SOLO LRI.
run: lri

info:
	@echo "Compiler:     $(CXX)"
	@echo "nuSQuIDS:     $$($(PKG_CONFIG) --modversion nusquids 2>/dev/null || echo NO_ENCONTRADO)"
	@echo "HDF5:        $$($(PKG_CONFIG) --modversion hdf5 2>/dev/null || echo NO_ENCONTRADO)"
	@echo "Model:        $(MODEL)"
	@echo "Threads:      $(THREADS)"
	@echo "MCEq file:    $(MCEQ)"
	@echo "Potential:    $(POTENTIAL)"
	@echo "Overwrite:    $(OVERWRITE)"

help:
	@echo "Compilar:         make -C LRI/propagation"
	@echo "Prueba LRI:       make -C LRI/propagation smoke MODEL=Le-Lmu"
	@echo "Prueba standard:  make -C LRI/propagation smoke-standard"
	@echo "Solo standard:    make -C LRI/propagation standard THREADS=32"
	@echo "Solo LRI:         make -C LRI/propagation lri MODEL=Le-Lmu THREADS=32"
	@echo "Ambos:           make -C LRI/propagation both MODEL=Le-Lmu THREADS=32"
	@echo "Opciones extra:   EXTRA_ARGS='--mass-begin 0 --mass-end 1 --coupling-begin 0 --coupling-end 1'"
	@echo "Para Lmu-Ltau:   MODEL=Lmu-Ltau EXTRA_ARGS='--coupling-kind bigg'"
	@echo "Limpiar:         make -C LRI/propagation clean"

clean:
	rm -f $(TARGET)
