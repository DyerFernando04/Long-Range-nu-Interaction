# ===== Makefile para examples/LRI (Ubuntu/Debian con HDF5 "serial") =====

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra \
            -I../../include -I../../src \
            -I/usr/include/hdf5/serial

# Rutas de librerías para el linker (compile/link time)
LDFLAGS  := -L../../lib \
            -L/usr/lib/x86_64-linux-gnu/hdf5/serial

# ===== CLAVE: embebemos un RUNPATH en el ejecutable =====
# $ORIGIN = directorio donde está el ejecutable (lri). Así es portable.
# Desde examples/LRIfixedAngle/lri, ../../lib es la lib de nuSQuIDS.
RPATH    := -Wl,-rpath,'$$ORIGIN/../../lib'

LIBS     := -lnuSQuIDS -lSQuIDS -lgsl -lgslcblas -lhdf5 -lhdf5_hl -fopenmp -lm

TARGET   := lri
OBJS     := main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $^ $(LDFLAGS) $(RPATH) $(LIBS) -o $@

main.o: main.cpp LRI.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o $@

# Opcional: sigue existiendo "make run", pero ya no es necesario
run: all
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run clean

