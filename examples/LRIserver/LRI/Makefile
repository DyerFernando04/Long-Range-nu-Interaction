# ===== Makefile para examples/LRI (Ubuntu/Debian con HDF5 "serial") =====
CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra \
           -I../../include -I../../src \
           -I/usr/include/hdf5/serial          # <-- aquí está H5public.h

LDFLAGS  = -L../../lib \
           -L/usr/lib/x86_64-linux-gnu/hdf5/serial  # <-- libs de HDF5 "serial"

# PRUEBA 1: nombres de libs comunes
LIBS     = -lnuSQuIDS -lSQuIDS -lgsl -lgslcblas -lhdf5 -lhdf5_hl -fopenmp -lm
# Si falla el link con 'undefined reference' a HDF5, usa PRUEBA 2:
# LIBS   = -lnuSQuIDS -lsquids -lgsl -lgslcblas -lhdf5_serial -lhdf5_serial_hl -fopenmp -lm

all: lri

lri: main.o
	$(CXX) $^ $(LDFLAGS) $(LIBS) -o $@

main.o: main.cpp LRI.h
	$(CXX) $(CXXFLAGS) -c main.cpp

run: all
	LD_LIBRARY_PATH=../../lib:$$LD_LIBRARY_PATH ./lri

clean:
	rm -f *.o lri

.PHONY: all run clean
