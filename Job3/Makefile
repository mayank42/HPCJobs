NVCC = nvcc
CC = g++
CUDACFLAGS = -Xcompiler -fopenmp -Xcompiler -Wall -Werror cross-execution-space-call
CFLAGS = -Wall -Werror
VERSION = v0.4

matRedux: ./src/$(VERSION)/MatReduxHost.cu ./build/matKernel.o ./build/cpuRedux.o ./build/stackTimer.o
	$(NVCC) $(CUDACFLAGS) -I/usr/include/cuda/ -I./include ./build/matKernel.o ./build/cpuRedux.o ./build/stackTimer.o ./src/$(VERSION)/MatReduxHost.cu -o ./bin/matRedux

./build/matKernel.o: ./src/$(VERSION)/MatReduxKernel.cu
	$(NVCC) $(CUDACFLAGS) -I/usr/include/cuda/ -I./include ./src/$(VERSION)/MatReduxKernel.cu -c -o ./build/matKernel.o

./build/cpuRedux.o: ./src/$(VERSION)/cpuRedux.cpp
	$(CC) $(CFLAGS)  -I./include -c ./src/$(VERSION)/cpuRedux.cpp -o ./build/cpuRedux.o

./build/stackTimer.o: ./src/$(VERSION)/stackTimer.cpp
	$(CC) $(CFLAGS) -I./include -c ./src/$(VERSION)/stackTimer.cpp -o ./build/stackTimer.o

run:
	./bin/matRedux

clean:
	rm -rf ./bin/*
	rm -rf ./build/*

