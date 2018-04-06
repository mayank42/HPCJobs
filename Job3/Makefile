NVCC = nvcc
CC = g++
CUDACFLAGS = -Xcompiler -fopenmp -Xcompiler -Wall -Werror cross-execution-space-call
CFLAGS = -Wall -Werror

matRedux: ./src/v0.0/MatReduxHost.cu ./build/matKernel.o ./build/cpuRedux.o ./build/stackTimer.o
	$(NVCC) $(CUDACFLAGS) -I/usr/include/cuda/ -I./include ./build/matKernel.o ./build/cpuRedux.o ./build/stackTimer.o ./src/v0.0/MatReduxHost.cu -o ./bin/matRedux

./build/matKernel.o: ./src/v0.0/MatReduxKernel.cu
	$(NVCC) $(CUDACFLAGS) -I/usr/include/cuda/ -I./include ./src/v0.0/MatReduxKernel.cu -c -o ./build/matKernel.o

./build/cpuRedux.o: ./src/v0.0/cpuRedux.cpp
	$(CC) $(CFLAGS)  -I./include -c ./src/v0.0/cpuRedux.cpp -o ./build/cpuRedux.o

./build/stackTimer.o: ./src/v0.0/stackTimer.cpp
	$(CC) $(CFLAGS) -I./include -c ./src/v0.0/stackTimer.cpp -o ./build/stackTimer.o

run:
	./bin/matRedux

clean:
	rm -rf ./bin/*
	rm -rf ./build/*

