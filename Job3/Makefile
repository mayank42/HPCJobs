NVCC = nvcc
CFLAGS = -Xcompiler -Wall -Werror cross-execution-space-call

matRedux: ./src/v0.0/MatReduxHost.cu ./build/matKernel.o
	$(NVCC) $(CFLAGS) -I/usr/include/cuda/ -I./include ./build/matKernel.o ./src/v0.0/MatReduxHost.cu -o ./bin/matRedux

./build/matKernel.o: ./src/v0.0/MatReduxKernel.cu
	$(NVCC) $(CFLAGS) -I/usr/include/cuda/ ./src/v0.0/MatReduxKernel.cu -c -o ./build/matKernel.o

      
run:
	./bin/matRedux

clean:
	rm -rf ./bin/*
	rm -rf ./build/*

