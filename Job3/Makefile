NVCC = nvcc

matRedux: ./src/v0.0/MatReduxHost.cu ./build/matKernel.o
	$(NVCC) -I/usr/include/cuda/ -I./include ./build/matKernel.o ./src/v0.0/MatReduxHost.cu -o ./bin/matRedux

./build/matKernel.o: ./src/v0.0/MatReduxKernel.cu
	$(NVCC) -I/usr/include/cuda/  ./src/v0.0/MatReduxKernel.cu -c -o ./build/matKernel.o

      
run:
	./bin/matRedux

clean:
	rm -rf ./bin/*
	rm -rf ./build/*
