NVCC = nvcc

matRedux: ./src/v0.0/MatReduxHost.cu ./src/v0.0/MatReduxKernel.cu  
	$(NVCC) -I/usr/include/cuda/ -I./include ./src/v0.0/MatReduxKernel.cu ./src/v0.0/MatReduxHost.cu -o ./bin/matRedux

      
run:
	./bin/matRedux

clean:
	rm -rf ./bin/*

