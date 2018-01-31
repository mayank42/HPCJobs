NVCC = nvcc

Job1: job1.h kernel.cu host.cu
	$(NVCC) -I ./ kernel.cu host.cu -o Job1


run:
	./Job1

clean:
	rm -rf Job1
