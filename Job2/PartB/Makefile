NVCC = nvcc

Job2B: job2b.h kernel.cu host.cu  
	$(NVCC) -I/usr/include/cuda/ -I./ kernel.cu host.cu -o Job2B

      
run:
	./Job2B

clean:
	rm -rf Job2B

