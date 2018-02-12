NVCC = nvcc

Job2A: job2a.h kernel.cu host.cu  
	$(NVCC) -I./ -I/usr/include/cuda -L/usr/lib64/cuda kernel.cu host.cu -o Job2A

      
run:
	./Job2A

clean:
	rm -rf Job2A

