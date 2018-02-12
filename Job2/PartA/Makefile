NVCC = nvcc

Job2A: job2a.h kernel.cu host.cu  
	$(NVCC) -I ./ kernel.cu host.cu -o Job2A

      
run:
	./Job2A

clean:
	rm -rf Job2A

