NVCC = nvcc
ifndef TEST
TEST = test2
endif

Job2B: $(TEST).cu
	$(NVCC) -I/usr/include/cuda/ -o $(TEST) $(TEST).cu

      
run:
	./$(TEST)

clean:
	rm -rf $(TEST)
