#include "job2a.h"
int main(){
	int **h_A = (int**)malloc(size*sizeof(int*));
	int **h_R = (int**)malloc(size*sizeof(int*));
	int a,b;
	cudaError_t err;
	for(a=0;a<size;a++)h_A[a]=(int*)malloc(size*sizeof(int));
	for(a=0;a<size;a++)h_R[a]=(int*)malloc(size*sizeof(int));
	for(a=0;a<size;a++){
		for(b=0;b<size;b++){
			h_A[a][b]=rand();
		}
	}
	int * d_A;
	err = cudaMalloc((void**)&d_A,size*size*sizeof(int));
	debug(err,"Unable to allocate matrix on device")
	//copy
	for(a=0;a<size;a++){
		err = cudaMemcpy(d_A+a*size,h_A[a],size*sizeof(int),cudaMemcpyHostToDevice);
		debug(err,"Failed to copy to device")
	}
	printf("Orignal Matrix:\n");
	fflush(stdout);
	for(a=0;a<size;a++){
		for(b=0;b<size;b++){
			printf("%d ",h_A[a][b]);
		}
		printf("\n");
	}
	fflush(stdout);
	dim3 blockdim(8,8,1);
	int gridx = (int)ceil(8.0/size);
	dim3 griddim(gridx,gridx,1);
	printf("Swapping...");
	swap<<<griddim,blockdim>>>(d_A,size);
	printf("Done.\n");
	fflush(stdout);
	err = cudaGetLastError();
	debug(err,"Last error in execution")
	for(a=0;a<size;a++){
		err = cudaMemcpy(h_R[a],d_A+a*size,size*sizeof(int),cudaMemcpyDeviceToHost);
		debug(err,"Failed to copy back to host")
	}
	printf("Swapped array:\n");
	for(a=0;a<size;a++){
		for(b=0;b<size;b++){
			printf("%d ",h_R[a][b]);
		}
		printf("\n");
	}
	fflush(stdout);
	for(a=0;a<size;a++){
		free(h_A[a]);
		free(h_R[a]);
	}
	free(h_A);
	free(h_R);
	err = cudaFree(d_A);
	debug(err,"Failed to free memory on device")
	err = cudaDeviceReset();
	debug(err,"Unable to reset device")
	printf("Last err: %d\n",err);
	return 0;
}

		
	
	
