#include "job2b.h"
int main(){
	int gridx,gridy;
	printf("Running 1D operation:\n");
	fflush(stdout);
	//Host arrays
	int *h_arr = (int*)malloc(oneDsize*sizeof(int));
	int h_oneDmask[5]={1,1,0,1,1};
	int *h_oneDres=(int*)malloc(oneDsize*sizeof(int));
	int a,b;
	for(a=0;a<oneDsize;a++)h_arr[a]=(int)(INPUT_MAX*(float)rand()/RAND_MAX);
	//Device arrays
	cudaError_t err;
	int *d_arr;
	int *d_oneDmask;
	int *d_oneDres;
	err=cudaMalloc((void**)&d_arr,oneDsize*sizeof(int));
	//debug
	//copy
	err=cudaMemcpy(d_arr,h_arr,oneDsize*sizeof(int),cudaMemcpyHostToDevice);
	//debug
	err=cudaMalloc((void**)&d_oneDres,oneDsize*sizeof(int));
	//debug
	err=cudaMalloc((void**)&d_oneDmask,5*sizeof(int));
	//debug
	//copy
	err=cudaMemcpy(d_oneDmask,h_oneDmask,5*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 block(1,64,1);
	gridy = (int)ceil(oneDsize/64.0);
	dim3 grid(1,gridy,1);
	printf("Convolving...");
	conv1D<<<grid,block>>>(d_arr,d_oneDmask,d_oneDres,oneDsize,5,2);
	printf("Done.\n");
	fflush(stdout);
	err=cudaGetLastError();
	//debug
	//copy
	err=cudaMemcpy(h_oneDres,d_oneDres,oneDsize*sizeof(int),cudaMemcpyDeviceToHost);
	printf("Array:\n");
	for(a=0;a<oneDsize;a++)printf("%d ",h_arr[a]);
	printf("\n");
	printf("Convolution:\n");
	for(a=0;a<oneDsize;a++)printf("%d ",h_oneDres[a]);
	printf("\n");
	fflush(stdout);
	free(h_arr);
	free(h_oneDres);
	err=cudaFree(d_arr);
	//debug
	cudaFree(d_oneDres);
	//debug
	cudaFree(d_oneDmask);
	//debug
	/***********************************************************************************
	*										   *
	*      TWO D CONVOLUTION							   *
	*										   *
	***********************************************************************************/
	int **h_A = (int**)malloc(twoDsize1*sizeof(int*));
	float **h_R = (float**)malloc(twoDsize1*sizeof(float*));
	float h_twoDmask[3][3] = {{0.125,0.125,0.125},{0.125,0.0,0.125},{0.125,0.125,0.125}};
	for(a=0;a<twoDsize1;a++)h_A[a]=(int*)malloc(twoDsize2*sizeof(int));
	for(a=0;a<twoDsize1;a++)h_R[a]=(float*)malloc(twoDsize2*sizeof(float));
	for(a=0;a<twoDsize1;a++){
		for(b=0;b<twoDsize2;b++){
			h_A[a][b]=(int)(INPUT_MAX*(float)rand()/RAND_MAX);
		}
	}
	int *d_A;
	float *d_R;
	float *d_twoDmask;
	err = cudaMalloc((void**)&d_A,twoDsize1*twoDsize2*sizeof(int));
	//debug
	//copy
	for(a=0;a<twoDsize1;a++){
		err = cudaMemcpy(d_A+a*twoDsize2,h_A[a],twoDsize2*sizeof(int),cudaMemcpyHostToDevice);
		//debug
	}
	err = cudaMalloc((void**)&d_R,twoDsize1*twoDsize2*sizeof(float));
	//debug
	//copy
	for(a=0;a<twoDsize1;a++){
		err = cudaMemcpy(d_R+a*twoDsize2,h_R[a],twoDsize2*sizeof(float),cudaMemcpyHostToDevice);
		//debug
	}
	err = cudaMalloc((void**)&d_twoDmask,3*3*sizeof(float));
	//debug
	//copy
	for(a=0;a<3;a++){
		err = cudaMemcpy(d_twoDmask+a*3,h_twoDmask[a],3*sizeof(float),cudaMemcpyHostToDevice);
		//debug
	}
	printf("Orignal Matrix:\n");
	fflush(stdout);
	for(a=0;a<twoDsize1;a++){
		for(b=0;b<twoDsize2;b++){
			printf("%d ",h_A[a][b]);
		}
		printf("\n");
	}
	fflush(stdout);
	dim3 blockdim(8,8,1);
	gridx = (int)ceil(twoDsize1/8.0);
	gridy = (int)ceil(twoDsize2/8.0);
	dim3 griddim(gridx,gridy,1);
	printf("Convolving...");
	conv2D<<<griddim,blockdim>>>(d_A,d_twoDmask,d_R,twoDsize1,twoDsize2,3,3,1,1);
	printf("Done.\n");
	fflush(stdout);
	err = cudaGetLastError();
	//debug
	for(a=0;a<twoDsize1;a++){
		err = cudaMemcpy(h_R[a],d_R+a*twoDsize2,twoDsize2*sizeof(float),cudaMemcpyDeviceToHost);
		//debug
	}
	printf("Convolution:\n");
	fflush(stdout);
	for(a=0;a<twoDsize1;a++){
		for(b=0;b<twoDsize2;b++){
			printf("%f ",h_R[a][b]);
		}
		printf("\n");
	}
	fflush(stdout);
	for(a=0;a<twoDsize1;a++){
		free(h_A[a]);
		free(h_R[a]);
	}
	free(h_A);
	free(h_R);
	err = cudaFree(d_A);
	//debug
	err = cudaFree(d_R);
	//debug
	err = cudaFree(d_twoDmask);
	//debug
	err = cudaDeviceReset();
	//debug
	printf("Last err: %d\n",err);
	return 0;
}	
	
	
