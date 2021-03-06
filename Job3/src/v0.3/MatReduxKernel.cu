#include "kernelRedux.h"
__global__ void  row_kernel(double *imat , double *omat,size_t grids){
	__shared__ double  sdata [1024*4];
	unsigned  int tid = 4*threadIdx.x;
	unsigned  int i = 4*(blockIdx.x*blockDim.x*2 + threadIdx.x);
	if(blockIdx.x*2+1>=gridDim.x)return;
	else if(blockIdx.x*2+3==gridDim.x){
		sdata[tid] = imat[i] + imat[i+4*blockDim.x] + imat[i+8*blockDim.x];
		sdata[tid+1] = imat[i+1] + imat[i+4*blockDim.x+1] + imat[i+8*blockDim.x+1];
		sdata[tid+2] = imat[i+2] + imat[i+4*blockDim.x+2] + imat[i+8*blockDim.x+2];
		sdata[tid+3] = imat[i+3] + imat[i+4*blockDim.x+3] + imat[i+8*blockDim.x+3];

	}
	else{	
		sdata[tid] = imat[i]+imat[i+4*blockDim.x];
		sdata[tid+1] = imat[i+1] + imat[i+4*blockDim.x+1];
		sdata[tid+2] = imat[i+2] + imat[i+4*blockDim.x+2];
		sdata[tid+3] = imat[i+3] + imat[i+4*blockDim.x+3];
	}
	__syncthreads();
	for(unsigned  int s=2*blockDim.x;s>=4;s>>=1)
	{
		if (tid<s){
			sdata[tid]+= sdata[tid+s];
			sdata[tid+1]+=sdata[tid+1+s];
			sdata[tid+2]+=sdata[tid+2+s];
			sdata[tid+3]+=sdata[tid+3+s];
		}
		__syncthreads();
	}
	if (tid ==0){
		unsigned int bid = 4*blockIdx.x;
		omat[bid] = sdata [0];
		omat[bid+1] = sdata[1];
		omat[bid+2] = sdata[2];
		omat[bid+3] = sdata[3];
	}
}
