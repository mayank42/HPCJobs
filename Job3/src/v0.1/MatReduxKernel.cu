#include "kernelRedux.h"
__global__ void  row_kernel(double *imat , double *omat,size_t grids){
	__shared__ double  sdata [1024*4];
	unsigned  int tid = 4*threadIdx.x;
	unsigned  int i = 4*(blockIdx.x*blockDim.x + threadIdx.x);
	sdata[tid] = imat[i];
	sdata[tid+1] = imat[i+1];
	sdata[tid+2] = imat[i+2];
	sdata[tid+3] = imat[i+3];
	tid/=4;
	__syncthreads();
	unsigned int index;
	for(unsigned  int s=4;s<4*blockDim.x;s<<=1)
	{
		index = 2*s*tid;
		if (index<4*blockDim.x){
			sdata[index]+= sdata[index+s];
			sdata[index+1]+=sdata[index+1+s];
			sdata[index+2]+=sdata[index+2+s];
			sdata[index+3]+=sdata[index+3+s];
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
