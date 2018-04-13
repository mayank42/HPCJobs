#include "kernelRedux.h"
__global__ void  row_kernel(double *imat , double *omat,size_t grids){
	__shared__ double  sdata [1024*4];
	unsigned  int tid = 4*threadIdx.x;
	unsigned  int i = 4*(blockIdx.x*blockDim.x + threadIdx.x);
	sdata[tid] = imat[i];
	sdata[tid+1] = imat[i+1];
	sdata[tid+2] = imat[i+2];
	sdata[tid+3] = imat[i+3];
	__syncthreads();
	for(unsigned  int s=4;s<4*blockDim.x;s*= 2)
	{
		if (tid  %(2*s)==0){
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
__global__ void  col_kernel(double *imat , double *omat,size_t length){
	__shared__ double  sdata [1024*4];
	unsigned  int tid = threadIdx.x;
	unsigned  int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = imat[i];
	sdata[tid+1*1024] = imat[i+1*length];
	sdata[tid+2*1024] = imat[i+2*length];
	sdata[tid+3*1024] = imat[i+3*length];
	__syncthreads();
	for(unsigned  int s=1;s<blockDim.x;s*= 2)
	{
		if (tid  %(2*s)==0){
			sdata[tid]+= sdata[tid+s];
			sdata[tid+1*1024]+=sdata[tid+1*1024+s];
			sdata[tid+2*1024]+=sdata[tid+2*1024+s];
			sdata[tid+3*1024]+=sdata[tid+3*1024+s];
		}
		__syncthreads();
	}
	if (tid ==0){
		unsigned int bid = blockIdx.x;
		omat[bid] = sdata [0];
		omat[bid+1*length/1024] = sdata[1*1024];
		omat[bid+2*length/1024] = sdata[2*1024];
		omat[bid+3*length/1024] = sdata[3*1024];
	}
}
