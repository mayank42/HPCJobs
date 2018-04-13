#include "kernelRedux.h"
__global__ void  row_kernel(double *imat , double *omat,size_t grids){
	unsigned  int tid = 4*threadIdx.x;
	unsigned  int i = 4*(blockIdx.x*blockDim.x*2 + threadIdx.x);
	if(blockIdx.x*2+1==grids)return;
	__shared__ double  sdata [1024*MAT_SIZE];
	if(blockIdx.x*2+3==grids){
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
	IFREDOP(sdata,tid,4*512);
	IFREDOP(sdata,tid,4*256);
	IFREDOP(sdata,tid,4*128);
	IFREDOP(sdata,tid,4*64);
	if(tid<4*32){
		REDOP(sdata,tid,4*32);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOP(sdata,tid,4*16);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOP(sdata,tid,4*8);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOP(sdata,tid,4*4);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOP(sdata,tid,4*2);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOP(sdata,tid,4*1);
		if (tid ==0){
			unsigned int bid = 4*blockIdx.x;
			omat[bid] = sdata [0];
			omat[bid+1] = sdata[1];
			omat[bid+2] = sdata[2];
			omat[bid+3] = sdata[3];
		}
	}
}
__global__ void  col_kernel(double *imat , double *omat,size_t length){
	__shared__ double  sdata [1024*MAT_SIZE];
	unsigned  int tid = threadIdx.x;
	unsigned  int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	if(blockIdx.x*2+1>=gridDim.x)return;
	else if(blockIdx.x*2+3==gridDim.x){
		sdata[tid+0*1024] = imat[i+0*length] + imat[i+blockDim.x+0*length] + imat[i+2*blockDim.x+0*length];
		sdata[tid+1*1024] = imat[i+1*length] + imat[i+blockDim.x+1*length] + imat[i+2*blockDim.x+1*length];
		sdata[tid+2*1024] = imat[i+2*length] + imat[i+blockDim.x+2*length] + imat[i+2*blockDim.x+2*length];
		sdata[tid+3*1024] = imat[i+3*length] + imat[i+blockDim.x+3*length] + imat[i+2*blockDim.x+3*length];

	}
	else{	
		sdata[tid+0*1024] = imat[i+0*length] + imat[i+blockDim.x+0*length];
		sdata[tid+1*1024] = imat[i+1*length] + imat[i+blockDim.x+1*length];
		sdata[tid+2*1024] = imat[i+2*length] + imat[i+blockDim.x+2*length];
		sdata[tid+3*1024] = imat[i+3*length] + imat[i+blockDim.x+3*length];
	}
	__syncthreads();
	IFREDOPCOL(sdata,tid,512);
	IFREDOPCOL(sdata,tid,256);
	IFREDOPCOL(sdata,tid,128);
	IFREDOPCOL(sdata,tid,64);
	if(tid<4*32){
		REDOPCOL(sdata,tid,32);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOPCOL(sdata,tid,16);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOPCOL(sdata,tid,8);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOPCOL(sdata,tid,4);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOPCOL(sdata,tid,2);
		//__syncthreads();
		//__threadfence();
		__threadfence_block();
		//__syncwarp(0xA);
		REDOPCOL(sdata,tid,1);
		if (tid ==0){
			unsigned int bid = blockIdx.x;
			omat[bid] = sdata [0];
			omat[bid+1*length/1024] = sdata[1*1024];
			omat[bid+2*length/1024] = sdata[2*1024];
			omat[bid+3*length/1024] = sdata[3*1024];
		}
	}
}
