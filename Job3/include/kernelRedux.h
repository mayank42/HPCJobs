#ifndef __KERNEL_REDUX_H__
#define __KERNEL_REDUX_H__
__global__ void row_kernel(double*,double*,size_t);
__global__ void col_kernel(double*,double*,size_t);
#define GRID_FACTOR 2
#define REDOP(SDATA,TID,S) \
{ \
	SDATA[TID] += SDATA[TID+S]; \
	SDATA[TID+1] += SDATA[TID+S+1]; \
	SDATA[TID+2] += SDATA[TID+S+2]; \
	SDATA[TID+3] += SDATA[TID+S+3]; \
}
#define IFREDOP(SDATA,TID,S) \
{ \
	if(TID<S){ \
		REDOP(SDATA,TID,S); \
		__syncthreads(); \
	} \
}
#define REDOPCOL(SDATA,TID,S) \
{ \
	SDATA[TID] += SDATA[TID+S]; \
	SDATA[TID+1*1024] += SDATA[TID+S+1*1024]; \
	SDATA[TID+2*1024] += SDATA[TID+S+2*1024]; \
	SDATA[TID+3*1024] += SDATA[TID+S+3*1024]; \
}
#define IFREDOPCOL(SDATA,TID,S) \
{ \
	if(TID<S){ \
		REDOPCOL(SDATA,TID,S); \
		__syncthreads(); \
	} \
}
#endif
#ifndef MAT_SIZE
#define MAT_SIZE 4
#endif
