#include<stdio.h> 
#include<malloc.h>
#include<cuda.h> //general device functions and device structs
#include<cuda_runtime.h> //general purpose functions defined for device
__global__ void process_kernel1(float*, float*, float*, int);
__global__ void process_kernel2(float*, float*, int);
__global__ void process_kernel3(float*, float*, int);

