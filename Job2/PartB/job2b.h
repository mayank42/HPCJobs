#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>
#include<cuda.h> //general device functions and device structs
#include<cuda_runtime.h> //general purpose functions defined for device
#define oneDsize 10
#define twoDsize1 5
#define twoDsize2 5
#define INPUT_MAX 5
__global__ void conv1D(int*,int*,int*,int,int,int);
__global__ void conv2D(int*,float*,float*,int,int,int,int,int,int);
