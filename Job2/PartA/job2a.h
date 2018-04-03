#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>
#include<cuda.h> //general device functions and device structs
#include<cuda_runtime.h> //general purpose functions defined for device
#define size 3
__global__ void swap(int*,int);
#define debug(err,msg) if(err!=cudaSuccess){printf(msg);printf("\nError code: %d\nExiting.\n",err);fflush(stdout);return 0;}
