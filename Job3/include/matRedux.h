#include<iostream>
#include<fstream>
#include<sstream>
#include<iomanip>
#include<string>
#include<vector>
#include<ctime>
#include<cuda.h>
#include "cpuRedux.h"
#include "kernelRedux.h"
#include "stackTimer.h"
#ifndef __MAT_REDUX_H__
#define __MAT_REDUX_H__
#define OK "[  OK  ]"
#define FAIL "[  FAIL  ]"
#define DEBUG(A) { \
	cout<<tab<<A<<"Exiting. "<<endl; \
	exit(0); \
}
#define MSG(A) cout<<tab<<A<<"...";
#define ACK(A) cout<<A<<endl;
#define POST(A,B) cout<<tab<<A<<":   "<<B<<endl;
#define CDEBUG(A,E) { \
	if(E!=cudaSuccess){ \
		ACK(FAIL); \
		cout<<tab<<"Cuda Error: "<<A<<". Exiting."<<endl; \
		exit(0); \
	} \
	else{ \
		ACK(OK); \
	} \
}
#define CEVENTSET(START,STOP,CTIME) \
{ \
	cudaEventCreate(&START); \
	cudaEventCreate(&STOP); \
	cudaEventRecord(START,0); \
}
#define CEVENTGET(START,STOP,CTIME) \
{ \
	cudaEventRecord(stop, 0); \
	cudaEventSynchronize(stop); \
	cudaEventElapsedTime(&ctime, start, stop); \
	cudaEventDestroy(start); \
	cudaEventDestroy(stop); \
}
#define FILE_PATH "./data/mat.dat"
#define SYNC_LEN 2
#define LAG_THRESH 2000
#define MAT_SIZE 4
#define VOLUME 1.0
#define BLOCKX 1024
#define TAB0 ""
#define TAB1 "\t"
#define TAB2 "\t\t"
#define TAB3 "\t\t\t"

using namespace std;
string tab="";
bool sumTest(vector<float>&,string&,double*);
cudaError_t rowRedux(dim3&,dim3&,float*,float*,size_t,vector<float>&,int,double*,double*);
cudaError_t getRowResult(float*,size_t,vector<float>&,int,double*,double*);
void addMat(vector<float>&,vector<float>&,int);
#endif
