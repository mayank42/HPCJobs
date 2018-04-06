#include "cpuRedux.h"
int lowRedux(float *mat,unsigned int  length,float *ans,int flag){
	float a1,a2,a3,a4;
	a1=a2=a3=a4=0;
	for(unsigned int a=0;a<length;++a){
		a1+=mat[MAT_SIZE*a];
		a2+=mat[MAT_SIZE*a+1];
		a3+=mat[MAT_SIZE*a+2];
		a4+=mat[MAT_SIZE*a+3];
	}
	ans[0]=a1;
	ans[1]=a2;
	ans[2]=a3;
	ans[3]=a4;
	return 0;
}
