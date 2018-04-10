#include "cpuRedux.h"
int lowRedux(double *mat,unsigned int  length,double *ans,int flag){
	double a1,a2,a3,a4;
	a1=a2=a3=a4=0;
	if(!flag)
		for(unsigned int a=0;a<length;++a){
			a1+=mat[MAT_SIZE*a];
			a2+=mat[MAT_SIZE*a+1];
			a3+=mat[MAT_SIZE*a+2];
			a4+=mat[MAT_SIZE*a+3];
		}
	else
		for(unsigned int a=0;a<length;++a){
			a1+=mat[a];
			a2+=mat[a+length];
			a3+=mat[a+2*length];
			a4+=mat[a+3*length];
		}
	ans[0]=a1;
	ans[1]=a2;
	ans[2]=a3;
	ans[3]=a4;
	return 0;
}
