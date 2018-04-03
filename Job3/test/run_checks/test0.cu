#include<stdio.h>
int main(){
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	printf("Device count = %d\n",nDevices);
	return 0;
}
