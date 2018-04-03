#include<iostream>
#include<string>
#include<cuda.h>
using namespace std;
int main(){
	struct cudaDeviceProp prop;
	cudaError_t err;
	err = cudaGetDeviceProperties(&prop,0);
	if(err!=cudaSuccess){
		cout<<"Get failed. Exiting."<<endl;
	}
	else{
		cout<<"Name   :   "<<string(prop.name)<<endl;
		cout<<"Total global memory   :   "<<prop.totalGlobalMem/(1024*1024*1024.0)<<" GB"<<endl;
        	cout<<"Shared memmory per block   :   "<<prop.sharedMemPerBlock/(1024.0)<<" KB"<<endl;
	        cout<<"32 bit registers per block   :   "<<prop.regsPerBlock<<endl;
	        cout<<"Warp size (in threads)   :   "<<prop.warpSize<<endl;
	        cout<<"Max pitch allowed by mem copy   :   "<<prop.memPitch/(1024*1024*1024.0)<<" GB"<<endl;
	        cout<<"Max threads per block   :   "<<prop.maxThreadsPerBlock<<endl;
	        cout<<"Max thread dimensions   :   "<<"("<<prop.maxThreadsDim[0]<<","<<prop.maxThreadsDim[1]<<","<<prop.maxThreadsDim[2]<<")"<<endl;
		cout<<"Max grid dimensions   :   "<<"("<<prop.maxGridSize[0]<<","<<prop.maxGridSize[1]<<","<<prop.maxGridSize[2]<<")"<<endl;
	        cout<<"Max const memory   :   "<<prop.totalConstMem/1024.0<<" KB"<<endl;
	        cout<<"Major compute capability   :   "<<prop.major<<endl;
	        cout<<"Minor compute capability   :   "<<prop.minor<<endl;
	        cout<<"Clock frequency   :   "<<prop.clockRate/1000.0<<" MHz"<<endl;
	        cout<<"Alignment requirement for textures   :   "<<prop.textureAlignment<<endl;
	        cout<<"Device can concurrently copy memory and execute a kernel   :   "<<(bool)prop.deviceOverlap<<endl;
	        cout<<"Number of multiprocessors on device   :   "<<prop.multiProcessorCount<<endl;
	        cout<<"Specified whether there is a run time limit on kernels   :   "<<(bool)prop.kernelExecTimeoutEnabled<<endl;
	        cout<<"Integrated   :   "<<(bool)prop.integrated<<endl;
	        cout<<"Can map host memory   :   "<<(bool)prop.canMapHostMemory<<endl;
	        cout<<"Compute Mode   :   "<<prop.computeMode<<endl;
	        cout<<"Concurrent kernels   :   "<<(bool)prop.concurrentKernels<<endl;
	        cout<<"ECC support   :   "<<(bool)prop.ECCEnabled<<endl;
	        cout<<"PCI bus id   :   "<<prop.pciBusID<<endl;
	        cout<<"PCI device id   :   "<<prop.pciDeviceID<<endl;
	        cout<<"TCC Driver   :   "<<(bool)prop.tccDriver<<endl;
	}
	return 0;
}
