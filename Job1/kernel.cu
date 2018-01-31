//No headers
//ncvv automatically imports the required
__global__ void process_kernel1(float *input1,float *input2,float *output,int datasize){
	int blockNum = blockIdx.z * (gridDim.x *gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x*blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	int n = datasize/sizeof(input1[0]);
	if(id<n)output[id] = sin(input1[id]) + cos(input2[id]);
}

__global__ void process_kernel2(float *input,float *output,int datasize){
	int blockNum = blockIdx.z * (gridDim.x *gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x*blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	int n = datasize/sizeof(input[0]);
	if(id<n)output[id] = log(input[id]);
}

__global__ void process_kernel3(float *input,float *output,int datasize){
	int blockNum = blockIdx.z * (gridDim.x *gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x*blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	int n = datasize/sizeof(input[0]);
	if(id<n)output[id] = sqrt(input[id]);
}

