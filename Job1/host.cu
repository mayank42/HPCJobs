#include<job1.h>

void prepHostMem(void **,int,size_t);
void prepDeviceMem(void **,void **,int,int);
void freeDeviceMem(float **,int);
void kernel1Run(float **,float **);
void kernel2Run(float **,float **);
void kernel3Run(float **,float **);

/**
 * Host main Routine
 */
int main(void){
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	cudaError_t err;
	float **list_host,**list_device;
	
	/*************************************************************************************************
	* process_kernel1 segment 
	**************************************************************************************************/
	printf("Preparing for process_kernel1.\n");
	
	/*****************************************************************************/
	//Host memory preparation
	list_host = (float**)malloc(3*sizeof(float*));
	prepHostMem((void**)list_host,3,size);
	for(int a=0;a<3;a++)list_host[a]=(float*)list_host[a];
	// Initialize the host input vectors
    	for (int i = 0; i < numElements; i++){
        	list_host[0][i] = rand()/(float)RAND_MAX;
	        list_host[1][i] = rand()/(float)RAND_MAX;
    	}
	/*****************************************************************************/

	/*****************************************************************************/	
	//Device arrays
	list_device=(float**)malloc(3*sizeof(float*));
	prepDeviceMem((void**)list_host,(void**)list_device,3,2);
	/*****************************************************************************/
    	
	/*****************************************************************************/    	
	//process_kernel1	
	kernel1Run(list_host,list_device);
	//Freeing host and device list
	free(list_host);
	free(list_device);
	printf("Job process_kernel1 finished.\n");
	/*****************************************************************************/    	
	
	
	/*************************************************************************************************
	* process_kernel2 segment 
	**************************************************************************************************/
	printf("Preparing for process_kernel2.\n");
	
	/*****************************************************************************/
	//Host memory preparation
	list_host = (float**)malloc(2*sizeof(float*));
	prepHostMem((void**)list_host,2,size);
	for(int a=0;a<2;a++)list_host[a]=(float*)list_host[a];
	// Initialize the host input vectors
    	for (int i = 0; i < numElements; i++){
        	list_host[0][i] = rand()/(float)RAND_MAX;
    	}
	/*****************************************************************************/

	/*****************************************************************************/	
	//Device arrays
	list_device=(float**)malloc(2*sizeof(float*));
	prepDeviceMem((void**)list_host,(void**)list_device,2,1);
	/*****************************************************************************/
    	
	/*****************************************************************************/    	
	//process_kernel1	
	kernel2Run(list_host,list_device);
	//Freeing host and device list
	free(list_host);
	free(list_device);
	printf("Job process_kernel2 finished.\n");
	/*****************************************************************************/

	/*************************************************************************************************
	* process_kernel3 segment 
	**************************************************************************************************/
	printf("Preparing for process_kernel3.\n");
	
	/*****************************************************************************/
	//Host memory preparation
	list_host = (float**)malloc(2*sizeof(float*));
	prepHostMem((void**)list_host,2,size);
	for(int a=0;a<2;a++)list_host[a]=(float*)list_host[a];
	// Initialize the host input vectors
    	for (int i = 0; i < numElements; i++){
        	list_host[0][i] = rand()/(float)RAND_MAX;
    	}
	/*****************************************************************************/

	/*****************************************************************************/	
	//Device arrays
	list_device=(float**)malloc(2*sizeof(float*));
	prepDeviceMem((void**)list_host,(void**)list_device,2,1);
	/*****************************************************************************/
    	
	/*****************************************************************************/    	
	//process_kernel1	
	kernel3Run(list_host,list_device);
	//Freeing host and device list
	free(list_host);
	free(list_device);
	printf("Job process_kernel3 finished.\n");
	/*****************************************************************************/    	


	/*****************************************************************************/
    	// Reset the device and exit
    	// cudaDeviceReset causes the driver to clean up all state. While
    	// not mandatory in normal operation, it is good practice.  It is also
    	// needed to ensure correct operation when the application is being
    	// profiled. Calling cudaDeviceReset causes all profile data to be
    	// flushed before the application exits
    	err = cudaDeviceReset();

    	if (err != cudaSuccess){
        	fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
	/*****************************************************************************/
    	printf("All jobs finished.\n");
	fflush(stdout);
    	return 0;
}
/**
 *This function mallocs a list of host arrays each of size "size".
 *The list is supposed to be of length "l".
 */
void prepHostMem(void **list_host,int l,size_t size){
	printf("Allocating host memory...");
	for(int a=0;a<l;a++){
		list_host[a] = malloc(size);
		if (list_host[a]==NULL){
			fprintf(stderr,"Failed to allocate host vectors!\n");
			printf("Fail.\n");
			exit(EXIT_FAILURE);
		}
	}
	printf("OK.\n");
}
/**
 *This function mallocs a list of device arrays each corresponding to a host array.
 *The function not only cudaMallocs, but also copies memory from host array to device array.
 *Thus it takes list of host and device arrays. It cudaMallocs "l1" number of device arrays
 *and copies "l2" number of host arrays to device arrays.
 */
void prepDeviceMem(void **list_host,void **list_device,int l1,int l2){
	printf("Preparing device memory:\n");
	cudaError_t err = cudaSuccess;
	printf("Allcating device memory...");
	//Allocating device memory
	for(int a=0;a<l1;a++){
		err = cudaMalloc((void **)&list_device[a], sizeof(list_host[a]));
    		if (err != cudaSuccess){
        		fprintf(stderr, "Failed to allocate device vector %d (error code %s)!\n", a+1,cudaGetErrorString(err));
			printf("Fail.\n");
        		exit(EXIT_FAILURE);
    		}	
	}
	printf("OK.\n");
	//Copying from host to device
	printf("Copying to device memory...");
	//Allocating device memory
	for(int a=0;a<l2;a++){
		err = cudaMemcpy(list_device[a], list_host[a], sizeof(list_host[a]), cudaMemcpyHostToDevice);
    		if (err != cudaSuccess){
        		fprintf(stderr, "Failed to copy vector %d from host to device (error code %s)!\n", a+1, cudaGetErrorString(err));
			printf("Fail.\n");
        		exit(EXIT_FAILURE);
    		}
	}
	printf("OK.\n");
}
/**
 *This function frees a list of device arrays.
 *The list is supposed to be of length "l".
 */
void freeDeviceMem(float **list_device,int l){
	printf("Freeing device memory...");
	cudaError_t err;
	for(int a=0;a<l;a++){
		err = cudaFree(list_device[a]);
    		if (err != cudaSuccess){
        		fprintf(stderr, "Failed to free device vector %d (error code %s)!\n", a+1, cudaGetErrorString(err));
			printf("Fail.\n");
        		exit(EXIT_FAILURE);
    		}
	}
	printf("OK.\n");
}
/**
 *This function encapsulates the run of process_kernel1.
 *It takes the list of host and device arrays required to run.
 *This functions completes the GPU run and also does the testing part.
 */
void kernel1Run(float **list_host,float **list_device){
	printf("CUDA kernel launch with (4,2,2) blocks of (32,32,1) threads.\n");
	cudaError_t err;
	int l=3;
	int size = sizeof(list_device[0]);
	int numElements = size/sizeof(float);
	dim3 X(4,2,2);
	dim3 Y(32,32,1);
	process_kernel1<<<X, Y>>>(list_device[0], list_device[1], list_device[2], size);
    	err = cudaGetLastError();
    	if (err != cudaSuccess){
        	fprintf(stderr, "Failed to launch process1 kernel (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
	printf("Kernel successfully returned from device.\n");

    	// Copy the device result vector in device memory to the host result vector
    	// in host memory.
    	printf("Copying output data from the CUDA device to the host memory...");
    	err = cudaMemcpy((void*)list_host[2], (void*)list_device[2], size, cudaMemcpyDeviceToHost);
    	if (err != cudaSuccess){
        	fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		printf("Fail.\n");
        	exit(EXIT_FAILURE);
    	}
	printf("OK.\n");
    	// Verify that the result vector is correct
	printf("Verifying results...");
    	for (int i = 0; i < numElements; ++i){
        	if (fabs(sin(list_host[0][i]) + cos(list_host[1][i]) - list_host[2][i]) > 1e-5){
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			printf("Fail.\n");
        	    	exit(EXIT_FAILURE);
        	}
    	}
    	printf("OK [Test Passed].\n");

	//Free device global memory
	freeDeviceMem(list_device,l);

    	// Free host memory
	printf("Freeing host memory...");
    	for(int a=0;a<l;a++)free(list_host[a]);
	printf("OK.\n");
}
/**
 *This function encapsulates the run of process_kernel2.
 *It takes the list of host and device arrays required to run.
 *This functions completes the GPU run and also does the testing part.
 */
void kernel2Run(float **list_host,float **list_device){
	int size = sizeof(list_device[0]);
	int numElements = size/sizeof(float);
	int blockz = 16;
	int gridy = (int)ceil((float)numElements/(8*8*blockz));
	printf("CUDA kernel launch with (2,%d,1) blocks of (8,8,%d) threads.\n",gridy,blockz);
	cudaError_t err;
	int l=2;
	dim3 X(2,gridy,1);
	dim3 Y(8,8,blockz);
	process_kernel2<<<X, Y>>>(list_device[0], list_device[1], size);
    	err = cudaGetLastError();
    	if (err != cudaSuccess){
        	fprintf(stderr, "Failed to launch process1 kernel (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
	printf("Kernel successfully returned from device.\n");

    	// Copy the device result vector in device memory to the host result vector
    	// in host memory.
    	printf("Copying output data from the CUDA device to the host memory...");
    	err = cudaMemcpy((void*)list_host[1], (void*)list_device[1], size, cudaMemcpyDeviceToHost);
    	if (err != cudaSuccess){
        	fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		printf("Fail.\n");
        	exit(EXIT_FAILURE);
    	}
	printf("OK.\n");
    	// Verify that the result vector is correct
	printf("Verifying results...");
    	for (int i = 0; i < numElements; ++i){
        	if (fabs(log(list_host[0][i]) - list_host[1][i]) > 1e-5){
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			printf("Fail.\n");
        	    	exit(EXIT_FAILURE);
        	}
    	}
    	printf("OK [Test Passed].\n");

	//Free device global memory
	freeDeviceMem(list_device,l);

    	// Free host memory
	printf("Freeing host memory...");
    	for(int a=0;a<l;a++)free(list_host[a]);
	printf("OK.\n");
}
/**
 *This function encapsulates the run of process_kernel3.
 *It takes the list of host and device arrays required to run.
 *This functions completes the GPU run and also does the testing part.
 */
void kernel3Run(float **list_host,float **list_device){
	int size = sizeof(list_device[0]);
	int numElements = size/sizeof(float);
	int blocky = 4;
	int gridx = (int)ceil((float)numElements/(128*blocky));
	printf("CUDA kernel launch with (%d,1,1) blocks of (128,%d,1) threads.\n",gridx,blocky);
	cudaError_t err;
	int l=2;
	dim3 X(gridx,1,1);
	dim3 Y(128,blocky,1);
	process_kernel3<<<X, Y>>>(list_device[0], list_device[1], size);
    	err = cudaGetLastError();
    	if (err != cudaSuccess){
        	fprintf(stderr, "Failed to launch process1 kernel (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
	printf("Kernel successfully returned from device.\n");

    	// Copy the device result vector in device memory to the host result vector
    	// in host memory.
    	printf("Copying output data from the CUDA device to the host memory...");
    	err = cudaMemcpy((void*)list_host[1], (void*)list_device[1], size, cudaMemcpyDeviceToHost);
    	if (err != cudaSuccess){
        	fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		printf("Fail.\n");
        	exit(EXIT_FAILURE);
    	}
	printf("OK.\n");
    	// Verify that the result vector is correct
	printf("Verifying results...");
    	for (int i = 0; i < numElements; ++i){
        	if (fabs(sqrt(list_host[0][i]) - list_host[1][i]) > 1e-5){
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			printf("Fail.\n");
        	    	exit(EXIT_FAILURE);
        	}
    	}
    	printf("OK [Test Passed].\n");

	//Free device global memory
	freeDeviceMem(list_device,l);

    	// Free host memory
	printf("Freeing host memory...");
    	for(int a=0;a<l;a++)free(list_host[a]);
	printf("OK.\n");
}
