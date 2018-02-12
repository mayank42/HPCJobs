__global__ void swap(int *A,int n){
	int idi = blockIdx.y*blockDim.y+threadIdx.y;
	int idj = blockIdx.x*blockDim.x+threadIdx.x;
	if(idi<n && idj<=idi){
		if(idj%2==0 && idj<n-1){
			int temp=A[idi*n+idj];
			A[idi*n+idj]=A[idi*n+idj+1];
			A[idi*n+idj+1]=temp;
		}
		int temp=idi;
		idi=idj;
		idj=temp;
		if(idi!=idj && idj%2==0 && idj<n-1){
			int temp=A[idi*n+idj];
			A[idi*n+idj]=A[idi*n+idj+1];
			A[idi*n+idj+1]=temp;
		}
		__syncthreads();
		temp=A[idi*n+idj];
		A[idi*n+idj]=A[idj*n+idi];
		A[idj*n+idi]=temp;
	}
}
