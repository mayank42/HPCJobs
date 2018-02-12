__global__ void conv1D(int *arr,int *mask,int *res,int n,int m,int c){
	int idi = blockIdx.y*blockDim.y+threadIdx.y;
	if(idi<n){
		res[idi]=0;
		int a,b;
		b=idi-c;
		for(a=0;a<m;a++,b++){
			if(b>=0 && b<n){
				res[idi]+=mask[a]*arr[b];
			}
		}
	}
}
__global__ void conv2D(int *arr,float *mask,float *res,int n1,int n2, int m1,int m2,int c1,int c2){
	int idi = blockIdx.y*blockDim.y+threadIdx.y;
	int idj = blockIdx.x*blockDim.x+threadIdx.x;
	if(idi<n1 && idj<n2){
		int a,b,c,d;
		c=idi-c1;
		res[idi*n1+idj]=0.0;
		for(a=0;a<m1;a++,c++){
			d=idj-c2;
			for(b=0;b<m2;b++,d++){
				if(c>=0 && c<n1 && d>=0 && d<n2){
					res[idi*n1+idj]+=mask[a*m1+b]*arr[c*n1+d];
				}
			}
		}
	}
}
