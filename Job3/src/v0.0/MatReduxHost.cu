#include "MatRedux.h"
int main(int argc,char *argv[]){
	tab=TAB0;
	MSG("Checking data file");
	ifstream data(FILE_PATH,ios::binary);
	if(data.is_open()){
		ACK(OK);
	}
	else{
		ACK(FAIL);
		DEBUG("Unable to open file.");
	}
	
	size_t length;
	data.read(reinterpret_cast<char*>(&length),sizeof(length));
	POST("Number of 2x2 matrices",length);
	int flag;
	data.read(reinterpret_cast<char*>(&flag),sizeof(flag));
	if(flag){
		POST("Alignment","Column Major");
	}
	else{
		 POST("Alignment","Row Major");
	}

	POST("Batch volume percentage",VOLUME);
	size_t bsize = (size_t)(length*VOLUME);
	if(bsize%1024!=0){
		POST("Batch size unaligned. Rounding to nearest multiple of 1024","");
		bsize+=1024-bsize%1024;
	}
	POST("Batch size",bsize);
	POST("Block dimension",BLOCKX);
	MSG("Calibrating grid dimension");
	unsigned long long gridx = (unsigned long long)(bsize/1024);
	if(gridx>2147483647){
		ACK(FAIL);
		DEBUG("Batch volume too large. Reduce it");
	}
	ACK(OK);
	POST("Grid dimension",gridx);
	dim3 grid(gridx,1,1);
	dim3 block(BLOCKX,1,1);

	
	size_t bstart = 0;
	int count = 1;
	vector<float> bans;
	cudaError_t err;
	clock_t begin,end;
	double cudaTotal,serialTotal;
	cudaTotal=serialTotal=0.0;
	while(bstart<length){
		POST("Processing batch count",count);		
		tab=TAB1;

		/*******************HOST MEMORY ALLOC*******************/
		MSG("Allocating memory on host");
		float *h_mat;
		try{
			h_mat = new float[MAT_SIZE*bsize];
		}catch(exception &e){
			ACK(FAIL);
			DEBUG(string("Allocation error. ")+e.what());
		};
		ACK(OK);
		/*******************HOST MEMORY ALLOC*******************/


		/*******************DATA FILE READ*******************/
		MSG("Reading data from file");
		try{
			if(bstart+bsize<length)
				data.read(reinterpret_cast<char*>(h_mat),MAT_SIZE*bsize*sizeof(float));
			else{
				data.read(reinterpret_cast<char*>(h_mat),MAT_SIZE*(length-bstart)*sizeof(float));
				for(size_t loopvar = (length-bstart)*MAT_SIZE;loopvar<MAT_SIZE*bsize;++loopvar)h_mat[loopvar]=0.0;
			}
		}catch(exception &e){
			ACK(FAIL);
			DEBUG(string("Data read failed. ")+e.what());
		}
		ACK(OK);
		/*******************DATA FILE READ*******************/

		
		/*******************DEVICE MEMORY ALLOC*******************/		
		float *d_imat;
		float *d_omat;
		MSG("Allocating memory on device");
		err = cudaMalloc(&d_imat,MAT_SIZE*bsize*sizeof(float));
		CDEBUG("Device memory allocation",err);
		MSG("Allocating memory on device");
		err = cudaMalloc(&d_omat,MAT_SIZE*bsize/1024*sizeof(float));
		CDEBUG("Device memory allocation",err);
		/*******************DEVICE MEMORY ALLOC*******************/

		
		/*******************DEVICE MEM COPY*******************/
		MSG("Copying from host to device");
		err = cudaMemcpy(d_imat,h_mat,MAT_SIZE*bsize*sizeof(float),cudaMemcpyHostToDevice);
		CDEBUG("Host to device memcpy",err);
		/*******************DEVICE MEM COPY*******************/		

		
		/*******************REDUCTION*******************/
		POST("Starting reduction","");
		tab=TAB2;
		if(flag){
				//colRedux(grid,block,d_imat,d_omat,h_mat,bsize,bans);
			err = cudaGetLastError();
			CDEBUG("Column kernel reduction",err);
		}
		else{
			err = rowRedux(grid,block,d_imat,d_omat,bsize,bans,&cudaTotal);
			MSG("Reduction status");
			CDEBUG("Row kernel reduction",err);
		}
		tab=TAB1;
		/*******************REDUCTION*******************/
		

		/*******************MEMORY FREE*******************/
		MSG("Freeing host memory");
		delete[] h_mat;
		ACK(OK);
		MSG("Freeing device input memory");
		err = cudaFree(d_imat);
		CDEBUG("Free device memory",err);
		MSG("Freeing device output memory");
		err = cudaFree(d_omat);
		CDEBUG("Free device memory",err);
		/*******************MEMORY FREE*******************/

		
		bstart+=bsize;
		count++;
	}
	tab=TAB0;	

	MSG("Adding results on host");
	vector<float> ans(4,0.0);
	begin = clock();
	addMat(bans,ans,flag);
	end = clock();
	cout<<begin<<" "<<end<<endl;
	cudaTotal+=(double)(end-begin);
	ACK("[  Done  ]");
	
	
	data.close();	
	/*******************TESTING*******************/
	MSG("Testing");
	string comment;
	if(sumTest(ans,comment,&serialTotal)){
		ACK(OK);
		cout<<setprecision(8);
		POST("Cuda time",cudaTotal);
		POST("Serial time",serialTotal*1000/CLOCKS_PER_SEC);
	}
	else{
		ACK(FAIL);
		POST("Fail message",comment);
	}
	/*******************TESTING*******************/
	
	MSG("Resetting device");
	err = cudaDeviceReset();
	CDEBUG("Device reset",err);
	return 0;
}

bool sumTest(vector<float> &ans,string &comm,double *tot){
	ifstream data(FILE_PATH,ios::binary);
	if(!data.is_open()){
		comm = "Unable to open data file.";
		return false;
	}
	size_t length;
	int flag;
	data.read(reinterpret_cast<char*>(&length),sizeof(length));
	data.read(reinterpret_cast<char*>(&flag),sizeof(flag));
	vector<float> res(4,0.0);
	if(!flag){
		float *mat = new float[1024*MAT_SIZE];
		int reads = 1024;
		clock_t end,begin;
		begin = clock();
		for(size_t start=0;start<length;start+=1024){
			if(start+1024>length)reads = length-start;
			data.read(reinterpret_cast<char*>(mat),reads*MAT_SIZE*sizeof(float));
			begin=clock();
			for(int a=0;a<reads;a++){
				res[0]+=mat[a*4];
				res[1]+=mat[a*4+1];
				res[2]+=mat[a*4+2];
				res[3]+=mat[a*4+3];
			}
			end=clock();
			*tot = *tot + (double)(end-begin);
		}
	}
	if(ans[0]==res[0] && ans[1]==res[1] && ans[2]==res[2] && ans[3]==res[3]){
		comm="Test successful.";
		return true;
	}
	else{
		stringstream com;
		com<<"Test unsuccessful.\nSerial result:\n";
		com<<setprecision(8);
		com<<res[0]<<" "<<res[1]<<"\n"<<res[2]<<" "<<res[3]<<"\nCuda result:\n";
		com<<ans[0]<<" "<<ans[1]<<"\n"<<ans[2]<<" "<<ans[3];
		comm=com.str();

		return false;
	}
}

cudaError_t rowRedux(dim3 &grid,dim3 &block,float *d_imat,float *d_omat,size_t length,vector<float> &bans,double *tot){
	float *arr[]={d_imat,d_omat};
	int pos=0;
	int sync_len=0;
	cudaError_t err;
	POST("Reducing on length",length);
	while(length>SYNC_LIM){
		length/=1024;
		sync_len++;
	}
	POST("Sync length",sync_len);
	float ctime;
	for(int a=0;a<sync_len;a++){
		POST("Iteration number",a+1);
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		row_kernel<<<grid,block>>>(arr[pos%2],arr[(pos+1)%2]);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ctime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cout<<"Here"<<ctime<<endl;
		*tot=*tot+(double)ctime;
		err = cudaGetLastError();
		if(err!=cudaSuccess)return err;
		pos++;
	}
	return getRowResult(arr[pos%2],length,bans);
}
cudaError_t getRowResult(float *d_omat,size_t length,vector<float> &bans){
	float *h_mat = new float[MAT_SIZE*length];
	cudaError_t err;
	err = cudaMemcpy(h_mat,d_omat,MAT_SIZE*length*sizeof(float),cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess)return err;
	float a,b,c,d;
	a=b=c=d=0.0;
	for(int i=0;i<length;++i){
		a+=h_mat[MAT_SIZE*i];
		b+=h_mat[MAT_SIZE*i+1];
		c+=h_mat[MAT_SIZE*i+2];
		d+=h_mat[MAT_SIZE*i+3];
	}
	bans.push_back(a);
	bans.push_back(b);
	bans.push_back(c);
	bans.push_back(d);
	return cudaSuccess;
}
void addMat(vector<float> &bans,vector<float> &ans,int flag){
	int l = bans.size()/4;
	ans[0]=ans[1]=ans[2]=ans[3]=0.0;
	for(int a=0;a<l;a++){
		ans[0]+=bans[a*4];
		ans[1]+=bans[a*4+1];
		ans[2]+=bans[a*4+2];
		ans[3]+=bans[a*4+3];
	}
}
