#include "matRedux.h"
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
	vector<double> bans;
	cudaError_t err;
	float ctime;
	double diff;
	cudaEvent_t start,stop;
	double cudaRedux,cudaMem,serialTotal;
	cudaRedux=cudaMem=serialTotal=0.0;
	while(bstart<length){
		POST("Processing batch count",count);		
		tab=TAB1;

		/*******************HOST MEMORY ALLOC*******************/
		MSG("Allocating memory on host");
		float *h_mat_f;
		double *h_mat;
		try{
			h_mat_f = new float[MAT_SIZE*bsize];
			h_mat = new double[MAT_SIZE*bsize];
			
		}catch(exception &e){
			ACK(FAIL);
			DEBUG(string("Allocation error. ")+e.what());
		};
		ACK(OK);
		/*******************HOST MEMORY ALLOC*******************/


		/*******************DATA FILE READ*******************/
		MSG("Reading data from file");
		try{
			if(bstart+bsize<=length)
				data.read(reinterpret_cast<char*>(h_mat_f),MAT_SIZE*bsize*sizeof(float));
			else{
				data.read(reinterpret_cast<char*>(h_mat_f),MAT_SIZE*(length-bstart)*sizeof(float));
				for(size_t loopvar = (length-bstart)*MAT_SIZE;loopvar<MAT_SIZE*bsize;++loopvar)h_mat_f[loopvar]=0.0f;
			}
		}catch(exception &e){
			ACK(FAIL);
			DEBUG(string("Data read failed. ")+e.what());
		}
		ACK(OK);
		if(!flag)
			for(size_t a=0;a<MAT_SIZE*bsize;a++)h_mat[a]=h_mat_f[a];
		else
			for(size_t a=0;a<bsize;++a){
				for(int b=0;b<MAT_SIZE;++b){
					h_mat[b*MAT_SIE+a] = h_mat_f[a*4+b];
				}
			}
		delete[] h_mat_f;
		/*******************DATA FILE READ*******************/

		
		/*******************DEVICE MEMORY ALLOC*******************/		
		double *d_imat;
		double *d_omat;
		MSG("Allocating memory on device");
		err = cudaMalloc(&d_imat,MAT_SIZE*bsize*sizeof(double));
		CDEBUG("Device memory allocation",err);
		MSG("Allocating memory on device");
		err = cudaMalloc(&d_omat,MAT_SIZE*(bsize/1024+1024)*sizeof(double));
		CDEBUG("Device memory allocation",err);
		/*******************DEVICE MEMORY ALLOC*******************/

		
		/*******************DEVICE MEM COPY*******************/
		MSG("Copying from host to device");
		CEVENTSET(start,stop,ctime);
		err = cudaMemcpy(d_imat,h_mat,MAT_SIZE*bsize*sizeof(double),cudaMemcpyHostToDevice);
		CEVENTGET(start,stop,ctime);
		cudaMem+=ctime;
		CDEBUG("Host to device memcpy",err);
		/*******************DEVICE MEM COPY*******************/		

		
		/*******************REDUCTION*******************/
		POST("Starting reduction","");
		tab=TAB2;
		if(flag){
			err = colRedux(grid,block,d_imat,d_omat,bsize,bans,flag,&cudaRedux,&cudaMem);
			err = cudaGetLastError();
			CDEBUG("Column kernel reduction",err);
		}
		else{
			err = rowRedux(grid,block,d_imat,d_omat,bsize,bans,flag,&cudaRedux,&cudaMem);
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
		tab = TAB0;
	}
	tab=TAB0;	

	MSG("Adding results on host");
	vector<double> ans(4,0.0);
	push_clock();
	addMat(bans,ans,flag);
	diff = push_clock();
	cudaRedux+=diff;
	ACK("[  Done  ]");
	clear_clock();
	
	
	data.close();	
	/*******************TESTING*******************/
	MSG("Testing");
	string comment;
	if(sumTest(ans,comment,&serialTotal)){
		ACK(OK);
		cout<<setprecision(8);
		stringstream cudaDisp;
		cudaDisp<<setprecision(8)<<cudaRedux<<" ( + "<<cudaMem<<" memcpy )";
		POST("Cuda time (ms)",cudaDisp.str());
		POST("Serial time (ms)",serialTotal*1000/CLOCKS_PER_SEC);
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

bool sumTest(vector<double> &ans,string &comm,double *tot){
	ifstream data(FILE_PATH,ios::binary);
	if(!data.is_open()){
		comm = "Unable to open data file.";
		return false;
	}
	size_t length;
	int flag;
	double ctime;
	data.read(reinterpret_cast<char*>(&length),sizeof(length));
	data.read(reinterpret_cast<char*>(&flag),sizeof(flag));
	vector<double> res(4,0.0);
	double *acc = new double[4];
	size_t readBuf = VOLUME*length;
	size_t readCount = 0;
	size_t reads;
	if(!flag){
		float *mat_f = new float[MAT_SIZE*readBuf];
		double *mat = new double[MAT_SIZE*readBuf];
		while(readCount<length){
			if(readCount+readBuf<=length)
				reads = readBuf;
			else
				reads = length-readCount;
			data.read(reinterpret_cast<char*>(mat_f),reads*MAT_SIZE*sizeof(float));
			for(size_t a=0;a<MAT_SIZE*reads;++a)mat[a] = mat_f[a];
			push_clock();	
			lowRedux(mat,reads,acc,flag);		
			res[0]+=acc[0];
			res[1]+=acc[1];
			res[2]+=acc[2];
			res[3]+=acc[3];
			ctime = push_clock();
			*tot = *tot + ctime;
			readCount+=reads;
		}
		delete[] mat_f;
		delete[] mat;
	}
	clear_clock();
	data.close();
	delete[] acc;
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

cudaError_t rowRedux(dim3 grid,dim3 block,double *d_imat,double *d_omat,size_t length,vector<double> &bans,int flag,double *rtot,double *mtot){
	double *arr[]={d_imat,d_omat};
	int pos=0;
	cudaError_t err;
	POST("Reducing on length",length);
	POST("Sync length",SYNC_LEN);
	float ctime;
	double diff;
	cudaEvent_t start,stop;
	unsigned int lag;
	int a=0;
	while(length>SYNC_LEN){
		POST("Iteration number",a+1);
		CEVENTSET(start,stop,ctime);
		row_kernel<<<grid,block>>>(arr[pos%2],arr[(pos+1)%2]);
		CEVENTGET(start,stop,ctime);
		*rtot=*rtot+(double)ctime;
		err = cudaGetLastError();
		if(err!=cudaSuccess){
			cout<<err<<endl;
			cout<<length<<","<<grid.x<<endl;
			cout<<a+1<<endl;
			return err;
		}
		pos++;
		length = grid.x;
		lag = (1024-length%1024)%1024;
		if(length<=SYNC_LEN || (lag>LAG_THRESH  && (length-length%1024)<=SYNC_LEN))break;
		else if(length%1024!=0){
			if(lag<LAG_THRESH){
				length = length+lag;
				MSG("Length lag < threshold. Adjusting with memset");
				CEVENTSET(start,stop,ctime);
				err = cudaMemset(arr[pos%2]+MAT_SIZE*grid.x,0,MAT_SIZE*sizeof(double)*(lag));
				CEVENTGET(start,stop,ctime);
				*rtot=*rtot+(double)ctime;
				CDEBUG("Memset during length lag adjustment",err);
				grid.x = length/1024;
			}
			else{
				lag = length%1024;
				length-=lag;
				double *temp = new double[(lag+1)*MAT_SIZE];
				double *temp_ans = new double[MAT_SIZE];
				MSG("Length lag>=threshold. Adjusting with collapsing");
				CEVENTSET(start,stop,ctime);
				err = cudaMemcpy(temp,arr[pos%2]+MAT_SIZE*(length-1),MAT_SIZE*sizeof(double)*(lag+1),cudaMemcpyDeviceToHost);
				CEVENTGET(start,stop,ctime);
				*mtot=*mtot+(double)ctime;
				CDEBUG("Memcpy during length lag collapse",err);
				push_clock();
				lowRedux(temp,lag+1,temp_ans,flag);
				diff = push_clock();
				*rtot = *rtot + diff;
				MSG("Copying back after collapse");
				CEVENTSET(start,stop,ctime);
				err = cudaMemcpy(arr[pos%2]+MAT_SIZE*(length-1),temp_ans,MAT_SIZE*sizeof(double),cudaMemcpyHostToDevice);
				CEVENTGET(start,stop,ctime);
				*mtot=*mtot+(double)ctime;
				CDEBUG("Memcpy after collapse",err);
				delete[] temp;
				delete[] temp_ans;
				grid.x = length/1024;
			}
		}			
		else grid.x/=1024;
		++a;
	}
	clear_clock();
	return getRowResult(arr[pos%2],length,bans,flag,rtot,mtot);
}
cudaError_t getRowResult(double *d_omat,size_t length,vector<double> &bans,int flag,double *rtot,double *mtot){
	double *h_mat = new double[MAT_SIZE*length];
	cudaError_t err;
	double diff;
	cudaEvent_t start,stop;
	float ctime=0;
	CEVENTSET(start,stop,ctime);
	err = cudaMemcpy(h_mat,d_omat,MAT_SIZE*length*sizeof(double),cudaMemcpyDeviceToHost);
	CEVENTGET(start,stop,ctime);
	*mtot=*mtot+(double)ctime;
	if(err!=cudaSuccess)return err;
	double *h_ans = new double[MAT_SIZE];
	push_clock();
	lowRedux(h_mat,length,h_ans,flag);
	diff = push_clock();
	*rtot = *rtot + diff;
	bans.push_back(h_ans[0]);
	bans.push_back(h_ans[1]);
	bans.push_back(h_ans[2]);
	bans.push_back(h_ans[3]);
	delete[] h_mat;
	delete[] h_ans;
	clear_clock();
	return cudaSuccess;
}
cudaError_t colRedux(dim3 grid,dim3 block,double *d_imat,double *d_omat,size_t length,vector<double> &bans,int flag,double *rtot,double *mtot){
	double *arr[]={d_imat,d_omat};
	int pos=0;
	cudaError_t err;
	POST("Reducing on length",length);
	POST("Sync length",SYNC_LEN);
	float ctime;
	double diff;
	cudaEvent_t start,stop;
	unsigned int lag;
	int a=0;
	while(length>SYNC_LEN){
		POST("Iteration number",a+1);
		CEVENTSET(start,stop,ctime);
		col_kernel<<<grid,block>>>(arr[pos%2],arr[(pos+1)%2],length);
		CEVENTGET(start,stop,ctime);
		*rtot=*rtot+(double)ctime;
		err = cudaGetLastError();
		if(err!=cudaSuccess){
			cout<<err<<endl;
			cout<<length<<","<<grid.x<<endl;
			cout<<a+1<<endl;
			return err;
		}
		pos++;
		length = grid.x;
		if(length<=SYNC_LEN ||  (length-length%1024)<=SYNC_LEN)break;
		else if(length%1024!=0){
			/**lag = length%1024;
			length-=lag;
			double *temp = new double[(lag+1)*MAT_SIZE];
			double *temp_ans = new double[MAT_SIZE];
			MSG("Length lag>=threshold. Adjusting with collapsing");
			CEVENTSET(start,stop,ctime);
			err = cudaMemcpy(temp,arr[pos%2]+MAT_SIZE*(length-1),MAT_SIZE*sizeof(double)*(lag+1),cudaMemcpyDeviceToHost);
			CEVENTGET(start,stop,ctime);
			*mtot=*mtot+(double)ctime;
			CDEBUG("Memcpy during length lag collapse",err);
			push_clock();
			lowRedux(temp,lag+1,temp_ans,flag);
			diff = push_clock();
			*rtot = *rtot + diff;
			MSG("Copying back after collapse");
			CEVENTSET(start,stop,ctime);
			err = cudaMemcpy(arr[pos%2]+MAT_SIZE*(length-1),temp_ans,MAT_SIZE*sizeof(double),cudaMemcpyHostToDevice);
			CEVENTGET(start,stop,ctime);
			*mtot=*mtot+(double)ctime;
			CDEBUG("Memcpy after collapse",err);
			delete[] temp;
			delete[] temp_ans;
			grid.x = length/1024;*/
		}		
		else grid.x/=1024;
		++a;
	}
	clear_clock();
	return getColResult(arr[pos%2],length,bans,flag,rtot,mtot);
}
cudaError_t getColResult(double *d_omat,size_t length,vector<double> &bans,int flag,double *rtot,double *mtot){
	double *h_mat = new double[MAT_SIZE*length];
	cudaError_t err;
	double diff;
	cudaEvent_t start,stop;
	float ctime=0;
	CEVENTSET(start,stop,ctime);
	err = cudaMemcpy(h_mat,d_omat,MAT_SIZE*length*sizeof(double),cudaMemcpyDeviceToHost);
	CEVENTGET(start,stop,ctime);
	*mtot=*mtot+(double)ctime;
	if(err!=cudaSuccess)return err;
	double *h_ans = new double[MAT_SIZE];
	push_clock();
	lowRedux(h_mat,length,h_ans,flag);
	diff = push_clock();
	*rtot = *rtot + diff;
	bans.push_back(h_ans[0]);
	bans.push_back(h_ans[1]);
	bans.push_back(h_ans[2]);
	bans.push_back(h_ans[3]);
	delete[] h_mat;
	delete[] h_ans;
	clear_clock();
	return cudaSuccess;
}
void addMat(vector<double> &bans,vector<double> &ans,int flag){
	int l = bans.size()/4;
	ans[0]=ans[1]=ans[2]=ans[3]=0.0;
	if(!flag){
		for(int a=0;a<l;a++){
			ans[0]+=bans[a*4];
			ans[1]+=bans[a*4+1];
			ans[2]+=bans[a*4+2];
			ans[3]+=bans[a*4+3];
		}
	}
}
