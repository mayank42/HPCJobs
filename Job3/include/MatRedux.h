#include<iostream>
#include<string>
#include<vector>
#include<cuda.h>
#define OK "[  OK  ]"
#define FAIL "[  FAIL  ]"
#define DEBUG(A) { \
	cout<<A<<"Exiting. "<<endl; \
	exit(0); \
}
#define MSG(A) cout<<A<<"...";
#define ACK(A) cout<<A<<endl;
#define POST(A,B) cout<<A<<":   "<<B<<endl;
#define CDEBUG(A,E) { \
	if(E!=cudaSuccess){ \
		cout<<"..."<<FAIL<<endl; \
		cout<<"Cuda Error: "<<A<<". Exiting."<<endl; \
		exit(0); \
	} \
	else{ \
		cout<<"..."<<OK<<endl; \
	} \
}
#define FILE_PATH "./data/mat.dat"
#define SYNC_LIM 1024
#define MAT_SIZE 4
using namespace std;
