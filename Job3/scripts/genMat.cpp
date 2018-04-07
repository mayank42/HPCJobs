#include<iostream>
#include<fstream>
#include<cstdlib>
using namespace std;
#define FILE_PATH "./data/mat.dat"
#define LENGTH 1024*64*1024
#define FLAG 0
#define NUM 1.0
int main(int argc,char *argv[]){
	size_t length = LENGTH;
	ofstream file(FILE_PATH,ios::binary);
	file.write(reinterpret_cast<char*>(&length),sizeof(length));
	int flag = FLAG;
	file.write(reinterpret_cast<char*>(&flag),sizeof(flag));
	float num = NUM;
	float *arr = new float[4*64*1024];
	for(int a=0;a<4*64*1024;a++)arr[a]=1.0;
	for(size_t a=0;a<1024;a++){
		//file.write(reinterpret_cast<char*>(&num),sizeof(num));
		//file.write(reinterpret_cast<char*>(&num),sizeof(num));
		//file.write(reinterpret_cast<char*>(&num),sizeof(num));
		//file.write(reinterpret_cast<char*>(arr),512*1024*sizeof(float));
		//file.write(reinterpret_cast<char*>(arr),512*1024*sizeof(float));	
		//file.write(reinterpret_cast<char*>(arr),512*1024*sizeof(float));
		file.write(reinterpret_cast<char*>(arr),4*1024*64*sizeof(float));
	}
	file.close();
	delete[] arr;
	return 0;
}

