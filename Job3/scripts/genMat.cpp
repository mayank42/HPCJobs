#include<iostream>
#include<fstream>
using namespace std;
#define FILE_PATH "./data/mat.dat"
#define LENGTH 1024*1024
#define FLAG 0
#define NUM 1.0
int main(){
	ofstream file(FILE_PATH,ios::binary);
	size_t length = LENGTH;
	file.write(reinterpret_cast<char*>(&length),sizeof(length));
	int flag = FLAG;
	file.write(reinterpret_cast<char*>(&flag),sizeof(flag));
	float num = NUM;
	for(size_t a=0;a<length;a++){
		file.write(reinterpret_cast<char*>(&num),sizeof(num));
		file.write(reinterpret_cast<char*>(&num),sizeof(num));
		file.write(reinterpret_cast<char*>(&num),sizeof(num));
		file.write(reinterpret_cast<char*>(&num),sizeof(num));
	}
	file.close();
	return 0;
}

