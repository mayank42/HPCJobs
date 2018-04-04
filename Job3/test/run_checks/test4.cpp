#include<iostream>
#include<fstream>
using namespace std;
int main(){
	ifstream file("./data/mat.dat",ios::binary);
	if(file.is_open())cout<<"Mayank"<<endl;
	file.close();
	return 0;
}
