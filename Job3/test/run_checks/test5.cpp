#include<iostream>
#include<cfloat>
#include<iomanip>
using namespace std;
int main(){
	float f=0.0;
	float b=2.0;
	for(int a=0;a<20000000;a++){
		f=f+b;
	}
	cout<<setprecision(8)<<f<<endl;
	return 0;
}
