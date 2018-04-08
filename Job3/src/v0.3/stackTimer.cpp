#include "stackTimer.h"
using namespace std;
deque<struct timespec> deq;
double push_clock(){
	struct timespec t1;
	clock_gettime(CLOCK_MONOTONIC,&t1);
	double diff;
	if(deq.empty()){
		deq.push_back(t1);
		return 0.0;
	}
	else{
		diff = (t1.tv_nsec-deq.back().tv_nsec)*1.0/1000000.0;
		deq.push_back(t1);
		return diff;
	}
}
void clear_clock(){
	deq.clear();
}
