#include<stdio.h>

// __global__ indicates that:
// 	Runs on the Device (GPU)
// 	called from host code (CPU)
__global__ void myKernel(void){
	printf("Hello world\n");
}

int main (void) {

//  Hello world
//	printf("Hello world\n");
//	return 0;

//  Hello world with Device Code 
//  <<<1,1>>> triple angle brackets mean
// 	a call from host code to device code (aka kernel launch)
//	the parameters means that it will 
	myKernel<<<1,1>>>();
	return 0;
}