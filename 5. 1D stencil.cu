#include<stdio.h>
#define N (2048*2048)
#define M 512
#define RADIUS 3


// Introducing Stencil 
// Consider applying 1D stencil to a 1D array of elements
// 		if the radius is 3, then 1 at the center, right and left are 3
// 		output is the sum of 3+3+1=7 input elements

// Sharing data between threads
// within a block, threads share data via shared memory
// Extremely fast on-chip memory
// 		- By opposition to device memory, referred to as global memory
// 		- Like a user-managed cache
// declared using __shared__ , allocated per block
// Data is not visible to threads in other blocks 
//
// Implementing with Shared Memory
// Cache data in shared memory
// Read (blockDim.x + 2*radius) input elements from global memory to shared memory
// Compute blockDim.x output elements
// Write blockDim.x output elements to global memory
// halo on the right and left are created, based on the number of radius 

// void __syncthreads();
// to prevent data hazards
// synchronize all threads within a block
// 		used to prevent RAW/WAR/WAW hazards
// All threads must reach the barrier
//		in conditional code, the condition must be uniform across the block

// Launching parallel threads
// 		- Launch N blocks with M threads per block with kernel<<<N,M>>>();
// 		- Use blockIdx.x to access block index within grid
//		- Use threadIdx.x to access thread index within block
// Allocate elements to threads:
//		int index = threadIdx.x + blockIdx.x * blockDim.x

__global__ void stencil_1d(int* in, int* out) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + RADIUS;

	// Read input elements into shared memory
	temp[lindex] = in[gindex];
	if(threadIdx.x < RADIUS) {
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
	}

	// Synchronize (ensure all the data is available)
	__syncthreads();

	// Apply the stencil
	int result = 0;
	for (int offset = -RADIUS; offset<=RADIUS; offset++) {
		result += temp[lindex+offset];
	}

	out[gindex] = result;
}