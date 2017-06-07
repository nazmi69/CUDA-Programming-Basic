#include<stdio.h>
#define N 512

// __global__ indicates that:
// 	Runs on the Device (GPU)
// 	called from host code (CPU)

// Introducing Threads
// A block can be split into parallel threads
// instead of using parallel blocks, we use threads
// by using threadIdx.x instead of blockIdx.x

__global__ void add(int *a, int *b, int *c){
	// *c = *a + *b;
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int *a, int n) {
	int i;
	for(i=0; i<n; ++i) {
		a[i] = rand()%1000;
	}
}

// To use threads, instead of <<<N,1>>>>, we use<<<1,N>>>>

int main (void) {
	// Declare the host copies of a,b,c
	// int a, b, c;
	int *a, *b, *c;

	// Declare the device copies of a,b,c
	int *d_a, *d_b, *d_c;

	// Declare the size of int
	int size = N * sizeof(int);

	// Allocate space for device copies of a,b,c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	
	// Allocate space for host copies of a,b,c
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);

	// Setup the input values
	// a = 4;
	// b = 6

	// Copy inputs to device
	// cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	// <<<1,N>>> , with N threads
	add<<<1,N>>>(d_a, d_b, d_c);

	// Copy result back to host
	// cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	// Print the value
	for (int count=0; count<10; count++) {
		printf("%i + %i = %i\n", a[count], b[count], c[count]);
	}

	// Cleanup
	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}