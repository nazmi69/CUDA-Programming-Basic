#include<stdio.h>
#define N (2048*2048)
#define M 512

// __global__ indicates that:
// 	Runs on the Device (GPU)
// 	called from host code (CPU)

// Combination of both blocks and threads
// But it's mpt as simple as using blockIdx.x and threadIdx.x
// Consider indexing an array with one element per thread
// 8 threads/block

//			 threadIdx.x = from 0 to 7
// |-----------------------------------------|
// |	 |	   |	 | 	   |	 | 	   |	 |
// |  0  |  2  |  3  |  4  |  5  |  6  |  7  |
// |	 |	   |	 |     |	 | 	   |	 |
// |-----------------------------------------|
//				    blockIdx.x = 0
//

//			 threadIdx.x = from 0 to 7
// |-----------------------------------------|
// |	 |	   |	 | 	   |	 | 	   |	 |
// |  0  |  2  |  3  |  4  |  5  |  6  |  7  |
// |	 |	   |	 |     |	 | 	   |	 |
// |-----------------------------------------|
//				    blockIdx.x = 1
//

// index = thread + block*M

// Why bother with threads as they seem unnecessary
// 		They add a level of complexity
//		What do we gain?
//			threads have mechanisms to efficiently: communicate, synchronize

__global__ void add(int *a, int *b, int *c, int n){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (index < n)
		c[index] = a[index] + b[index];
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
	add<<<(N + M-1)/M, M>>>(d_a, d_b, d_c, N);

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