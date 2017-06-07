#include <stdio.h>
#include <stdlib.h>
#define N 512

// 	__global__ indicates that:
// 	Runs on the Device (GPU)
// 	called from host code (CPU)
// 	we use pointers for the variables 
// 	pointed to the device memory

	
// Introducing Block
// each parallel invocation of add() is referred as Block
//		The set of blocks is referred to as a grid
//		Each invocation can refer to its block index using blockIdx.x
//		By using blockIdx.x to index into the array,
//			each block handles a different element of the array

// On the device, each block can execute in parallel
// Block 0 : c[0] = a[0] + b[0]
// Block 1 : c[1] = a[1] + b[1]
// Block 2 : c[2] = a[2] + b[2]
// Block 3 : c[3] = a[3] + b[3]

// Don't forget to change the size !

__global__ void add(int *a, int *b, int *c){
	// *c = *a + *b;

	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];

	// printf("Executed!\n");
}

void random_ints(int *a, int n) {
	int i;
	for(i=0; i<n; ++i) {
		a[i] = rand()%1000;
	}
}


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
	// <<<N,1>>> , execute N times in parallel
	add<<<N,1>>>(d_a, d_b, d_c);

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