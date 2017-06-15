#include <stdio.h>

#define DIM_GRID 512
#define THREADS_PER_BLOCK 16

// Define the structure of a matrix
typedef struct {
	int width;
	int height;
	int* elements;
} Matrix;

// Run on device (GPU)
__global__ void MatMultKernel(Matrix a, Matrix b, Matrix c) {

	// Initialize the results to 0
	int cvalue = 0; 

	// Which row and column we are currently in
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > a.height || col > b.width) return;

	// Matrix multiplication
	for (int i=0; i<a.width; ++i)
		cvalue += (a.elements[row * a.width + i]) * (b.elements[i * b.width + col]);

	// Assign the results
	c.elements[row * c.width + col] = cvalue;
}

// Function which initialize all the 
// Device variables, 
// Device memory allocations,
// Matrix dimensions in device (GPU)
// 
void MatMultInit(const Matrix a, const Matrix b, Matrix c) {

	// Define device matrix variable
	// Define the width, height, size, device memory allocation and the value
	Matrix d_a;
	d_a.width = a.width;
	d_a.height = a.height;
	size_t size = a.width * a.height * sizeof(int);
	cudaMalloc(&d_a.elements, size);
	cudaMemcpy(d_a.elements, a.elements, size, cudaMemcpyHostToDevice);

	// Define device matrix variable
	// Define the width, height, size, device memory allocation and the value
	Matrix d_b;
	d_b.width = b.width;
	d_b.height = b.height;
	size = b.width * b.height * sizeof(int);
	cudaMalloc(&d_b.elements, size);
	cudaMemcpy(d_b.elements, b.elements, size, cudaMemcpyHostToDevice);

	// Define device matrix variable
	// Define the width, height, size, device memory allocation
	// Value is not defined because the value will be set in the device function
	Matrix d_c;
	d_c.width = c.width;
	d_c.height = c.height;
	size = c.width * c.height * sizeof(int);
	cudaMalloc(&d_c.elements, size);

	// Initialize the dimension size for the block, and the grid
	// The grid dimension is set as per calculated,
	// 		to avoid accessing beyond the end of the arrays 
	dim3 dimBlock(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
	dim3 dimGrid(
		(b.width + dimBlock.x - 1) / dimBlock.x, 
		(a.height + dimBlock.y - 1) / dimBlock.y
	);

	// Call the device code
	MatMultKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

	// cudaThreadSynchronize (Deprecated, currently cudaDeviceSynchronize)
	// Synchronizing the CPU with the GPU 
	// Halt the execution in the CPU, until GPU has finished processing
	// 		all previously requested CUDA tasks
	cudaThreadSynchronize();

	// Copy the results to the CPU
	cudaMemcpy(c.elements, d_c.elements, size, cudaMemcpyDeviceToHost);

	// Free up the memory of d_a, d_b
	cudaFree(d_a.elements);
	cudaFree(d_b.elements);
}

int main (void) {
	Matrix a, b, c;

	// Define Matrix a 3x4
	a.height = 3;
	a.width = 4;

	// Define Matrix b 4x4
	b.height = 4;
	b.width = 4;

	// Define Matrix c 3x4
	c.height = 3;
	c.width = 4;


	// Allocate block of memory for each variable
	a.elements = (int*) malloc(a.width * a.height * sizeof(int));
	b.elements = (int*) malloc(b.width * b.height * sizeof(int));
	c.elements = (int*) malloc(c.width * c.height * sizeof(int));

	// Initialize variable with random number
	for(int i = 0; i < a.height; i++)
		for(int j = 0; j < a.width; j++)
			a.elements[i*a.width + j] = (int)(rand() % 100);
	
	// Initialize variable with random number
	for(int i = 0; i < b.height; i++)
		for(int j = 0; j < b.width; j++)
			b.elements[i*b.width + j] = (int)(rand() % 100);

	// Execute the matrix multiplication
	MatMultInit(a, b, c);

	// Print A value
	for(int i = 0; i < min(10, a.height); i++){
		for(int j = 0; j < min(10, a.width); j++)
			printf("%d ", a.elements[i*a.width + j]);
			printf("\n");
		}
	printf("\n");

	// Print B value
	for(int i = 0; i < min(10, b.height); i++){
		for(int j = 0; j < min(10, b.width); j++)
			printf("%d ", b.elements[i*b.width + j]);
			printf("\n");
		}
	printf("\n");

	// Print C value
	for(int i = 0; i < min(10, c.height); i++){
		for(int j = 0; j < min(10, c.width); j++)
			printf("%d ", c.elements[i*c.width + j]);
			printf("\n");
		}
	printf("\n");

}