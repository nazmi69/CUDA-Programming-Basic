#include <stdio.h>

__global__ void kernel (int a, int b, int* c) {
	*c = a * b;
}

int main (void) {
	int c;
	int *dev_c;

	cudaMalloc(&dev_c, sizeof(int));
	kernel<<<1,1>>>(2, 3, dev_c);

	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("%d x %d = %d\n", 2, 3, c);
	cudaFree(dev_c);

	return 0;
}