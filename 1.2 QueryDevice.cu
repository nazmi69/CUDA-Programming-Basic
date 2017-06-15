#include <stdio.h>

__global__ void kernel() {

}

int main(void) {

	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);

	for(int i=0; i<count; i++) {
		cudaGetDeviceProperties(&prop, i);

		printf("------ General Information for device %d ------\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock Rate: %d\n", prop.clockRate);
		printf("Device Copy Overlap: ");
		if(prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel Execution timeout: ");
		if(prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");

		printf("\n");

		printf("------ Memory Information for device %d ------\n", i);
		printf("Total Global Memory: %zd MB\n", prop.totalGlobalMem/(1024*1024));
		printf("Total Constant Memory: %zd kB\n", prop.totalConstMem/1024);
		printf("Max Memory Pitch: %zd MB\n", prop.memPitch/(1024*1024));
		printf("Texture Alignment: %zd\n", prop.textureAlignment);

		printf("\n");

		printf("------ MP Information for device %d ------\n", i);
		printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
		printf("Shared Memory per MP: %zd\n", prop.sharedMemPerBlock);
		printf("Registers per MP: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
		printf("Max Threads Dimension: (%d, %d, %d)\n", 
			prop.maxThreadsDim[0],
			prop.maxThreadsDim[1],
			prop.maxThreadsDim[2]
		);
		printf("Max Grid Dimension: (%d, %d, %d)\n", 
			prop.maxGridSize[0],
			prop.maxGridSize[1],
			prop.maxGridSize[2]
		);
		printf("\n");

	}
}