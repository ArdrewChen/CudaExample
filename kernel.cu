#include <stdio.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include<device_functions.h>



__global__ void printThreadIndex(float* A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
		"global index %2d ival %2d\n", threadIdx.x, threadIdx.y,
		blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}


__global__ void checkIndex(void)
{
	printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) gridDim(%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
		blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
		gridDim.x, gridDim.y, gridDim.z);
}

extern "C" int addkernel() {

	int nx = 8, ny = 6;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);



	//cudaMalloc
	float* A_dev = NULL;
	cudaMalloc((void**)&A_dev, nBytes);

	dim3 block(4, 2);
	dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);

	printThreadIndex << <grid, block >> > (A_dev, nx, ny);

	cudaDeviceSynchronize();
	cudaFree(A_dev);

	cudaDeviceReset();
	return 0;
}