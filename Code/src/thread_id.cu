#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



__global__ void thread_id(void)
{
	printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) gridDim(%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
		blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
		gridDim.x, gridDim.y, gridDim.z);
}

void kernel_thread_id() 
{
	dim3 block(3,4,1);
	dim3 grid(2,3,1);

	thread_id << <grid, block >> > ();
	cudaDeviceReset(); 
}