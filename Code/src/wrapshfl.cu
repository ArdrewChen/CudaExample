// 线程束洗牌指令
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

# define N 32

__device__ int warpShfl(int value, int lane)
{
	return __shfl_sync(0xFFFFFFFF, value, lane);
}

__device__ int warpShflXor(int value, int lane, int width)
{
	return __shfl_xor_sync(0xFFFFFFFF, value, lane, width);
}

__device__ int warpShflUp(int value, int lane, int width)
{
	return __shfl_up_sync(0xFFFFFFFF, value, lane, width);
}


__device__ int warpShflDown(int value, int lane, int width)
{
	return __shfl_down_sync(0xFFFFFFFF, value, lane, width);
}



__global__ void kernel_wrapshfl(int* in, int* out)
{
	int value = in[threadIdx.x];
	//value = warpShfl(value, 1);
	//value = warpShflXor(value, 1, 32);
	//value = warpShflUp(value, 1, 32);
	value = warpShflDown(value, 1, 32);
	out[threadIdx.x] = value;
}



void kernel_wrapshfl()
{
	// 选择设备
	int dev = 0;
	cudaSetDevice(dev);
	
	int nBytes = N * sizeof(int);
	int* h_in = (int*)malloc(nBytes);
	int* h_out = (int*)malloc(nBytes);

	int* d_in = NULL;
	int* d_out = NULL;

	cudaMalloc((int**)&d_in, nBytes);
	cudaMalloc((int**)&d_out, nBytes);

	printf("input data: ");

	for(int i=0; i<N; i++)
	{
		h_in[i] = i;
		printf("%d ", h_in[i]);
	}
	printf("\n");

	dim3 block(N, 1);
	dim3 grid(1, 1);
	cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);
	kernel_wrapshfl << <grid, block >> > (d_in,d_out);
	cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	printf("output data: ");
	for (int i = 0; i < N; i++)
	{
		printf("%d ", h_out[i]);
	}
	printf("\n");
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
}