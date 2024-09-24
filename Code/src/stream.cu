// 一个简单的流的应用示例

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N  10000


__global__ void kernel_1()
{
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_2()
{
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_3()
{
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_4()
{
	double sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}


void kernel_stream()
{
	int dev = 0;
	cudaSetDevice(dev);
	const int n_stream = 8;

	// 创建流
	cudaStream_t stream[n_stream];
	for(int i=0; i<n_stream; i++)
	{
		cudaStreamCreate(&stream[i]);
	}
	dim3 block(1);
	dim3 grid(1);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < n_stream; i++)
	{
		kernel_1 << <grid, block, 0, stream[i] >> > ();
		kernel_2 << <grid, block, 0, stream[i] >> > ();
		kernel_3 << <grid, block, 0, stream[i] >> > ();
		kernel_4 << <grid, block, 0, stream[i] >> > ();
	}
	cudaEventRecord(stop, 0);

	// 同步
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to do multiple streams: %3.1f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	for (int i = 0; i < n_stream; i++)
	{
		cudaStreamDestroy(stream[i]);
	}
	cudaDeviceReset();

}