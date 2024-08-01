#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "tools.cuh"


__global__ void printThreadIndex()
{
	printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
	//std::cout<< "blockIdx.x: " << blockIdx.x << ", blockIdx.y: " << blockIdx.y << ", blockIdx.z: " << blockIdx.z << std::endl;
}

void kernel_sumMatrix()
{	
	int dev = 0;
	cudaSetDevice(dev);

	//定义矩阵大小
	int nx = 16;
	int ny = 16;
	int nBytes = nx * ny * sizeof(float);
	
	//主机内存申请
	float* h_a = (float*)malloc(nBytes);

	//设备内存申请
	float* d_a = NULL;
	cudaMalloc((float**)&d_a, nBytes);

	//初始化主机数据
	initialData(h_a, nx * ny);


	
	dim3 block(16, 16);
	dim3 grid(2, 2);
	//printThreadIndex<<<grid, block>>>();
	cudaDeviceSynchronize();
}