#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "tools.cuh"


__global__ void printThreadIndex()
{	
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	printf("ix: %d, iy: %d\n", ix, iy);
}

__global__ void sumMatrix_GPU()
{

}

void sumMatrix_CPU()
{

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
	float* h_b = (float*)malloc(nBytes);
	float* h_res = (float*)malloc(nBytes);

	//设备内存申请
	float* d_a = NULL;
	float* d_b = NULL;
	float* d_res = NULL;
	cudaMalloc((float**)&d_a, nBytes);
	cudaMalloc((float**)&d_b, nBytes);
	cudaMalloc((float**)&d_res, nBytes);

	//初始化主机数据
	initialData(h_a, nx * ny);
	initialData(h_b, nx * ny);	
	
	// 将主机数据拷贝到设备
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

	//定义线程块和网格
	dim3 block(4, 3);
	dim3 grid(2, 1);
	
	// 打印线程索引，测试线程索引公式
	// printThreadIndex<<<grid, block>>>();
	cudaDeviceSynchronize();
}