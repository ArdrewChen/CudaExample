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

	//��������С
	int nx = 16;
	int ny = 16;
	int nBytes = nx * ny * sizeof(float);
	
	//�����ڴ�����
	float* h_a = (float*)malloc(nBytes);

	//�豸�ڴ�����
	float* d_a = NULL;
	cudaMalloc((float**)&d_a, nBytes);

	//��ʼ����������
	initialData(h_a, nx * ny);


	
	dim3 block(16, 16);
	dim3 grid(2, 2);
	//printThreadIndex<<<grid, block>>>();
	cudaDeviceSynchronize();
}