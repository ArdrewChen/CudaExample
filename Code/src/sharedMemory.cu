// �����ڴ�Ĵ洢�����ξ���
// �Ż�Ŀ�꣬���⹲���ڴ�洢���ͻ
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "hello_world.cuh"

#define N  32

__global__ void sharedMemory()
{
	
}


void kernel_sharedMemory() 
{
	// ѡ���豸
	int dev = 0;
	cudaSetDevice(dev);

	//��������С
	int nx = 32;
	int ny = 32;
	int nBytes = nx * ny * sizeof(float);
}