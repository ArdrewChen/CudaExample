// 共享内存的存储，方形矩阵
// 优化目标，避免共享内存存储体冲突
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
	// 选择设备
	int dev = 0;
	cudaSetDevice(dev);

	//定义矩阵大小
	int nx = 32;
	int ny = 32;
	int nBytes = nx * ny * sizeof(float);
}