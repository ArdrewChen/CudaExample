// �ȽϽṹ�����������ṹ�������
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "hello_world.cuh"

struct  MyStruct
{
	float x;
	float y;
};

__global__ void sumArrayAOSGPU() 
{}

void kernel_AOSwithSOA()
{
	int dev = 0;
	cudaSetDevice(dev);

	int nums = 1024;

	// �ṹ������
	MyStruct *d_AOS;
	cudaMalloc((void**)&d_AOS, nums * sizeof(MyStruct));

}