#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <windows.h>
#include "tools.cuh"

__global__ void printThreadIndex()
{	
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	printf("ix: %d, iy: %d\n", ix, iy);
}

// GPU计算二维矩阵加法
__global__ void sumMatrix_GPU(float* d_a, float* d_b, float* d_res)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = ix + iy*blockDim.x*gridDim.x;   // 此处设计成16*16的线程，可以这样计算矩阵索引
	d_res[index] = d_a[index] + d_b[index];
}

void sumMatrix_CPU(float* h_a, float* h_b, float* h_res, const int nx, const int ny)
{
	for (int i = 0; i < ny ; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			h_res[j] = h_a[j] + h_b[j];
		}
		h_a = h_a + nx;
		h_b = h_b + nx;
		h_res = h_res + nx;
	}
	// 直接相加，可以运行
	//for (int i = 0; i < nx * ny; i++)
	//{
	//	h_res[i] = h_a[i] + h_b[i];
	//}
}


void kernel_sumMatrix()
{	
	// 选择设备
	int dev = 0;
	cudaSetDevice(dev);

	//定义矩阵大小
	int nx = 1600;
	int ny = 1600;
	int nBytes = nx * ny * sizeof(float);
	
	//主机内存申请
	float* h_a = (float*)malloc(nBytes);  
	float* h_b = (float*)malloc(nBytes);
	float* h_res = (float*)malloc(nBytes);
	float* h_res_fromGPU = (float*)malloc(nBytes);

	//设备内存申请
	float* d_a = NULL;
	float* d_b = NULL;
	float* d_res = NULL;
	cudaMalloc((float**)&d_a, nBytes);
	cudaMalloc((float**)&d_b, nBytes);
	cudaMalloc((float**)&d_res, nBytes);

	//初始化主机数据，将二维矩阵转换为内存存储的一维数组
	initialData(h_a, nx * ny);
	initialData(h_b, nx * ny);	
	
	// 将主机数据拷贝到设备
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

	//定义线程块和网格
	dim3 block(20, 40);
	dim3 grid(80, 40);
	
	// 调用核函数
	sumMatrix_GPU<< <grid,block>> >(d_a, d_b, d_res);
	cudaDeviceSynchronize();

	cudaEvent_t start, stop;
	float duration_gpu = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	sumMatrix_GPU << <grid, block >> > (d_a, d_b, d_res);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration_gpu, start, stop);
	std::cout << "GPU运行时间 = " << duration_gpu << "ms" << std::endl;

    // Windows下CPU精准计时方法
	LARGE_INTEGER  large_interger;
	double dff;
	__int64  c1, c2;
	double duration;
	QueryPerformanceFrequency(&large_interger);
	dff = large_interger.QuadPart;
	QueryPerformanceCounter(&large_interger);
	c1 = large_interger.QuadPart;

	sumMatrix_CPU(h_a, h_b, h_res, nx, ny);

	QueryPerformanceCounter(&large_interger);
	c2 = large_interger.QuadPart;
	duration = (c2 - c1) * 1000 / dff;
	std::cout << "CPU运行时间 = " << duration << "ms" << std::endl;
	
	cudaMemcpy(h_res_fromGPU, d_res, nBytes, cudaMemcpyDeviceToHost);  //要将显存中的数据复制到主机端才能进行比较
	checkResult(h_res, h_res_fromGPU, nx * ny);


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);

	free(h_a);
	free(h_b);
	free(h_res);
	free(h_res_fromGPU);
}