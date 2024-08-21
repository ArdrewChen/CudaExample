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

// GPU�����ά����ӷ�
__global__ void sumMatrix_GPU(float* d_a, float* d_b, float* d_res)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = ix + iy*blockDim.x*gridDim.x;   // �˴���Ƴ�16*16���̣߳��������������������
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
	// ֱ����ӣ���������
	//for (int i = 0; i < nx * ny; i++)
	//{
	//	h_res[i] = h_a[i] + h_b[i];
	//}
}


void kernel_sumMatrix()
{	
	// ѡ���豸
	int dev = 0;
	cudaSetDevice(dev);

	//��������С
	int nx = 1600;
	int ny = 1600;
	int nBytes = nx * ny * sizeof(float);
	
	//�����ڴ�����
	float* h_a = (float*)malloc(nBytes);  
	float* h_b = (float*)malloc(nBytes);
	float* h_res = (float*)malloc(nBytes);
	float* h_res_fromGPU = (float*)malloc(nBytes);

	//�豸�ڴ�����
	float* d_a = NULL;
	float* d_b = NULL;
	float* d_res = NULL;
	cudaMalloc((float**)&d_a, nBytes);
	cudaMalloc((float**)&d_b, nBytes);
	cudaMalloc((float**)&d_res, nBytes);

	//��ʼ���������ݣ�����ά����ת��Ϊ�ڴ�洢��һά����
	initialData(h_a, nx * ny);
	initialData(h_b, nx * ny);	
	
	// ���������ݿ������豸
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

	//�����߳̿������
	dim3 block(20, 40);
	dim3 grid(80, 40);
	
	// ���ú˺���
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
	std::cout << "GPU����ʱ�� = " << duration_gpu << "ms" << std::endl;

    // Windows��CPU��׼��ʱ����
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
	std::cout << "CPU����ʱ�� = " << duration << "ms" << std::endl;
	
	cudaMemcpy(h_res_fromGPU, d_res, nBytes, cudaMemcpyDeviceToHost);  //Ҫ���Դ��е����ݸ��Ƶ������˲��ܽ��бȽ�
	checkResult(h_res, h_res_fromGPU, nx * ny);


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);

	free(h_a);
	free(h_b);
	free(h_res);
	free(h_res_fromGPU);
}