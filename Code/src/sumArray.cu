/*ʵ�������������*/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

// GPU�˺�����ʵ���������
__global__ void sumArrayGPU(float* d_a, float* d_b, float* d_res)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_res[i] = d_a[i] + d_b[i];
}

// CPU������ʵ���������
void sumArrays(float* a, float* b, float* res, const int size)
{
	for (int i = 0; i < size; i++)
	{
		res[i] = a[i] + b[i];
	}
}

// ��֤����Ƿ���ȷ
void checkResult(float* res, float* res_from_gpu, const int size)
{
	for (int i = 0; i < size; i++)
	{
		if (res[i] != res_from_gpu[i])
		{
			printf("Error: %d element do not match!\n", i);
		}
	}
	printf("Check result success!\n");
}

// �����������
void initialData(float* ip, int size)
{
	// generate different seed for random number
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
	return;
}

void kernel_sumArray()
{
	int dev = 0;
	cudaSetDevice(dev);

	int nElem = 32;
	int nByte = nElem * sizeof(float);
	float* h_a = (float*)malloc(nByte);
	float* h_b = new float[nElem];
	float* h_res = (float*)malloc(nByte);
	float* h_res_from_gpu = new float[nElem];
	memset(h_res, 0, nByte);
	memset(h_res_from_gpu, 0, nByte);

	float* d_a, * d_b, * d_res;
	cudaMalloc((float**)&d_a, nByte);						//�����豸�ڴ�
	cudaMalloc((float**)&d_b, nByte);
	cudaMalloc((float**)&d_res, nByte);

	initialData(h_a, nElem);
	initialData(h_b, nElem);

	cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice);   //��h_a������d_a
	cudaMemcpy(d_b, h_b, nByte, cudaMemcpyHostToDevice);

	dim3 block(nElem / 4);
	dim3 grid(nElem / block.x);

	sumArrayGPU << <grid, block >> > (d_a, d_b, d_res);  // ��ʱǰ��ΪԤ�Ⱥ���

	// ��Ӽ�ʱ��
	cudaEvent_t start, stop;
	float duration_gpu = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	sumArrayGPU << <grid, block >> > (d_a, d_b, d_res);  // ִ�к˺���

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration_gpu, start, stop);
	std::cout << "����ʱ�� = " << duration_gpu << "ms" << std::endl;

	printf("Execution configuration<<<%d,%d>>>\n", block.x, grid.x);
	cudaMemcpy(h_res_from_gpu, d_res, nByte, cudaMemcpyDeviceToHost);
	sumArrays(h_a, h_b, h_res, nElem);

	checkResult(h_res, h_res_from_gpu, nElem);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);			//�ͷ��豸�ڴ�

	free(h_a);
	free(h_b);
	free(h_res);
	free(h_res_from_gpu);		//�ͷ������ڴ�

	cudaDeviceReset();
}