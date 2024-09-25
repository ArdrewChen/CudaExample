// �����ڴ�Ĵ洢�����ξ���
// �Ż�Ŀ�꣬���⹲���ڴ�洢���ͻ
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "tools.cuh"

#define N  32


// ��ȫ���ڴ�����ݿ����������ڴ�,������
__global__ void sharedMemoryRow(float* out,float* in)
{
	__shared__ float shared[N][N];
	unsigned int ix = threadIdx.x + threadIdx.y * blockDim.x;
	shared[threadIdx.y][threadIdx.x] = in[ix];
	__syncthreads();
	out[ix] = shared[threadIdx.y][threadIdx.x];
}



// ��ȫ���ڴ�����ݿ����������ڴ�,������
__global__ void sharedMemoryCol(float* out,float* in)
{
	__shared__ float shared[N][N];
	unsigned int ix = threadIdx.x + threadIdx.y * blockDim.x;
	shared[threadIdx.x][threadIdx.y] = in[ix];
	__syncthreads();
	out[ix] = shared[threadIdx.x][threadIdx.y];
}

// ��ȫ���ڴ�����ݿ�������̬�����ڴ�,������
__global__ void sharedMemoryRowDynamic(float* out, float* in)
{
	extern __shared__ float shared[];  //extern�ؼ��֣���ʾ��̬�����ڴ�,
	unsigned int ix = threadIdx.x + threadIdx.y * blockDim.x;
	shared[threadIdx.y * N + threadIdx.x] = in[ix];
	__syncthreads();
	out[ix] = shared[threadIdx.y * N + threadIdx.x];
}


void kernel_sharedMemory() 
{
	// ѡ���豸
	int dev = 0;
	cudaSetDevice(dev);


	int nBytes = N * N * sizeof(float);
	float* h_in = (float*)malloc(nBytes);	
	float* h_out_row = (float*)malloc(nBytes);
	float* h_out_col = (float*)malloc(nBytes);

	float* d_in = NULL;
	float* d_out_row = NULL;
	float* d_out_col = NULL;
	cudaMalloc((float**)&d_in, nBytes);
	cudaMalloc((float**)&d_out_row,nBytes);
	cudaMalloc((float**)&d_out_col, nBytes);


	dim3 block(N, N);
	dim3 grid(1, 1);
	initialData(h_in, N * N);
	cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

	sharedMemoryRow << <grid, block >> >(d_out_row,d_in);
	sharedMemoryCol << <grid, block >> >(d_out_col,d_in);
	sharedMemoryRowDynamic << <grid, block, nBytes >> >(d_out_row, d_in); // �˺�����Ҫָ����̬�����ڴ��С

	cudaMemcpy(h_out_row, d_out_row, nBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out_col, d_out_row, nBytes, cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	// ��ӡ���
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.2f ", h_out_row[i * N + j]);
		}
		printf("\n");
	}
	free(h_in);
    free(h_out_row);
	free(h_out_col);
	cudaFree(d_in);
    cudaFree(d_out_row);
	cudaFree(d_out_col);

}