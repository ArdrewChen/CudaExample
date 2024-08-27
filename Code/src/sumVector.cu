// ���һ������������Ԫ�صĺͣ�ʹ�ù�Լ��
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "tools.cuh"

// CPUʵ����ͣ���֤�����ȷ��
float sumVector_CPU(float *idata, int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum = sum + idata[i];
	}
	return sum;
}

// GPU���й�Լʵ�֣��������֮�ͣ�δ���Ʒֻ����⣬�߳�����Ч�ʵ�
__global__ void sumVector_GPU(float *idata, float *odata, unsigned int size) 
{
	int tid = threadIdx.x;
	float *idata_tmp = idata + blockIdx.x * blockDim.x;

	// forѭ��һ���Ǽ���һ���߳̿��Ԫ��
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if (tid % (2 * stride) == 0)              // �ڴ˴�������֧
		{
			idata_tmp[tid] += idata_tmp[tid + stride];
		}
		__syncthreads(); // ͬ�������߳�
	}
	if (tid == 0)
	{
		odata[blockIdx.x] = idata_tmp[0];
	}
}

// ���Ʒֻ�1���߳�������������������һһ��Ӧ
__global__ void sumVectorImproved1_GPU(float *idata, float *odata, unsigned int size) 
{
	int tid = threadIdx.x;
	float *idata_tmp = idata + blockIdx.x * blockDim.x;
	for(int stride=1; stride<blockDim.x; stride *= 2)
	{
		int index = 2 * stride * tid;	// �ؼ����裬�����������������ӳ�䵽�߳����������Ǹ����߳����������ƶ������ٷֻ�
										// ȷ��ǰ�����߳�������	 
		if(index < blockDim.x)
		{
			idata_tmp[index] += idata_tmp[index + stride];
		}
		__syncthreads();
	}
	if(tid == 0)
	{
		odata[blockIdx.x] = idata_tmp[0];
	}

}

// ���Ʒֻ�2����������������ֳɶ���飬ʹ�ö���߳̿鴦�����ݣ��Ƚ���һ�������ӷ���Ȼ���Լ
__global__ void sumVectorImproved2_GPU(float* idata, float* odata, unsigned int size)
{
	int tid = threadIdx.x;
	float* idata_tmp = idata + blockIdx.x * blockDim.x * 2;
	int idx = tid + blockIdx.x * blockDim.x * 2;
	if (idx + blockDim.x < size)		// �ؼ����裬�Ƚ���һ�������ӷ�
	{
		idata[idx] += idata[idx + blockDim.x];
	}
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < blockDim.x)
		{
			idata_tmp[tid] += idata_tmp[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		odata[blockIdx.x] = idata_tmp[0];
	}
}


// Ԥ�Ⱥ���
__global__ void warmup() {
	printf("just warmup!\n");
}

void kernel_sumVector() 
{
	// ѡ���豸
	int dev = 0;
	cudaSetDevice(dev);

	//��������С
	int nx = 1024;
	int nBytes = nx  * sizeof(float);

	//�����ڴ�����
	float* h_a = (float*)malloc(nBytes);
	float* h_res_fromGPU = (float*)malloc(nBytes);

	//�豸�ڴ�����
	float* d_a = NULL;
	float* d_res = NULL;
	cudaMalloc((float**)&d_a, nBytes);
	cudaMalloc((float**)&d_res, nBytes);

	//��ʼ���������ݣ���ʼ������
	initialData(h_a, nx);	
	
	//�����߳̿������
	dim3 block(nx, 1);    // ����ά�ȣ��޷���������
	dim3 grid((nx-1)/ block.x +1, 1);

	// Ԥ��
	warmup<< <1,1 >> > ();
	cudaDeviceSynchronize();

	// ���ú˺���
	// ���������ݿ������豸
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice); // δ���Ʒֻ�����
	cudaEvent_t start, stop;
	float duration_gpu = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	sumVector_GPU << <grid, block >> > (d_a, d_res, nx);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration_gpu, start, stop);
	std::cout << "sumVector_GPU����ʱ�� = " << duration_gpu << "ms" << std::endl;

	// ���������������
	cudaMemcpy(h_res_fromGPU, d_res, nBytes, cudaMemcpyDeviceToHost);
	float sum = 0;
	sum = h_res_fromGPU[0];
	std::cout << "sumVector_GPU sum = " << sum << std::endl;


	// ���и��Ʒֻ�1
	// ���������ݿ������豸
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice); // δ���Ʒֻ�����
	cudaEvent_t start1, stop1;
	float duration_gpu1 = 0.0000f;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);

	sumVectorImproved1_GPU << <grid, block >> > (d_a, d_res, nx);
	cudaDeviceSynchronize();

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(start1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&duration_gpu1, start1, stop1);
	std::cout << "sumVectorImproved1_GPU����ʱ�� = " << duration_gpu1 << "ms" << std::endl;

	// ���������������
	cudaMemcpy(h_res_fromGPU, d_res, nBytes, cudaMemcpyDeviceToHost);
	float sum1 = 0;
	sum1 = h_res_fromGPU[0];
	std::cout << "sumVectorImproved_GPU sum = " << sum1 << std::endl;


	// ���и��Ʒֻ�2������2��Ҫ����飬��Ҫ����block��grid��ֵ
	int block_n = 2; // ���ֵĿ���
	dim3 block2(nx/block_n, 1);
	dim3 grid2((nx/block_n - 1) / block.x + 1, 1);
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice); 
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);

	sumVectorImproved2_GPU << <grid2, block2 >> > (d_a, d_res, nx);
	cudaDeviceSynchronize();

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(start1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&duration_gpu1, start1, stop1);
	std::cout << "sumVectorImproved2_GPU����ʱ�� = " << duration_gpu1 << "ms" << std::endl;

	// ���������������
	cudaMemcpy(h_res_fromGPU, d_res, nBytes, cudaMemcpyDeviceToHost);
	sum1 = h_res_fromGPU[0];
	std::cout << "sumVectorImproved2_GPU sum = " << sum1 << std::endl;


	// CPUִ��
	float sum_cpu = 0;
	sum_cpu = sumVector_CPU(h_a, nx);
	std::cout << "CPU sum = " << sum_cpu << std::endl;



	// �ͷ��ڴ�
	cudaFree(d_a);
	cudaFree(d_res);


	free(h_a);
	free(h_res_fromGPU);

}