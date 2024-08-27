// 求解一个向量的所有元素的和，使用归约法
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "tools.cuh"

// CPU实现求和，验证结果正确性
float sumVector_CPU(float *idata, int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum = sum + idata[i];
	}
	return sum;
}

// GPU并行归约实现，求解向量之和，未改善分化问题，线程利用效率低
__global__ void sumVector_GPU(float *idata, float *odata, unsigned int size) 
{
	int tid = threadIdx.x;
	float *idata_tmp = idata + blockIdx.x * blockDim.x;

	// for循环一次是计算一个线程块个元素
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if (tid % (2 * stride) == 0)              // 在此处产生分支
		{
			idata_tmp[tid] += idata_tmp[tid + stride];
		}
		__syncthreads(); // 同步块内线程
	}
	if (tid == 0)
	{
		odata[blockIdx.x] = idata_tmp[0];
	}
}

// 改善分化1：线程索引和数组索引不再一一对应
__global__ void sumVectorImproved1_GPU(float *idata, float *odata, unsigned int size) 
{
	int tid = threadIdx.x;
	float *idata_tmp = idata + blockIdx.x * blockDim.x;
	for(int stride=1; stride<blockDim.x; stride *= 2)
	{
		int index = 2 * stride * tid;	// 关键步骤，将数组的索引不在是映射到线程索引，而是根据线程索引进行移动，减少分化
										// 确保前几个线程束跑满	 
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

// 改善分化2：将待处理的向量分成多个块，使用多个线程块处理数据，先进行一次向量加法，然后归约
__global__ void sumVectorImproved2_GPU(float* idata, float* odata, unsigned int size)
{
	int tid = threadIdx.x;
	float* idata_tmp = idata + blockIdx.x * blockDim.x * 2;
	int idx = tid + blockIdx.x * blockDim.x * 2;
	if (idx + blockDim.x < size)		// 关键步骤，先进行一次向量加法
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


// 预热函数
__global__ void warmup() {
	printf("just warmup!\n");
}

void kernel_sumVector() 
{
	// 选择设备
	int dev = 0;
	cudaSetDevice(dev);

	//定义矩阵大小
	int nx = 1024;
	int nBytes = nx  * sizeof(float);

	//主机内存申请
	float* h_a = (float*)malloc(nBytes);
	float* h_res_fromGPU = (float*)malloc(nBytes);

	//设备内存申请
	float* d_a = NULL;
	float* d_res = NULL;
	cudaMalloc((float**)&d_a, nBytes);
	cudaMalloc((float**)&d_res, nBytes);

	//初始化主机数据，初始化数组
	initialData(h_a, nx);	
	
	//定义线程块和网格
	dim3 block(nx, 1);    // 更改维度，无法正常运行
	dim3 grid((nx-1)/ block.x +1, 1);

	// 预热
	warmup<< <1,1 >> > ();
	cudaDeviceSynchronize();

	// 调用核函数
	// 将主机数据拷贝到设备
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice); // 未改善分化数据
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
	std::cout << "sumVector_GPU运行时间 = " << duration_gpu << "ms" << std::endl;

	// 将结果拷贝到主机
	cudaMemcpy(h_res_fromGPU, d_res, nBytes, cudaMemcpyDeviceToHost);
	float sum = 0;
	sum = h_res_fromGPU[0];
	std::cout << "sumVector_GPU sum = " << sum << std::endl;


	// 并行改善分化1
	// 将主机数据拷贝到设备
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice); // 未改善分化数据
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
	std::cout << "sumVectorImproved1_GPU运行时间 = " << duration_gpu1 << "ms" << std::endl;

	// 将结果拷贝到主机
	cudaMemcpy(h_res_fromGPU, d_res, nBytes, cudaMemcpyDeviceToHost);
	float sum1 = 0;
	sum1 = h_res_fromGPU[0];
	std::cout << "sumVectorImproved_GPU sum = " << sum1 << std::endl;


	// 并行改善分化2，改善2需要多个块，需要更改block和grid的值
	int block_n = 2; // 划分的块数
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
	std::cout << "sumVectorImproved2_GPU运行时间 = " << duration_gpu1 << "ms" << std::endl;

	// 将结果拷贝到主机
	cudaMemcpy(h_res_fromGPU, d_res, nBytes, cudaMemcpyDeviceToHost);
	sum1 = h_res_fromGPU[0];
	std::cout << "sumVectorImproved2_GPU sum = " << sum1 << std::endl;


	// CPU执行
	float sum_cpu = 0;
	sum_cpu = sumVector_CPU(h_a, nx);
	std::cout << "CPU sum = " << sum_cpu << std::endl;



	// 释放内存
	cudaFree(d_a);
	cudaFree(d_res);


	free(h_a);
	free(h_res_fromGPU);

}