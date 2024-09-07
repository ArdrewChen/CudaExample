#include <stdio.h>
#include <cuda_runtime.h>


// 核函数hello_world
__global__ void hello_world() {
	printf("Hello World from GPU!\n");
}


// 核函数过渡函数
void kernel_hello_world() {
	hello_world << <2, 5>> > ();
	
	cudaDeviceReset(); //这句话如果没有，则不能正常的运行
	printf("123");
}
