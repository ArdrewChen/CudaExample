#include <stdio.h>
#include <cuda_runtime.h>


// �˺���hello_world
__global__ void hello_world() {
	printf("Hello World from GPU!\n");
}


// �˺������ɺ���
void kernel_hello_world() {
	hello_world << <2, 5>> > ();
	
	cudaDeviceReset(); //��仰���û�У���������������
	printf("123");
}
