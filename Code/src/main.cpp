#include <iostream>
#include "hello_world.cuh"

int main() {
	//kernel_hello_world();
	kernel_thread_id();
	std::cout<< "helloworld from CPU" << std::endl;
}