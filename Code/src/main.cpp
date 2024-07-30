#include <iostream>
#include "hello_world.cuh"

int main() {
	kernel_hello_world();
	std::cout<< "helloworld from CPU" << std::endl;
}