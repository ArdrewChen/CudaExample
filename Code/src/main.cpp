#include <iostream>
#include "hello_world.cuh"

int main() {
	//kernel_hello_world();
	//kernel_thread_id();
	//kernel_sumArray();
	//kernel_sumMatrix();
	kernel_sumVector();
	std::cin.get();
}