# CUDA示例

## 项目介绍

在学习 CUDA 编程时的一些示例。

## 运行环境

`Windows 10/11`+`CUDA 11.6`+`VS2022`

安装教程：[CUDA与cuDNN安装教程（超详细）-CSDN博客](https://blog.csdn.net/anmin8888/article/details/127910084)

注意事项：若电脑没有 VS ，建议首先安装 VS ，然后安装 CUDA ，安装 CUDA 时记得勾选

![image-20240907163417896](images\image-20240907163417896.png)

检查是否安装成功，在命令行输入

```shell
nvcc -V
```

输出，即安装成功

![image-20240907163634257](images\image-20240907163634257.png)

## 函数介绍

### `kernel_hello_world()`

GPU输出`hello，world`。

### `kernel_thread_id()`

打印线程索引。

### `kernel_sumArray()`

求两个数组之和。

### `kernel_sumMatrix()`

求两个矩阵的和。

### `kernel_sumVector();`

求一个向量内部所有元素的累加和以及使用一些优化方法改进线程束优化。

### 未完待续