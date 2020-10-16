
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;
#include <stdio.h>
#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>

void checkCudaError(const char* msg)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error: %s: %s. \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__global__ void render(float3* result, unsigned int width, unsigned int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height))
		return;
	int index = j * width + i;
	result[index] = make_float3(i / (float)width, j / (float)height, 0.0f);
	return;
}

extern "C" void launch_pbo_kernel(float3* result, unsigned int width, unsigned int height)
{
	int tx = 8;
	int ty = 8;

	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);

	render << <blocks, threads >> > (result, width, height);

	checkCudaError("pbo kernel failed!");
}


__global__ void update(float3* pos, unsigned int N)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	float u = x / (float)N;
	float v = y / (float)N;
	float w = z / (float)N;

	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	w = w * 2.0f - 1.0f;

	unsigned int index = x * N * N + y * N + z;
	// vert start
	pos[2 * index] = make_float3(u, v, w);
	// vert end
	pos[2 * index + 1] = pos[2 * index] + make_float3(0.0f, 0.1f, 0.0f);
}

extern "C" void launch_vbo_kernel(float3* pos, unsigned int N)
{
	dim3 block(4, 4, 4);
	dim3 grid(N / block.x, N / block.y, N / block.z);

	update << <grid, block >> > (pos, N);

	checkCudaError("vbo kernel failed!");

	cudaThreadSynchronize();
}
