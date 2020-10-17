
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

typedef struct
{
	float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

extern "C"
void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix)
{
	cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix);
	checkCudaError("Constant memcpy failed!");
}

struct Ray
{
	float3 origin;
	float3 dir;
};

// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
__device__ bool intersectAABB(Ray ray, float3 boxMin, float3 boxMax, float* tNear, float* tFar)
{
	float3 invR = make_float3(1.0f) / ray.dir;
	float3 tBottom = invR * (boxMin - ray.origin);
	float3 tTop = invR * (boxMax - ray.origin);

	float3 tMin = fminf(tTop, tBottom);
	float3 tMax = fmaxf(tTop, tBottom);

	float largest_tMin = fmaxf(fmaxf(tMin.x, tMin.y), fmaxf(tMin.x, tMin.z));
	float smallest_tMax = fminf(fminf(tMax.x, tMax.y), fminf(tMax.x, tMax.z));

	*tNear = largest_tMin;
	*tFar = smallest_tMax;

	return smallest_tMax > largest_tMin;
}

__device__ float3 mul(const float3x4& M, const float3& v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

__device__ float4 mul(const float3x4& M, const float4& v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

__global__ void render(float3* result, unsigned int width, unsigned int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height))
		return;
	int index = j * width + i;

	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	float u = (i / (float)width) * 2.0f - 1.0f;
	float v = (j / (float)height) * 2.0f - 1.0f;

	// calculate eye ray in world space
	Ray ray;
	ray.origin = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	ray.dir = normalize(make_float3(u, -v, -2.0f));
	ray.dir = mul(c_invViewMatrix, ray.dir);

	// background color
	result[index] = ray.dir;

	// intersect with AABB
	float tnear, tfar;
	bool hit = intersectAABB(ray, boxMin, boxMax, &tnear, &tfar);

	if(hit) result[index] = make_float3(1.0f);

	
	//if (c_invViewMatrix.m[0].x == 1.0f) result[index] = make_float3(0.0f, 0.0f, 1.0f);
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
	float step = 2.0f / N;
	// vert end
	pos[2 * index + 1] = pos[2 * index] + make_float3(0.0f, step, 0.0f);
}

extern "C" void launch_vbo_kernel(float3* pos, unsigned int N)
{
	dim3 block(4, 4, 4);
	dim3 grid(N / block.x, N / block.y, N / block.z);

	update << <grid, block >> > (pos, N);

	checkCudaError("vbo kernel failed!");

	cudaThreadSynchronize();
}
