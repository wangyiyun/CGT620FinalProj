
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

typedef unsigned char VolumeType;
cudaArray* d_volumeArray = 0;	// 3D texture Data
cudaTextureObject_t volumeTexObject; // 3D texture Object
cudaArray* d_transferFuncArray;
cudaTextureObject_t transferTexObject; // Transfer texture Object

__constant__ float fadeOut = 0.5f;

extern "C" void checkCudaError(const char* msg)
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

extern "C" void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix)
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

__device__ unsigned int Index(int x, int y, int z, unsigned int N)
{
	return x * N * N + y * N + z;
}

extern "C" void copyVolumeTextures(void* h_volume, cudaExtent volumeSize)
{
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize);
	checkCudaError("Cuda malloc 3D array failed!");

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	checkCudaError("Cuda memcpy 3D array failed!");

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_volumeArray;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaCreateTextureObject(&volumeTexObject, &texRes, &texDescr, NULL);
	checkCudaError("Cuda create volume texture object failed!");

	// create transfer function texture
	float4 transferFunc[] =
	{
		{  0.0, 0.0, 0.0, 0.0, },
		{  1.0, 0.0, 0.0, 1.0, },
		{  1.0, 0.5, 0.0, 1.0, },
		{  1.0, 1.0, 0.0, 1.0, },
		{  0.0, 1.0, 0.0, 1.0, },
		{  0.0, 1.0, 1.0, 1.0, },
		{  0.0, 0.0, 1.0, 1.0, },
		{  1.0, 0.0, 1.0, 1.0, },
		{  0.0, 0.0, 0.0, 0.0, },
	};

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	cudaArray* d_transferFuncArray;
	cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc) / sizeof(float4), 1);
	checkCudaError("Cuda malloc transfer texture failed!");
	cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice);
	checkCudaError("Cuda memcpy transfer texture failed!");

	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_transferFuncArray;

	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp; // wrap texture coordinates

	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&transferTexObject, &texRes, &texDescr, NULL);
	checkCudaError("Cuda create transfer texture failed!");
}

__global__ void render_3D_texture(float3* result, unsigned int width, unsigned int height, cudaTextureObject_t volumeTex,
	cudaTextureObject_t transferTex,
	float density, float transferOffset, float transferScale, unsigned int N,
	float3* outputVF)
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

	// ray marching parameters
	const int maxSteps = 500;
	const float tStep = 0.01f;
	const float opacityThreshold = 0.95f;

	// calculate camera ray in world space
	Ray ray;
	ray.origin = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	ray.dir = normalize(make_float3(u, -v, -2.0f));
	ray.dir = mul(c_invViewMatrix, ray.dir);

	// background color
	result[index] = make_float3(0.0f);

	// intersect with AABB
	float tNear, tFar;
	bool hit = intersectAABB(ray, boxMin, boxMax, &tNear, &tFar);

	if (!hit) return;

	// ray marching
	float4 sum = make_float4(0.0f);
	float t = tNear;
	float3 pos = ray.origin + ray.dir * t;
	float3 step = ray.dir * tStep;

	for (int i = 0; i < maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float3 rePos = pos * 0.5f + 0.5f;
		unsigned int x = floor(rePos.x) * N;
		unsigned int y = floor(rePos.y) * N;
		unsigned int z = floor(rePos.z) * N;
		unsigned int index = x * N * N + y * N + z;

		rePos -= outputVF[index] * 0.05f;

		float sample = tex3D<float>(volumeTex, rePos.x, rePos.y, rePos.z);
		float4 color = tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);
		color.w *= density;
		
		color.x *= color.w;
		color.y *= color.w;
		color.z *= color.w;

		sum += color * (1.0f - sum.w);

		if (sum.w > opacityThreshold) break;

		t += tStep;
		if (t > tFar) break;

		pos += step;
	}
	
	result[index] = make_float3(sum.x, sum.y, sum.z);
	//if (c_invViewMatrix.m[0].x == 1.0f) result[index] = make_float3(0.0f, 0.0f, 1.0f);
	return;
}

extern "C" void launch_pbo_kernel(float3* result, unsigned int width, unsigned int height,
	float density, float transferOffset, float transferScale, unsigned int N, 
	float3* outputVF)
{
	int tx = 8;
	int ty = 8;

	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);

	render_3D_texture << <blocks, threads >> > (result, width, height, volumeTexObject, transferTexObject, 
		density, transferOffset, transferScale, N, outputVF);

	checkCudaError("pbo kernel failed!");
}

__device__ unsigned int Index(int x, int y, int z, int N)
{
	return x * N * N + y * N + z;
}

__device__ float3 advect(float3* input, unsigned int N, unsigned int x, unsigned int y, unsigned int z, float time)
{
	
	unsigned int index = Index(x, y, z, N);
	int x0 = x - input[index].x;
	int y0 = y - input[index].y;
	int z0 = z - input[index].z;

	x0 = min(max(x0, 0), N - 1);
	y0 = min(max(y0, 0), N - 1);
	z0 = min(max(z0, 0), N - 1);

	return input[Index(x0, y0, z0, N)];
}

__global__ void update_vector_field(float3* pos, unsigned int N, unsigned int currentPickedIndex,
	float3 previewVect, cudaTextureObject_t transferTex, float time, 
	float3* inputVF, float3* outputVF)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= N || y >= N || z >= N) return;

	float u = x / (float)N;
	float v = y / (float)N;
	float w = z / (float)N;

	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	w = w * 2.0f - 1.0f;

	unsigned int index = x * N * N + y * N + z;
	//float len = length(inputVF[index]);
		
	//float4 color = tex1D<float4>(transferTex, len-0.1f);

	// Vector Field advect

	outputVF[index] = advect(inputVF, N, x, y, z, time);

	float3 dir = normalize(outputVF[index]);
	pos[4 * index] = make_float3(u, v, w);// vert start pos
	pos[4 * index + 1] = dir * 0.5f + 0.5f;
	pos[4 * index + 2] = make_float3(u, v, w) + make_float3(0.01f) + outputVF[index] * 0.1f;	// vert end pos
	pos[4 * index + 3] = dir * 0.5f + 0.5f;

	if (index == currentPickedIndex)
	{
		pos[4 * index + 2] = make_float3(u, v, w) + make_float3(0.01f) + previewVect * 0.1f;
		pos[4 * index + 1] = make_float3(1.0f, 1.0f, 0.0f);
		pos[4 * index + 3] = make_float3(1.0f, 1.0f, 0.0f);
	}
}

extern "C" void launch_vbo_kernel(float3* pos, unsigned int N, unsigned int currentPickedIndex, 
	float3 previewVect, float time, 
	float3* inputVF, float3* outputVF)
{
	dim3 block(4, 4, 4);
	dim3 grid(N / block.x, N / block.y, N / block.z);

	update_vector_field << <grid, block >> > (pos, N, currentPickedIndex, previewVect, transferTexObject, time,
		inputVF, outputVF);

	checkCudaError("vbo kernel failed!");

	cudaThreadSynchronize();
}

extern "C" void freeCudaBuffers()
{
	cudaDestroyTextureObject(volumeTexObject);
	checkCudaError("Destroy texture object failed!");
	cudaFreeArray(d_volumeArray);
	checkCudaError("Free volume array failed!");
	cudaDestroyTextureObject(transferTexObject);
	checkCudaError("Destroy texture object failed!");
	cudaFreeArray(d_transferFuncArray);
	checkCudaError("Free volume array failed!");
}
