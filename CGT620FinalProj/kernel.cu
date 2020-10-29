
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

__device__ unsigned int Index_xyz(int x, int y, int z, unsigned int N)
{
	return x * N * N + y * N + z;
}

extern "C" void createTransferTexture()
{
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

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_transferFuncArray;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp; // wrap texture coordinates

	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&transferTexObject, &texRes, &texDescr, NULL);
	checkCudaError("Cuda create transfer texture failed!");
}

__device__ unsigned int Index_xyz(int x, int y, int z, int N)
{
	return x * N * N + y * N + z;
}

__device__ unsigned int Index_uvw(float u, float v, float w, int N)
{
	unsigned int x = floor(u * N);
	unsigned int y = floor(v * N);
	unsigned int z = floor(w * N);

	x = min(max(0, x), N - 1);
	y = min(max(0, y), N - 1);
	z = min(max(0, z), N - 1);

	return x * N * N + y * N + z;
}


__global__ void advect_texture(VolumeType* inputVolume, VolumeType* outputVolume, float3*VF,
	unsigned int volume_scale, unsigned int vf_scale)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= volume_scale || y >= volume_scale || z >= volume_scale) return;

	unsigned int volume_index = Index_xyz(x, y, z, volume_scale);

	// advection here
	float detlaTime = 0.000001f;
	float u = (float)x / volume_scale;
	float v = (float)y / volume_scale;
	float w = (float)z / volume_scale;

	float u0 = u - VF[Index_uvw(u, v, w, vf_scale)].x * detlaTime;
	float v0 = v - VF[Index_uvw(u, v, w, vf_scale)].y * detlaTime;
	float w0 = w - VF[Index_uvw(u, v, w, vf_scale)].z * detlaTime;

	outputVolume[volume_index] = inputVolume[Index_uvw(u0, v0, w0, volume_scale)];
	//outputVolume[volume_index] = unsigned char(u0 * 255);

	return;
}

extern "C" void launch_advect_kernel(VolumeType* inputVolume, VolumeType* outputVolume, float3* VF, 
	unsigned int volume_scale, unsigned int vf_scale)
{
	dim3 block(4, 4, 4);
	dim3 grid(volume_scale / block.x, volume_scale / block.y, volume_scale / block.z);

	advect_texture << <grid, block >> > (inputVolume, outputVolume, VF, volume_scale, vf_scale);

	checkCudaError("advert texture kernel failed!");

	cudaThreadSynchronize();
}

__device__ VolumeType samplerTex3D(VolumeType* volume, unsigned int volume_scale, float i, float j, float k)
{
	unsigned int x = floor(i * volume_scale);
	unsigned int y = floor(j * volume_scale);
	unsigned int z = floor(k * volume_scale);
	unsigned int index = Index_xyz(x, y, z, volume_scale);
	return volume[index];
}

__global__ void render_3D_texture(float3* result, unsigned int width, unsigned int height, cudaTextureObject_t transferTex,
	float density, float transferOffset, float transferScale, unsigned int volume_scale, VolumeType* volume)
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

		VolumeType sample = samplerTex3D(volume, volume_scale, rePos.x, rePos.y, rePos.z);
		float f_sample = sample / 255.0f;
		float4 color = tex1D<float4>(transferTex, (f_sample - transferOffset) * transferScale);
		
		// debug
		//float4 color = make_float4(rePos, 1.0f);

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
	float density, float transferOffset, float transferScale, unsigned int volume_scale, VolumeType* volume)
{
	int tx = 8;
	int ty = 8;

	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);

	render_3D_texture << <blocks, threads >> > (result, width, height, transferTexObject, 
		density, transferOffset, transferScale, volume_scale, volume);

	checkCudaError("pbo kernel failed!");
}

__global__ void update_vector_field(float3* pos, unsigned int N, unsigned int currentPickedIndex,
	float3 previewVect, cudaTextureObject_t transferTex, float time, 
	float3* inputVF, float3* outputVF, float3* user_input_VF)
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
	outputVF[index] = inputVF[index];

	// add user input after advect
	outputVF[index] += user_input_VF[index];
	// clear user input after add
	user_input_VF[index] = make_float3(0.0f);

	// update VBO
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
	float3* inputVF, float3* outputVF, float3* user_input_VF)
{
	dim3 block(4, 4, 4);
	dim3 grid(N / block.x, N / block.y, N / block.z);

	update_vector_field << <grid, block >> > (pos, N, currentPickedIndex, previewVect, transferTexObject, time,
		inputVF, outputVF, user_input_VF);

	checkCudaError("vbo kernel failed!");

	cudaThreadSynchronize();
}

extern "C" void freeCudaBuffers()
{
	cudaDestroyTextureObject(transferTexObject);
	checkCudaError("Destroy texture object failed!");
	cudaFreeArray(d_transferFuncArray);
	checkCudaError("Free volume array failed!");
}
