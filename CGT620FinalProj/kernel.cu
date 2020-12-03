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

#define PI 3.1415926

cudaArray* d_transferFuncArray;
cudaTextureObject_t transferTexObject; // Transfer texture Object

__constant__ unsigned int c_VF_data_scale = 300;
__constant__ unsigned int c_tex_width = 512;
__constant__ unsigned int c_tex_height = 512;
unsigned int h_VF_data_scale = 300;
unsigned int h_tex_width = 512;
unsigned int h_tex_height = 512;

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

__device__ float3 Clamp_01(float3 p)
{
	float3 result;
	result.x = min(max(0.0f, p.x), 1.0f);
	result.y = min(max(0.0f, p.y), 1.0f);
	result.z = min(max(0.0f, p.z), 1.0f);

	return result;
}

__device__ unsigned int Index_xyz(int x, int y, int z, int N)
{
	if (x < 0) x += N;
	else if (x >= N) x -= N;
	if (y < 0) y += N;
	else if (y >= N) y -= N;
	if (z < 0) z += N;
	else if (z >= N) z -= N;

	return x * N * N + y * N + z;
}

// only use for render! floor is not accurate 
__device__ unsigned int Index_uvw(float u, float v, float w, int N)
{
	unsigned int x = floor(u * N);
	unsigned int y = floor(v * N);
	unsigned int z = floor(w * N);

	return Index_xyz(x, y, z, N);
}

__global__ void fill_volume(float4* VF)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);
	
	// (0, 1)
	float u = (float)x / c_VF_data_scale;
	float v = (float)y / c_VF_data_scale;
	float w = (float)z / c_VF_data_scale;

	// (-1, 1)
	float u0 = u * 2.0f - 1.0f;
	float v0 = v * 2.0f - 1.0f;
	float w0 = w * 2.0f - 1.0f;

	// velocity
	VF[index].x = u0 * 0.01f;
	VF[index].y = v0 * 0.01f;
	VF[index].z = 0.0f;

	// Power
	if (u > 0.25f && u < 0.75f && v > 0.25f && v < 0.75f && w > 0.25f && w < 0.75f)
	//if (length(make_float3(u0,v0,w0) - make_float3(0.0f, -0.5f, 0.0f))< 0.5f ||
	//	length(make_float3(u0, v0, w0) - make_float3(0.0f, 0.5f, 0.0f)) < 0.5f)
	//if (u > 0.25f && u < 0.75f)
	{
		VF[index].w = 1.0f;
	}
		
}

extern "C" void launch_init_VF_kernel(float4* VF)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	fill_volume << <grid, block >> > (VF);

	checkCudaError("Init VF kernel failed!");

	cudaThreadSynchronize();
}

__global__ void clear_velocity(float4* VF_0, float4* VF_1)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	VF_0[index].x = 0.0f;
	VF_0[index].y = 0.0f;
	VF_0[index].z = 0.0f;

	VF_1[index].x = 0.0f;
	VF_1[index].y = 0.0f;
	VF_1[index].z = 0.0f;
}

extern "C" void launch_clear_velocity_kernel(float4* VF_0, float4* VF_1)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	clear_velocity << <grid, block >> > (VF_0, VF_1);

	checkCudaError("Clear velocity kernel failed!");

	cudaThreadSynchronize();
}

__global__ void clear_power(float4* VF_0, float4* VF_1)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	VF_0[index].w = 0.0f;
	VF_1[index].w = 0.0f;
}

extern "C" void launch_clear_power_kernel(float4* VF_0, float4* VF_1)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	clear_power << <grid, block >> > (VF_0, VF_1);

	checkCudaError("Clear power kernel failed!");

	cudaThreadSynchronize();
}

__global__ void set_velocity(float4* VF_0, float4* VF_1,
	float3 pos, float divergence, float3 curl, float3 wind, float radious)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	// (0, 1)
	float u = (float)x / c_VF_data_scale;
	float v = (float)y / c_VF_data_scale;
	float w = (float)z / c_VF_data_scale;

	float3 uvw = make_float3(u, v, w);
	float3 p = uvw - pos;

	if (length(p) < radious)
	{
		// divergence
		float3 div = (uvw - pos) * divergence * 0.05f;
		VF_0[index].x += div.x;
		VF_0[index].y += div.y;
		VF_0[index].z += div.z;
		VF_1[index].x += div.x;
		VF_1[index].y += div.y;
		VF_1[index].z += div.z;

		// curl
		float3 c = cross(p, curl) * 0.05f;
		VF_0[index].x += c.x;
		VF_0[index].y += c.y;
		VF_0[index].z += c.z;
		VF_1[index].x += c.x;
		VF_1[index].y += c.y;
		VF_1[index].z += c.z;

		//wind
		float3 uni_wind = wind * (radious - length(p)) * 0.2f;
		VF_0[index].x += uni_wind.x;
		VF_0[index].y += uni_wind.y;
		VF_0[index].z += uni_wind.z;
		VF_1[index].x += uni_wind.x;
		VF_1[index].y += uni_wind.y;
		VF_1[index].z += uni_wind.z;
	}
}

extern "C" void launch_set_velocity_kernel(float4* VF_0, float4* VF_1,
	float3 pos, float divergence, float3 curl, float3 wind, float radious)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	set_velocity << <grid, block >> > (VF_0, VF_1, pos, divergence, curl, wind, radious);

	checkCudaError("Set velocity kernel failed!");

	cudaThreadSynchronize();
}

__global__ void set_power(float4* VF_0, float4* VF_1,
	float3 pos, float radious, float density)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	// (0, 1)
	float u = (float)x / c_VF_data_scale;
	float v = (float)y / c_VF_data_scale;
	float w = (float)z / c_VF_data_scale;

	float3 uvw = make_float3(u, v, w);

	if (length(uvw - pos) < radious)
	{
		VF_0[index].w += density;
		VF_1[index].w += density;
	}
}

extern "C" void launch_set_power_kernel(float4* VF_0, float4* VF_1,
	float3 pos, float radious, float density)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	set_power << <grid, block >> > (VF_0, VF_1, pos, radious, density);

	checkCudaError("Gradient kernel failed!");

	cudaThreadSynchronize();
}

__global__ void calculte_gradient(float4* VF, float3* gradient)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	// Gx of VF[index].w
	gradient[index].x = 0.5f * (VF[Index_xyz(x + 1, y, z, c_VF_data_scale)].w - VF[Index_xyz(x - 1, y, z, c_VF_data_scale)].w);
	// Gy of VF[index].w
	gradient[index].y = 0.5f * (VF[Index_xyz(x, y + 1, z, c_VF_data_scale)].w - VF[Index_xyz(x, y - 1, z, c_VF_data_scale)].w);
	// Gz of of VF[index].w
	gradient[index].z = 0.5f * (VF[Index_xyz(x, y, z + 1, c_VF_data_scale)].w - VF[Index_xyz(x, y, z - 1, c_VF_data_scale)].w);
}

extern "C" void launch_gradient_kernel(float4* VF, float3* gradient)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	calculte_gradient << <grid, block >> > (VF, gradient);

	checkCudaError("Gradient kernel failed!");

	cudaThreadSynchronize();
}

__global__ void calculte_divergence(float3* gradient, float* divergence)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	float3 div = make_float3(0.0f);

	// Dx
	div.x = 0.5f * (gradient[Index_xyz(x + 1, y, z, c_VF_data_scale)].x - gradient[Index_xyz(x - 1, y, z, c_VF_data_scale)].x);
	// Dy
	div.y = 0.5f * (gradient[Index_xyz(x, y + 1, z, c_VF_data_scale)].y - gradient[Index_xyz(x, y - 1, z, c_VF_data_scale)].y);
	// Dz
	div.z = 0.5f * (gradient[Index_xyz(x, y, z + 1, c_VF_data_scale)].z - gradient[Index_xyz(x, y, z - 1, c_VF_data_scale)].z);

	divergence[index] = div.x + div.y + div.z;
}

extern "C" void launch_divergence_kernel(float3* gradient, float* divergence)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	calculte_divergence << <grid, block >> > (gradient, divergence);

	checkCudaError("Divergence kernel failed!");

	cudaThreadSynchronize();
}

__global__ void diffusion(float4* input_VF, float4* output_VF, float* divergence)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	// velocity
	output_VF[index].x = input_VF[index].x;
	output_VF[index].y = input_VF[index].y;
	output_VF[index].z = input_VF[index].z;
	
	float dt = 0.5f;

	// power
	output_VF[index].w = input_VF[index].w + divergence[index]* dt;
	//output_VF[index].w = input_VF[index].w;
}

extern "C" void launch_diffusion_kernel(float4* input_VF, float4* output_VF, float* divergence)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	diffusion << <grid, block >> > (input_VF, output_VF, divergence);

	checkCudaError("Diffusion kernel failed!");

	cudaThreadSynchronize();
}

__global__ void advect(float4* input_VF, float4* output_VF)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	float dt = 0.05f;

	unsigned int x0, y0, z0;
	// at least move one step when dt is too small
	if (output_VF[index].x < 0) x0 = ceil(x - output_VF[index].x * dt);
	else x0 = floor(x - output_VF[index].x * dt);
	if (output_VF[index].y < 0) y0 = ceil(y - output_VF[index].y * dt);
	else y0 = floor(y - output_VF[index].y * dt);
	if (output_VF[index].z < 0) z0 = ceil(z - output_VF[index].z * dt);
	else z0 = floor(z - output_VF[index].z * dt);

	// power
	output_VF[index].w = output_VF[index].w*0.5f + input_VF[Index_xyz(x0, y0, z0, c_VF_data_scale)].w*0.5f;
}

__global__ void swap(float4* input_VF, float4* output_VF)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= c_VF_data_scale || y >= c_VF_data_scale || z >= c_VF_data_scale) return;

	unsigned int index = Index_xyz(x, y, z, c_VF_data_scale);

	// power
	output_VF[index].w = input_VF[index].w;
}

extern "C" void launch_advect_kernel(float4* input_VF, float4* output_VF)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	advect << <grid, block >> > (input_VF, output_VF);

	checkCudaError("Advect kernel failed!");

	cudaThreadSynchronize();

	swap << <grid, block >> > (output_VF, input_VF);

	checkCudaError("Swap kernel failed!");

	cudaThreadSynchronize();
}

// https://www.willusher.io/webgl/2019/01/13/volume-rendering-with-webgl
__global__ void render(float4* VF, float3* gradient, float* divergence,
	float3* cuda_diffusion_result,
	float3* cuda_velocity_result,
	float3* cuda_gradient_result,
	float3* cuda_divergence_result,
	float density, float transferOffset, float transferScale, cudaTextureObject_t transferTex)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= c_tex_width) || (j >= c_tex_height))
		return;
	int index = j * c_tex_width + i;

	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	float u = (i / (float)c_tex_width) * 2.0f - 1.0f;
	float v = (j / (float)c_tex_height) * 2.0f - 1.0f;

	// calculate camera ray in world space
	Ray ray;
	ray.origin = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	ray.dir = normalize(make_float3(u, -v, -2.0f));
	ray.dir = mul(c_invViewMatrix, ray.dir);

	// background color
	cuda_diffusion_result[index] = make_float3(0.0f);
	cuda_velocity_result[index] = make_float3(0.0f);
	cuda_gradient_result[index] = make_float3(0.0f);
	cuda_divergence_result[index] = make_float3(0.0f);

	// intersect with AABB
	float tNear, tFar;
	bool hit = intersectAABB(ray, boxMin, boxMax, &tNear, &tFar);

	if (!hit) return;
	if (tNear > tFar) return;

	// ray marching
	float4 diffusion_sum = make_float4(0.0f);
	float4 velocity_sum = make_float4(0.0f);
	float4 gradient_sum = make_float4(0.0f);
	float4 divergence_sum = make_float4(0.0f);

	// ray marching parameters
	const float opacityThreshold = 0.95f;

	float dt = 0.05f;
	float INF = 0.01f;
	
	float3 pos = ray.origin + tNear * ray.dir;
	for (float t = tNear; t < tFar; t += dt)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float3 rePos = pos * 0.5f + 0.5f;

		unsigned int sampleIndex = Index_uvw(rePos.x, rePos.y, rePos.z, c_VF_data_scale);
		float4 sample = VF[sampleIndex];

		// diffusion
		float4 diffusion_color = tex1D<float4>(transferTex, (abs(sample.w) - transferOffset) * transferScale);
		//diffusion_color = make_float4(rePos, sample.w);
		diffusion_color.w = sample.w * density;

		diffusion_sum.x +=  (1.0f - diffusion_sum.w) * diffusion_color.w * diffusion_color.x;
		diffusion_sum.y +=  (1.0f - diffusion_sum.w) * diffusion_color.w * diffusion_color.y;
		diffusion_sum.z +=  (1.0f - diffusion_sum.w) * diffusion_color.w * diffusion_color.z;
		diffusion_sum.w += (1.0f - diffusion_sum.w) * diffusion_color.w;

		// velocity
		float3 velocity_dir = normalize(make_float3(sample.x, sample.y, sample.z)) * 0.5f + 0.5f;
		float4 velocity_color = make_float4(velocity_dir, 1.0f);

		velocity_sum.x += (1.0f - velocity_sum.w) * velocity_color.w * velocity_color.x;
		velocity_sum.y += (1.0f - velocity_sum.w) * velocity_color.w * velocity_color.y;
		velocity_sum.z += (1.0f - velocity_sum.w) * velocity_color.w * velocity_color.z;
		velocity_sum.w += (1.0f - velocity_sum.w) * velocity_color.w;

		// gradient
		float4 gradient_color = make_float4(gradient[sampleIndex]*100.0f, density);

		gradient_sum.x += (1.0f - gradient_sum.w) * gradient_color.w * gradient_color.x;
		gradient_sum.y += (1.0f - gradient_sum.w) * gradient_color.w * gradient_color.y;
		gradient_sum.z += (1.0f - gradient_sum.w) * gradient_color.w * gradient_color.z;
		gradient_sum.w += (1.0f - gradient_sum.w) * gradient_color.w;

		// divergence
		float4 divergence_color = make_float4(divergence[sampleIndex] * 10.0f);

		divergence_sum.x += (1.0f - divergence_sum.w) * divergence_color.w * divergence_color.x;
		divergence_sum.y += (1.0f - divergence_sum.w) * divergence_color.w * divergence_color.y;
		divergence_sum.z += (1.0f - divergence_sum.w) * divergence_color.w * divergence_color.z;
		divergence_sum.w += (1.0f - divergence_sum.w) * divergence_color.w;

		if (diffusion_sum.w > opacityThreshold && velocity_sum.w > opacityThreshold &&
			gradient_sum.w > opacityThreshold && divergence_sum.w > opacityThreshold) break;

		pos += ray.dir * dt;
	}

	cuda_diffusion_result[index] = Clamp_01(make_float3(diffusion_sum.x, diffusion_sum.y, diffusion_sum.z));
	cuda_velocity_result[index] = Clamp_01(make_float3(velocity_sum.x, velocity_sum.y, velocity_sum.z));
	cuda_gradient_result[index] = Clamp_01(make_float3(gradient_sum.x, gradient_sum.y, gradient_sum.z));
	cuda_divergence_result[index] = Clamp_01(make_float3(divergence_sum.x, divergence_sum.y, divergence_sum.z));
	return;
}

extern "C" void launch_display_kernel(float4* VF, float3* gradient, float* divergence,
	float3* cuda_diffusion_result,
	float3* cuda_velocity_result,
	float3* cuda_gradient_result,
	float3* cuda_divergence_result,
	float density, float transferOffset, float transferScale)
{
	int tx = 8;
	int ty = 8;

	dim3 blocks(h_tex_width / tx + 1, h_tex_height / ty + 1);
	dim3 threads(tx, ty);

	render << <blocks, threads >> > (VF, gradient, divergence, 
		cuda_diffusion_result, 
		cuda_velocity_result,
		cuda_gradient_result, 
		cuda_divergence_result, 
		density, transferOffset, transferScale, transferTexObject);

	checkCudaError("Display kernel failed!");

	cudaThreadSynchronize();
}

__global__ void sampling_VF(float3* cuda_vbo_result, float4* VF, unsigned int vf_view_scale)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= vf_view_scale || y >= vf_view_scale || z >= vf_view_scale) return;

	float u = (x+0.5) / (float)vf_view_scale;
	float v = (y+0.5) / (float)vf_view_scale;
	float w = (z+0.5) / (float)vf_view_scale;

	// sample from VF
	unsigned int VF_index = Index_uvw(u, v, w, c_VF_data_scale);
	float3 velocity = make_float3(VF[VF_index].x, VF[VF_index].y, VF[VF_index].z);
	float3 color = normalize(velocity) * 0.5f + 0.5f;

	// offset
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	w = w * 2.0f - 1.0f;

	unsigned int vbo_index = Index_xyz(x, y, z, vf_view_scale);


	// fill VBO
	cuda_vbo_result[4 * vbo_index] = make_float3(u, v, w);// vert start cuda_vbo_result
	cuda_vbo_result[4 * vbo_index + 1] = color;
	cuda_vbo_result[4 * vbo_index + 2] = make_float3(u, v, w) + velocity * 5.0f;	// vert end cuda_vbo_result
	cuda_vbo_result[4 * vbo_index + 3] = color;
}

extern "C" void launch_vbo_kernel(float3* cuda_vbo_result, float4* VF, unsigned int vf_view_scale)
{
	dim3 block(8, 8, 8);
	dim3 grid(h_VF_data_scale / block.x, h_VF_data_scale / block.y, h_VF_data_scale / block.z);

	sampling_VF << <grid, block >> > (cuda_vbo_result, VF, vf_view_scale);

	checkCudaError("VBO kernel failed!");

	cudaThreadSynchronize();
}

extern "C" void freeCudaTextureBuffers()
{
	cudaDestroyTextureObject(transferTexObject);
	checkCudaError("Destroy texture object failed!");
	cudaFreeArray(d_transferFuncArray);
	checkCudaError("Free volume array failed!");
}
