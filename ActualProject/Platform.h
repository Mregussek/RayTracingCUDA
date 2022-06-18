
#ifndef MEMORY_H
#define MEMORY_H


#include "defines.h"


#define CUDA_CHECK(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


#define RTX_CALL_KERNEL_AND_VALIDATE(...)		__VA_ARGS__;\
												CUDA_CHECK( cudaGetLastError() );\
												CUDA_CHECK( cudaDeviceSynchronize() )


#define CUDA_MALLOC_AND_MEMCPY(pDeviceData, hostData) \
    cudaMalloc(&pDeviceData, hostData.size() * sizeof(decltype(hostData[0])));\
    cudaMemcpy(pDeviceData, hostData.data(), hostData.size() * sizeof(decltype(hostData[0])), cudaMemcpyHostToDevice)


RTX_DEVICE f32 generateRandom(curandState* pRandState) {
	return curand_uniform(pRandState);
}


RTX_DEVICE f32 generateRandomInRange(curandState* pRandState, f32 min, f32 max) {
	return (max - min) * generateRandom(pRandState) - min;
}


RTX_DEVICE b8 epsilonEqual(f32 x) {
	return fabs(x) < (f32)RTX_EPS ? RTX_TRUE : RTX_FALSE;
}


#endif
