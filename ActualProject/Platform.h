
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


RTX_DEVICE f32 generateRandom(u32 seed) {
	curandState_t state;
	curand_init(seed, 0, 0, &state);
	const f32 randomNumber = curand(&state) % 100;
	return randomNumber;
}


RTX_DEVICE f32 generateRandomInRange(u32 seed, f32 min, f32 max) {
	return min + (max - min) * generateRandom(seed);
}


RTX_DEVICE f32 returnZero(u32 seed) {
	return 0.f;
}


RTX_DEVICE b8 epsilonEqual(f32 x) {
	return fabs(x) < (f32)RTX_EPS ? RTX_TRUE : RTX_FALSE;
}


#endif
