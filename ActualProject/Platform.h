
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


f32 generateRandom() {
	static std::uniform_real_distribution<f32> distribution(0.0f, 1.0f);
	static std::mt19937 generator;
	return distribution(generator);
}


f32 generateRandomInRange(f32 min, f32 max) {
	return min + (max - min) * generateRandom();
}


f32 returnZero() {
	return 0.f;
}


b8 epsilonEqual(f32 x) {
	constexpr f32 epsilon{ (f32)RTX_EPS };
	return fabs(x) < epsilon ? RTX_TRUE : RTX_FALSE;
}


#endif
