
#ifndef MEMORY_H
#define MEMORY_H


#include "defines.h"


template<typename T> void* allocate(i32 elemsCount) {
	if constexpr (USE_GPU_CUDA_COMPUTING) {
		return nullptr;
	}
	else {
		return (void*)new T[elemsCount];
	}
}


template<typename T> void free(T* pPointer) {
	if constexpr (USE_GPU_CUDA_COMPUTING) {
		return;
	}
	else {
		if (pPointer) {
			delete[] pPointer;
		}
	}
}


f32 generateRandom();
f32 generateRandomInRange(f32 min, f32 max);
f32 returnZero();
b8 epsilonEqual(f32 x);


#endif
