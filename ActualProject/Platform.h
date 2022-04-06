
#ifndef MEMORY_H
#define MEMORY_H


#include "defines.h"


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
