
#ifndef DEFINES_H
#define DEFINES_H


#define ENABLE_GAMMA_CORRECTION 1
#define USE_GPU_CUDA_COMPUTING 1

#if USE_GPU_CUDA_COMPUTING
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#define RTX_HOST __host__
#define RTX_DEVICE __device__
#define RTX_GLOBAL __global__
#else
#define RTX_HOST
#define RTX_DEVICE
#define RTX_GLOBAL
#endif

#define RTX_TRUE 1
#define RTX_FALSE 0
#define RTX_EPS 1e-8


#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <vector>


using i8 = int_fast8_t;
using i16 = int_least16_t;
using i32 = int_fast32_t;
using i64 = int_fast64_t;

using u8 = uint_fast8_t;
using u16 = uint_least16_t;
using u32 = uint_fast32_t;
using u64 = uint_fast64_t;

using f32 = float;
using f64 = double;

using b8 = i8;
using b32 = i32;

using radius = f32;


#endif
