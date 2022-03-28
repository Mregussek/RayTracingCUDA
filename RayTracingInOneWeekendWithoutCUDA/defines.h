
#ifndef DEFINES_H
#define DEFINES_H


#define ENABLE_GAMMA_CORRECTION 1

#define RTX_TRUE 1
#define RTX_FALSE 0
#define RTX_EPS 1e-8

#include <cstdint>


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


template<typename val = f32>
val generateRandom() {
    static std::uniform_real_distribution<val> distribution((val)0.0, (val)1.0);
    static std::mt19937 generator;
    return (val)distribution(generator);
}

template<typename val = f32>
val generateRandomInRange(val min, val max) {
    return min + (max - min) * generateRandom();
}

template<typename val = f32>
val returnZero() {
    return 0.f;
}

template<typename val = f32>
b8 epsilonEqual(val x) {
    constexpr val epsilon{ (val)RTX_EPS };
    return fabs(x) < epsilon ? RTX_TRUE : RTX_FALSE;
}


#endif
