
#ifndef VEC3_H
#define VEC3_H


#include <cmath>
#include <iostream>
#include "defines.h"


using std::sqrt;


template<typename val = f32>
struct vec3 {

    union { f32 x, r; };
    union { f32 y, g; };
    union { f32 z, b; };

    constexpr vec3() :
        x{ 0.f },
        y{ 0.f },
        z{ 0.f }
    {}

    constexpr vec3(val _x, val _y, val _z) :
        x{ _x },
        y{ _y },
        z{ _z }
    {}

    constexpr vec3 operator-() const { return vec3(-x, -y, -z); }

    constexpr vec3& operator+=(vec3 v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    constexpr vec3& operator*=(val v) {
        x *= v;
        y *= v;
        z *= v;
        return *this;
    }

    constexpr vec3& operator/=(val v) {
        return *this *= 1 / v;
    }

    constexpr val length() const {
        return sqrt(dot(*this, *this));
    }

    constexpr static val dot(vec3<val> u, vec3<val> v) {
        return u.x * v.x + u.y * v.y + u.z * v.z;
    }

    constexpr static vec3 cross(vec3 u, vec3 v) {
        return vec3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x
        );
    }

    constexpr static inline vec3 normalize(vec3 v) {
        return v / v.length();
    }

};


using vector3 = vec3<>;
using point3 = vec3<>;
using color = vec3<>;


template<typename val = f32>
constexpr std::ostream& operator<<(std::ostream& out, vec3<val> v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

template<typename val = f32>
constexpr vec3<val> operator+(vec3<val> u, vec3<val> v) {
    return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

template<typename val = f32>
constexpr vec3<val> operator-(vec3<val> u, vec3<val> v) {
    return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

template<typename val = f32>
constexpr vec3<val> operator*(vec3<val> u, vec3<val> v) {
    return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

template<typename val = f32>
constexpr vec3<val> operator+(val t, vec3<val> v) {
    return vec3(t + v.x, t + v.y, t + v.z);
}

template<typename val = f32>
constexpr vec3<val> operator+(vec3<val> v, val t) {
    return t + v;
}

template<typename val = f32>
constexpr vec3<val> operator*(val t, vec3<val> v) {
    return vec3(t * v.x, t * v.y, t * v.z);
}

template<typename val = f32>
constexpr vec3<val> operator*(vec3<val> v, val t) {
    return t * v;
}

template<typename val = f32>
constexpr vec3<val> operator/(vec3<val> v, val t) {
    return (1 / t) * v;
}

template<typename val = f32>
val clamp(val x, val min, val max) {
    if (x < min) {
        return min;
    }
    if (x > max) {
        return max;
    }
    return x;
}

template<typename val = f32>
void writeColor(std::ostream& out, vec3<val> pixel, i32 samplesPerPixel) {
    const f32 scale{ 1.f / (f32)samplesPerPixel };
    pixel *= scale;

    out << (i32)((val)255.999f * clamp(pixel.r, 0.f, 0.999f)) << ' '
        << (i32)((val)255.999f * clamp(pixel.g, 0.f, 0.999f)) << ' '
        << (i32)((val)255.999f * clamp(pixel.b, 0.f, 0.999f)) << '\n';
}


#endif
