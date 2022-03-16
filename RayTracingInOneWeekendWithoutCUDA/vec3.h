
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

    vec3() :
        x{ 0.f },
        y{ 0.f },
        z{ 0.f }
    {}
    vec3(val _x, val _y, val _z) :
        x{ _x },
        y{ _y},
        z{ _z }
    {}

    vec3 operator-() const { return vec3(-x, -y, -z); }

    vec3& operator+=(vec3 v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    vec3& operator*=(val v) {
        x *= v;
        y *= v;
        z *= v;
        return *this;
    }

    vec3& operator/=(val v) {
        return *this *= 1 / v;
    }

    val length() const {
        return sqrt(length_squared());
    }

    val length_squared() const {
        return x * x + y * y + z * z;
    }

};


using vector3 = vec3<>;
using point3 = vec3<>;
using color = vec3<>;


template<typename val = f32>
inline std::ostream& operator<<(std::ostream& out, vec3<val> v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

template<typename val = f32>
inline vec3<val> operator+(vec3<val> u, vec3<val> v) {
    return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

template<typename val = f32>
inline vec3<val> operator-(vec3<val> u, vec3<val> v) {
    return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

template<typename val = f32>
inline vec3<val> operator*(vec3<val> u, vec3<val> v) {
    return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

template<typename val = f32>
inline vec3<val> operator*(double t, vec3<val> v) {
    return vec3(t * v.x, t * v.y, t * v.z);
}

template<typename val = f32>
inline vec3<val> operator*(vec3<val> v, double t) {
    return t * v;
}

template<typename val = f32>
inline vec3<val> operator/(vec3<val> v, double t) {
    return (1 / t) * v;
}

template<typename val = f32>
inline double dot(vec3<val> u, vec3<val> v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

template<typename val = f32>
inline vec3<val> cross(vec3<val> u, vec3<val> v) {
    return vec3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

template<typename val = f32>
inline vec3<val> unit_vector(vec3<val> v) {
    return v / v.length();
}

template<typename val = f32>
void writeColor(std::ostream& out, vec3<val> pixel) {
    out << (i32)((val)255.999f * pixel.r) << ' '
        << (i32)((val)255.999f * pixel.g) << ' '
        << (i32)((val)255.999f * pixel.b) << '\n';
}


#endif
