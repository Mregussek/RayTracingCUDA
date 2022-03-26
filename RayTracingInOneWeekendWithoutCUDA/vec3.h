
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

    constexpr static inline val dot(vec3<val> u, vec3<val> v) {
        return u.x * v.x + u.y * v.y + u.z * v.z;
    }

    constexpr static inline vec3 cross(vec3 u, vec3 v) {
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
constexpr inline std::ostream& operator<<(std::ostream& out, vec3<val> v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

template<typename val = f32>
constexpr inline vec3<val> operator+(vec3<val> u, vec3<val> v) {
    return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

template<typename val = f32>
constexpr inline vec3<val> operator-(vec3<val> u, vec3<val> v) {
    return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

template<typename val = f32>
constexpr inline vec3<val> operator*(vec3<val> u, vec3<val> v) {
    return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

template<typename val = f32>
constexpr inline vec3<val> operator+(val t, vec3<val> v) {
    return vec3(t + v.x, t + v.y, t + v.z);
}

template<typename val = f32>
constexpr inline vec3<val> operator+(vec3<val> v, val t) {
    return t + v;
}

template<typename val = f32>
constexpr inline vec3<val> operator*(val t, vec3<val> v) {
    return vec3(t * v.x, t * v.y, t * v.z);
}

template<typename val = f32>
constexpr inline vec3<val> operator*(vec3<val> v, val t) {
    return t * v;
}

template<typename val = f32>
constexpr inline vec3<val> operator/(vec3<val> v, val t) {
    return (1 / t) * v;
}

template<typename val = f32>
void writeColor(std::ostream& out, vec3<val> pixel) {
    out << (i32)((val)255.999f * pixel.r) << ' '
        << (i32)((val)255.999f * pixel.g) << ' '
        << (i32)((val)255.999f * pixel.b) << '\n';
}


#endif
