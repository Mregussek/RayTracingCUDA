
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
    
    constexpr static vec3 square(vec3 v) {
        return vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
    }

    constexpr static vec3 cross(vec3 u, vec3 v) {
        return vec3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x
        );
    }

    constexpr static vec3 normalize(vec3 v) {
        return v / v.length();
    }

    static vec3 random() {
        return { generateRandom(), generateRandom(), generateRandom() };
    }

    static vec3 random(val min, val max) {
        return { generateRandomInRange(min, max), generateRandomInRange(min, max), generateRandomInRange(min, max) };
    }

    static b8 nearZero(vec3 v) {
        return epsilonEqual(v.x) & epsilonEqual(v.y) & epsilonEqual(v.z);
    }

    static vec3 reflect(vec3 vector, vec3 normal) {
        return vector - 2 * dot(vector, normal) * normal;
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


#endif
