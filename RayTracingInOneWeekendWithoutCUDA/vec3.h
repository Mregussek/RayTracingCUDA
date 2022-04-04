
#ifndef VEC3_H
#define VEC3_H


#include <cmath>
#include <iostream>
#include "defines.h"


struct vec3 {

    union { f32 x, r; };
    union { f32 y, g; };
    union { f32 z, b; };

    vec3();
    vec3(f32 _x, f32 _y, f32 _z);

    vec3 operator-() const;
    vec3& operator+=(vec3 v);
    vec3& operator*=(f32 v);
    vec3& operator/=(f32 v);

    f32 length() const;
    static f32 dot(vec3 u, vec3 v);
    static vec3 square(vec3 v);
    static vec3 cross(vec3 u, vec3 v);
    static vec3 normalize(vec3 v);
    static vec3 random();
    static vec3 random(f32 min, f32 max);
    static b8 nearZero(vec3 v);
    static vec3 reflect(vec3 vector, vec3 normal);

};


using vector3 = vec3;
using point3 = vec3;
using color = vec3;


std::ostream& operator<<(std::ostream& out, vec3 v);

vec3 operator+(vec3 u, vec3 v);
vec3 operator-(vec3 u, vec3 v);
vec3 operator*(vec3 u, vec3 v);
vec3 operator/(vec3 u, vec3 v);

vec3 operator+(f32 t, vec3 v);
vec3 operator-(f32 t, vec3 v);
vec3 operator*(f32 t, vec3 v);
vec3 operator/(f32 t, vec3 v);

vec3 operator+(vec3 v, f32 t);
vec3 operator-(vec3 v, f32 t);
vec3 operator*(vec3 t, f32 v);
vec3 operator/(vec3 v, f32 t);


#endif
