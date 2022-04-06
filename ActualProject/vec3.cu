
#include "vec3.h"

/*
RTX_HOST RTX_DEVICE vec3::vec3() :
    x{ 0.f },
    y{ 0.f },
    z{ 0.f }
{}
RTX_HOST RTX_DEVICE vec3::vec3(f32 _x, f32 _y, f32 _z) :
    x{ _x },
    y{ _y },
    z{ _z }
{}

RTX_HOST RTX_DEVICE f32 vec3::length() const {
    return sqrt(dot(*this, *this));
}
RTX_HOST RTX_DEVICE f32 vec3::dot(vec3 u, vec3 v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}
RTX_HOST RTX_DEVICE vec3 vec3::square(vec3 v) {
    return vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}
RTX_HOST RTX_DEVICE vec3 vec3::cross(vec3 u, vec3 v) {
    return vec3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}
RTX_HOST RTX_DEVICE vec3 vec3::normalize(vec3 v) {
    return v / v.length();
}
RTX_HOST RTX_DEVICE vec3 vec3::random() {
    return { generateRandom(), generateRandom(), generateRandom() };
}
RTX_HOST RTX_DEVICE vec3 vec3::random(f32 min, f32 max) {
    return { generateRandomInRange(min, max), generateRandomInRange(min, max), generateRandomInRange(min, max) };
}
RTX_HOST RTX_DEVICE b8 vec3::nearZero(vec3 v) {
    return epsilonEqual(v.x) & epsilonEqual(v.y) & epsilonEqual(v.z);
}
RTX_HOST RTX_DEVICE vec3 vec3::reflect(vec3 vector, vec3 normal) {
    return vector - 2 * dot(vector, normal) * normal;
}


RTX_HOST RTX_DEVICE vec3 operator+(vec3 left, vec3 right) {
    return left.add(right);
}


RTX_HOST RTX_DEVICE vec3 operator-(vec3 left, vec3 right) {
    return left.subtract(right);
}


RTX_HOST RTX_DEVICE vec3 operator*(vec3 left, vec3 right) {
    return left.multiply(right);
}


RTX_HOST RTX_DEVICE vec3 operator/(vec3 left, vec3 right) {
    return left.divide(right);
}


RTX_HOST RTX_DEVICE vec3 operator+(vec3 left, f32 right) {
    return left.add(right);
}


RTX_HOST RTX_DEVICE vec3 operator-(vec3 left, f32 right) {
    return left.subtract(right);
}


RTX_HOST RTX_DEVICE vec3 operator*(vec3 left, f32 right) {
    return left.multiply(right);
}


RTX_HOST RTX_DEVICE vec3 operator/(vec3 left, f32 right) {
    return left.divide(right);
}


RTX_HOST RTX_DEVICE vec3 operator+(f32 left, vec3 right) {
    return right.add(left);
}


RTX_HOST RTX_DEVICE vec3 operator-(f32 left, vec3 right) {
    return right.subtract(left);
}


RTX_HOST RTX_DEVICE vec3 operator*(f32 left, vec3 right) {
    return right.multiply(left);
}


RTX_HOST RTX_DEVICE vec3 operator/(f32 left, vec3 right) {
    return right.divide(left);
}
*/