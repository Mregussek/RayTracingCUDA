
#include "vec3.h"
#include "Platform.h"


vec3::vec3() :
    x{ 0.f },
    y{ 0.f },
    z{ 0.f }
{}

vec3::vec3(f32 _x, f32 _y, f32 _z) :
    x{ _x },
    y{ _y },
    z{ _z }
{}

vec3 vec3::operator-() const { return vec3(-x, -y, -z); }

vec3& vec3::operator+=(vec3 v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

vec3& vec3::operator*=(f32 v) {
    x *= v;
    y *= v;
    z *= v;
    return *this;
}

vec3& vec3::operator/=(f32 v) {
    return *this *= 1 / v;
}

f32 vec3::length() const {
    return sqrt(dot(*this, *this));
}

f32 vec3::dot(vec3 u, vec3 v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

vec3 vec3::square(vec3 v) {
    return vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

vec3 vec3::cross(vec3 u, vec3 v) {
    return vec3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

vec3 vec3::normalize(vec3 v) {
    return v / v.length();
}

vec3 vec3::random() {
    return { generateRandom(), generateRandom(), generateRandom() };
}

vec3 vec3::random(f32 min, f32 max) {
    return { generateRandomInRange(min, max), generateRandomInRange(min, max), generateRandomInRange(min, max) };
}

b8 vec3::nearZero(vec3 v) {
    return epsilonEqual(v.x) & epsilonEqual(v.y) & epsilonEqual(v.z);
}

vec3 vec3::reflect(vec3 vector, vec3 normal) {
    return vector - 2 * dot(vector, normal) * normal;
}

// JUST HEADER METHODS

std::ostream& operator<<(std::ostream& out, vec3 v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}


vec3 operator+(vec3 u, vec3 v) {
    return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}


vec3 operator-(vec3 u, vec3 v) {
    return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}


vec3 operator*(vec3 u, vec3 v) {
    return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}


vec3 operator/(vec3 u, vec3 v) {
    return vec3(u.x / v.x, u.y / v.y, u.z / v.z);
}


vec3 operator+(f32 t, vec3 v) {
    return vec3(t + v.x, t + v.y, t + v.z);
}


vec3 operator-(f32 t, vec3 v) {
    return vec3(t - v.x, t - v.y, t - v.z);
}


vec3 operator*(f32 t, vec3 v) {
    return vec3(t * v.x, t * v.y, t * v.z);
}


vec3 operator/(f32 t, vec3 v) {
    return vec3((1.f / t) * v.x, (1.f / t) * v.y, (1.f / t) * v.z);
}


vec3 operator+(vec3 v, f32 t) {
    return t + v;
}


vec3 operator-(vec3 v, f32 t) {
    return vec3(v.x - t, v.y - t, v.z - t);
}


vec3 operator*(vec3 v, f32 t) {
    return t * v;
}


vec3 operator/(vec3 v, f32 t) {
    return (1.f / t) * v;
}