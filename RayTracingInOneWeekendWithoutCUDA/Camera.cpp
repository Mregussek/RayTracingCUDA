
#include "Camera.h"


void Camera::initialize(CameraSpecification _specs) {
    specs = _specs;
    horizontal = { specs.width, 0.f, 0.f };
    vertical = { 0.f, specs.height, 0.f };
    lowerLeftCorner = calculateLowerLeftCorner();

}

point3 Camera::origin() const {
    return specs.origin;
}

vector3 Camera::calculateRayDirection(f32 u, f32 v) const {
    return lowerLeftCorner + u * horizontal + v * vertical - specs.origin;
}

point3 Camera::calculateLowerLeftCorner() const {
    return specs.origin - horizontal / 2.f - vertical / 2.f + point3(0.f, 0.f, -specs.focalLength);
}

