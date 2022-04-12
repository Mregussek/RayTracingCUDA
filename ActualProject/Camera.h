
#ifndef CAMERA_H
#define CAMERA_H


#include "defines.h"
#include "vec3.h"


struct CameraSpecification {

    f32 height{ 0.f };
    f32 width{ 0.f };
    f32 focalLength{ 1.f };
    point3 origin{ 0.f, 0.f, 0.f };

};


class Camera {
public:

    RTX_DEVICE void initialize(CameraSpecification _specs);

    RTX_DEVICE point3 origin() const;

    /*
    * @brief Calculates ray direction vec3 at the output scene. Needs param that are in <0, 1> range, where
    * bottom left corner is (0, 0) and top right corner is (1, 1).
    * @param u value at x-axis, where ray should point to
    * @param v value at y-axis, where ray should point to
    * @return calculated ray direction
    */
    RTX_DEVICE vector3 calculateRayDirection(f32 u, f32 v) const;

private:

    /*
    * @brief Calculates Bottom Left Corner at the image. Needs origin, horizontal, vertical and focalLength.
    * @return Calculated point3, which says, where bottom left corner is.
    */
    RTX_DEVICE point3 calculateLowerLeftCorner() const;

    CameraSpecification specs;
    f32 aspectRatio{ 0.f };
    vector3 horizontal{ 0.f, 0.f, 0.f };
    vector3 vertical{ 0.f, 0.f, 0.f };
    point3 lowerLeftCorner{ 0.f, 0.f, 0.f };


};


RTX_DEVICE void Camera::initialize(CameraSpecification _specs) {
    specs = _specs;
    horizontal = { specs.width, 0.f, 0.f };
    vertical = { 0.f, specs.height, 0.f };
    lowerLeftCorner = calculateLowerLeftCorner();

}


RTX_DEVICE point3 Camera::origin() const {
    return specs.origin;
}


RTX_DEVICE vector3 Camera::calculateRayDirection(f32 u, f32 v) const {
    return lowerLeftCorner + u * horizontal + v * vertical - specs.origin;
}


RTX_DEVICE point3 Camera::calculateLowerLeftCorner() const {
    return specs.origin - horizontal / 2.f - vertical / 2.f + point3(0.f, 0.f, -specs.focalLength);
}


#endif
