
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

    void initialize(CameraSpecification _specs);

    point3 origin() const;

    /*
    * @brief Calculates ray direction vec3 at the output scene. Needs param that are in <0, 1> range, where
    * bottom left corner is (0, 0) and top right corner is (1, 1).
    * @param u value at x-axis, where ray should point to
    * @param v value at y-axis, where ray should point to
    * @return calculated ray direction
    */
    vector3 calculateRayDirection(f32 u, f32 v) const;

private:

    /*
    * @brief Calculates Bottom Left Corner at the image. Needs origin, horizontal, vertical and focalLength.
    * @return Calculated point3, which says, where bottom left corner is.
    */
    point3 calculateLowerLeftCorner() const;

    CameraSpecification specs;
    f32 aspectRatio{ 0.f };
    vector3 horizontal{ 0.f, 0.f, 0.f };
    vector3 vertical{ 0.f, 0.f, 0.f };
    point3 lowerLeftCorner{ 0.f, 0.f, 0.f };


};

#endif
