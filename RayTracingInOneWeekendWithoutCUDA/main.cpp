
#include <iostream>
#include <fstream>
#include "vec3.h"
#include "ray.h"
#include "defines.h"


static constexpr color white{ 1.f, 1.f, 1.f };
static constexpr color blue{ 0.5f, 0.7f, 1.f };


struct ImageSpecification {

    i32 width{ 0 };
    i32 height{ 0 };
    f32 aspectRatio{ 0.f };

    constexpr ImageSpecification(i32 _w, i32 _h) :
        width(_w),
        height(_h),
        aspectRatio((f32)_w / (f32)_h)
    { }

};


struct CameraSpecification {

    f32 height{ 0.f };
    f32 width{ 0.f };
    f32 focalLength{ 1.f };

    point3 origin{ 0.f, 0.f, 0.f };
    vector3 horizontal{ 0.f, 0.f, 0.f };
    vector3 vertical{ 0.f, 0.f, 0.f };
    point3 lowerLeftCorner{ 0.f, 0.f, 0.f };

    constexpr CameraSpecification(f32 _height, f32 _aspectRatio, f32 _focalLength, point3 _origin) :
        height(_height),
        width(_height * _aspectRatio),
        focalLength(_focalLength),
        origin(_origin),
        horizontal(width, 0.f, 0.f),
        vertical(0.f, height, 0.f),
        lowerLeftCorner(calculateLowerLeftCorner())
    { }

    /*
    * @brief Calculates Bottom Left Corner at the image. Needs origin, horizontal, vertical and focalLength.
    * @return Calculated point3, which says, where bottom left corner is.
    */
    constexpr point3 calculateLowerLeftCorner() {
        return origin - horizontal / 2.f - vertical / 2.f + point3(0.f, 0.f, -focalLength);
    }

    /*
    * @brief Calculates ray direction vec3 at the output scene. Needs param that are in <0, 1> range, where
    * bottom left corner is (0, 0) and top right corner is (1, 1).
    * @param u value at x-axis, where ray should point to
    * @param v value at y-axis, where ray should point to
    * @return calculated ray direction
    */
    constexpr vector3 calculateRayDirection(f32 u, f32 v) const {
        return lowerLeftCorner + u * horizontal + v * vertical - origin;
    }

};


void printRemainingScanlinesWithInfo(ImageSpecification image, i32 remaining) {
    std::cerr << "\rImage " << image.width << "x" << image.height << " Scanlines remaining : " << remaining << ' ' << std::flush;
}


f32 hitSphere(const point3& center, f32 radius, const Ray& ray) {
    const vector3 oc{ ray.origin - center };
    const f32 a{ vector3::dot(ray.direction, ray.direction) };
    const f32 b{ 2.f * vector3::dot(oc, ray.direction) };
    const f32 c{ vector3::dot(oc, oc) - radius * radius };
    const f32 delta{ b * b - 4 * a * c };
    if (delta < 0) {
        return -1.f;
    }
    else {
        return (-b - sqrt(delta)) / (2.f * a);
    }
}


/*
* @brief Function linearly blends white and blue depending on the height of the y coordinate after scaling the ray
* direction to unit length (-1.f, 1.f). Because we're looking at the y height after normalizing the vector, you'll
* notice a horizontal gradient to the color in addition to the vertical gradient.
* @param r just ray, which shall be colored
* @return colored ray
*/
color colorRay(const Ray& r) {
    const f32 hitPoint{ hitSphere(point3(0.f, 0.f, -1.f), 0.5f, r) };
    if (hitPoint > 0.f) {
        const vector3 n{ vector3::normalize(r.at(hitPoint) - vector3(0.f, 0.f, -1.f)) };
        return 0.5f * (color(n.x, n.y, n.z) + 1.f);
    }
    const vector3 unitDirection{ vector3::normalize(r.direction) };
    const f32 t{ 0.5f * (unitDirection.y + 1.f) };          // scaling t to <0.f, 1.f>
    return (1.f - t) * white + t * blue;                    // blendedValue = (1 - t) * startValue + t * endValue
                                                            // when t=0 I want white, when t=1 I want blue
}


auto main() -> i32 {
	
    constexpr ImageSpecification image{ 720, 405 };
    constexpr CameraSpecification camera{ 2.f, image.aspectRatio, 1.f, point3{ 0.f, 0.f, 0.f } };

    std::ofstream file;
    file.open("output_filename.ppm");
    file << "P3\n" << image.width << ' ' << image.height << "\n255\n";
    for (i32 j = image.height - 1; j >= 0; j--) {
        printRemainingScanlinesWithInfo(image, j);
        for (i32 i = 0; i < image.width; i++) {
            const f32 u = (f32)i / ((f32)image.width - 1.f);
            const f32 v = (f32)j / ((f32)image.height - 1.f);
            const Ray r{ camera.origin, camera.calculateRayDirection(u, v) };
            color pixel{ colorRay(r) };
            writeColor(file, pixel);
        }
    }
    file.close();
    std::cerr << "\nDone.\n";
	return 0;
}
