
#include <iostream>
#include <fstream>
#include <random>
#include <limits>
#include "vec3.h"
#include "Ray.h"
#include "defines.h"
#include "HittableList.h"
#include "HittableObjects.h"
#include "HittableSphere.h"
#include "Material.h"
#include "Camera.h"


static constexpr f32 infinity{ std::numeric_limits<f32>::infinity() };

static constexpr color white{ 1.f, 1.f, 1.f };
static constexpr color blue{ 0.5f, 0.7f, 1.f };


struct ImageSpecification {

    i32 width{ 0 };
    i32 height{ 0 };
    f32 aspectRatio{ 0.f };
    i32 samplesPerPixel{ 0 };
    i32 recursionDepth{ 0 };

};


void printRemainingScanlinesWithInfo(ImageSpecification image, i32 remaining) {
    std::cerr << "\rImage " << image.width << "x" << image.height << " Scanlines remaining : " << remaining << ' ' << std::flush;
}


/*
* @brief Function linearly blends white and blue depending on the height of the y coordinate after scaling the ray
* direction to unit length (-1.f, 1.f). Because we're looking at the y height after normalizing the vector, you'll
* notice a horizontal gradient to the color in addition to the vertical gradient.
* @param r just ray, which shall be colored
* @return colored ray
*/
color colorRay(const Ray& ray, const HittableObject* pObject, i32 depth) {
    if (depth <= 0) {
        return color{ 0.f, 0.f, 0.f };
    }

    HitSpecification hitSpecs;
    if (pObject->hit(ray, HitInterval{ 0.001f, infinity }, &hitSpecs)) {
        Ray scattered;
        color attenuation;
        if (hitSpecs.pMaterial->scatter(ray, hitSpecs, &attenuation, &scattered))
            return attenuation * colorRay(scattered, pObject, depth - 1);
        return color(0.f, 0.f, 0.f);
    }
    const vector3 unitDirection{ vector3::normalize(ray.direction) };
    const f32 t{ 0.5f * (unitDirection.y + 1.f) };          // scaling t to <0.f, 1.f>
    return (1.f - t) * white + t * blue;                    // blendedValue = (1 - t) * startValue + t * endValue
                                                            // when t=0 I want white, when t=1 I want blue
}


auto main() -> i32 {

    ImageSpecification image{};
    image.width = 720;
    image.height = 405;
    image.aspectRatio = (f32)image.width / (f32)image.height;
    image.samplesPerPixel = 25;
    image.recursionDepth = 100;
    
    f32(*multisampleFunc)() = image.samplesPerPixel == 1 ? &returnZero<f32> : &generateRandom<f32>;

    CameraSpecification cameraSpecification{};
    cameraSpecification.height = 2.f;
    cameraSpecification.width = cameraSpecification.height * image.aspectRatio;
    cameraSpecification.focalLength = 1.f;
    cameraSpecification.origin = point3{ 0.f, 0.f, 0.f };
    
    Camera camera{ cameraSpecification };

    HittableList world{
        new HittableSphere{ point3{  0.0f,    0.0f,  -1.f}, radius{  0.5f }, new Metal{ color{ 0.8f, 0.8f, 0.8f } }},
        new HittableSphere{ point3{  1.5f,    0.0f,  -1.f}, radius{  0.5f }, new Lambertian{ color{ 0.7f, 0.3f, 0.3f } }},
        new HittableSphere{ point3{ -1.5f,    0.0f,  -2.f}, radius{  0.5f }, new Lambertian{ color{ 0.2f, 0.3f, 0.7f } }},
        new HittableSphere{ point3{ -1.0f,   -0.2f,  -1.f}, radius{  0.3f }, new Metal{ color{ 0.8f, 0.6f, 0.2f } }},
        new HittableSphere{ point3{  0.0f, -100.5f,  -1.f}, radius{ 100.f }, new Lambertian{ color{ 0.8f, 0.8f, 0.f } }}
    };

    std::ofstream file;
    file.open("output_filename.ppm");
    file << "P3\n" << image.width << ' ' << image.height << "\n255\n";
    for (i32 j = image.height - 1; j >= 0; j--) {
        printRemainingScanlinesWithInfo(image, j);
        for (i32 i = 0; i < image.width; i++) {
            color pixel{ 0.f, 0.f, 0.f };
            for (i32 s = 0; s < image.samplesPerPixel; s++) {
                const f32 u = ((f32)i + multisampleFunc()) / ((f32)image.width - 1.f);
                const f32 v = ((f32)j + multisampleFunc()) / ((f32)image.height - 1.f);
                const Ray ray{ camera.origin(), camera.calculateRayDirection(u, v) };
                pixel += colorRay(ray, &world, image.recursionDepth);
            }
            writeColor(file, pixel, image.samplesPerPixel);
        }
    }
    file.close();
    world.clear();
    std::cerr << "\nDone.\n";
	return 0;
}
