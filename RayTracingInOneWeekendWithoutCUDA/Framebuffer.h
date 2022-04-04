
#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H


#include "defines.h"
#include "vec3.h"
#include "HittableObjects.h"
#include "Camera.h"
#include "Ray.h"


static constexpr color white{ 1.f, 1.f, 1.f };
static constexpr color blue{ 0.5f, 0.7f, 1.f };


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


template<typename val = f32>
val clamp(val x, val min, val max) {
    if (x < min) {
        return min;
    }
    if (x > max) {
        return max;
    }
    return x;
}

template<typename val = f32>
void writePixelToFile(std::ostream& out, vec3<val> pixel) {
    out << (i32)(pixel.r) << ' '
        << (i32)(pixel.g) << ' '
        << (i32)(pixel.b) << '\n';
}


color applyPostProcessing(color pixel, i32 samplesPerPixel) {
    const f32 scale{ 1.f / (f32)samplesPerPixel };
    pixel *= scale;

    if constexpr (ENABLE_GAMMA_CORRECTION) {
        pixel = vector3::square(pixel);
    }

    pixel = {
        255.999f * clamp(pixel.r, 0.f, 0.999f),
        255.999f * clamp(pixel.g, 0.f, 0.999f),
        255.999f * clamp(pixel.b, 0.f, 0.999f)
    };

    return pixel;
}


void render(ImageSpecification image, Camera* pCamera, HittableObject* pWorld) {
    f32(*multisampleFunc)() = image.samplesPerPixel == 1 ? &returnZero<f32> : &generateRandom<f32>;

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
                const Ray ray{ pCamera->origin(), pCamera->calculateRayDirection(u, v) };
                pixel += colorRay(ray, pWorld, image.recursionDepth);
            }
            
            pixel = applyPostProcessing(pixel, image.samplesPerPixel);

            writePixelToFile(file, pixel);
        }
    }
    file.close();
}


#endif
