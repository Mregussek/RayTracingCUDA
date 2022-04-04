
#include "Framebuffer.h"
#include <fstream>
#include "HittableObjects.h"
#include "Camera.h"
#include "Ray.h"
#include "Material.h"
#include "Platform.h"


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


void Image::initialize(ImageSpecification _imageSpecs) {
    imageSpecs = _imageSpecs;
    multisampleFunc = imageSpecs.samplesPerPixel == 1 ? &returnZero : &generateRandom;
    countPixels = imageSpecs.width * imageSpecs.height;
    if (!pPixels) {
        pPixels = new color[countPixels];
    }
}

void Image::free() {
    if (pPixels) {
        delete[] pPixels;
    }
}

void Image::render(Camera* pCamera, HittableObject* pWorld) {
    i32 k = 0;
    for (i32 j = imageSpecs.height - 1; j >= 0; j--) {
        printRemainingScanlinesWithInfo(j);
        for (i32 i = 0; i < imageSpecs.width; i++) {
            color pixel{ 0.f, 0.f, 0.f };
            for (i32 s = 0; s < imageSpecs.samplesPerPixel; s++) {
                const f32 u = ((f32)i + multisampleFunc()) / ((f32)imageSpecs.width - 1.f);
                const f32 v = ((f32)j + multisampleFunc()) / ((f32)imageSpecs.height - 1.f);
                const Ray ray{ pCamera->origin(), pCamera->calculateRayDirection(u, v) };
                pixel += colorRay(ray, pWorld, imageSpecs.recursionDepth);
            }

            pixel = applyPostProcessing(pixel, imageSpecs.samplesPerPixel);
            pPixels[k] = pixel;
            k++;
        }
    }
}

f32 Image::clamp(f32 x, f32 min, f32 max) const {
    if (x < min) {
        return min;
    }
    if (x > max) {
        return max;
    }
    return x;
}

color Image::applyPostProcessing(color pixel, i32 samplesPerPixel) const {
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

void Image::printRemainingScanlinesWithInfo(i32 remaining) const {
    std::cerr << "\rImage " << imageSpecs.width << "x" << imageSpecs.height << " Scanlines remaining : " << remaining << ' ' << std::flush;
}


static void writePixelToFile(std::ostream& out, vec3 pixel) {
    out << (i32)(pixel.r) << ' '
        << (i32)(pixel.g) << ' '
        << (i32)(pixel.b) << '\n';
}


void writeImageToFile(const char* outputPath, Image* pImage) {
    std::ofstream file;
    file.open("output_filename.ppm");
    file << "P3\n" << pImage->width() << ' ' << pImage->height() << "\n255\n";

    for (i32 i = 0; i < pImage->pixelsCount(); i++) {
        writePixelToFile(file, pImage->pixel(i));
    }

    file.close();
}