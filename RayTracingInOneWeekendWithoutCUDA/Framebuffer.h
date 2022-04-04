
#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H


#include "defines.h"
#include "vec3.h"


class HittableObject;
struct Ray;
class Camera;


static const color white{ 1.f, 1.f, 1.f };
static const color blue{ 0.5f, 0.7f, 1.f };


/*
* @brief Function linearly blends white and blue depending on the height of the y coordinate after scaling the ray
* direction to unit length (-1.f, 1.f). Because we're looking at the y height after normalizing the vector, you'll
* notice a horizontal gradient to the color in addition to the vertical gradient.
* @param r just ray, which shall be colored
* @return colored ray
*/
color colorRay(const Ray& ray, const HittableObject* pObject, i32 depth);


struct ImageSpecification {

    i32 width{ 0 };
    i32 height{ 0 };
    f32 aspectRatio{ 0.f };
    i32 samplesPerPixel{ 0 };
    i32 recursionDepth{ 0 };

};


class Image {
public:

    void initialize(ImageSpecification _imageSpecs);
    void free();

    void render(Camera* pCamera, HittableObject* pWorld);

    i32 width() const { return imageSpecs.width; }
    i32 height() const { return imageSpecs.height; }
    i32 pixelsCount() const { return countPixels; }
    color pixel(i32 i) const { return pPixels[i]; }

private:

    f32 clamp(f32 x, f32 min, f32 max) const;

    color applyPostProcessing(color pixel, i32 samplesPerPixel) const;

    void printRemainingScanlinesWithInfo(i32 remaining) const;


    ImageSpecification imageSpecs{};
    color* pPixels{ nullptr };
    f32(*multisampleFunc)() { nullptr };
    i32 countPixels{ 0 };

};


void writeImageToFile(const char* outputPath, Image* pImage);


#endif
