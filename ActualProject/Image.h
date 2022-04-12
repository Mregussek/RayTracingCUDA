
#ifndef IMAGE_H
#define IMAGE_H


#include "defines.h"
#include "vec3.h"
#include "Platform.h"


struct ImageSpecification {

    u32 width{ 0 };
    u32 height{ 0 };
    i32 samplesPerPixel{ 0 };
    i32 recursionDepth{ 0 };

};


class Image {
public:

    void initialize(ImageSpecification _imageSpecs) {
        imageSpecs = _imageSpecs;
        aspectRatio = (f32)imageSpecs.width / (f32)imageSpecs.height;
        multisampleFunc = imageSpecs.samplesPerPixel == 1 ? &returnZero : &generateRandom;
        countPixels = imageSpecs.width * imageSpecs.height;
        imageSizeof = countPixels * sizeof(color);
        CUDA_CHECK( cudaMallocManaged((void**)&pPixels, imageSizeof) );
    }

    void free() {
        CUDA_CHECK(cudaFree(pPixels));
    }

    f32 getAspectRatio() const { return aspectRatio; }
    u32 getWidth() const { return imageSpecs.width; }
    u32 getHeight() const { return imageSpecs.height; }
    u32 getSizeof() const { return imageSizeof; }
    u32 getCount() const { return countPixels; }
    color* getPixels() { return pPixels; }

private:

    ImageSpecification imageSpecs;
    f32(*multisampleFunc)() { nullptr };
    color* pPixels;
    f32 aspectRatio{ 0.f };
    u32 countPixels{ 0 };
    u32 imageSizeof{ 0 };

};


static void writePixelToFile(std::ostream& out, vec3 pixel) {
    out << (i32)(pixel.r) << ' '
        << (i32)(pixel.g) << ' '
        << (i32)(pixel.b) << '\n';
}


void writeImageToFile(const char* outputPath, Image* pImage) {
    const u32 width{ pImage->getWidth() };
    const u32 height{ pImage->getHeight() };

    std::ofstream file;
    file.open(outputPath);
    file << "P3\n" << width << ' ' << height << "\n255\n";

    for (i32 i = pImage->getCount() - 1; i >= 0; i--) {
        writePixelToFile(file, pImage->getPixels()[i]);
    }

    file.close();
}


#endif
