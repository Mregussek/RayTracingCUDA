
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

    void initialize(ImageSpecification _imageSpecs);
    void free();

    f32 getAspectRatio() const { return aspectRatio; }
    u32 getWidth() const { return imageSpecs.width; }
    u32 getHeight() const { return imageSpecs.height; }
    u32 getSizeof() const { return imageSizeof; }
    u32 getCount() const { return countPixels; }
    i32 getSamples() const { return imageSpecs.samplesPerPixel; }
    i32 getDepth() const { return imageSpecs.recursionDepth; }
    color* getPixels() { return pPixels; }

private:

    ImageSpecification imageSpecs;
    color* pPixels;
    f32 aspectRatio{ 0.f };
    u32 countPixels{ 0 };
    u32 imageSizeof{ 0 };

};


void Image::initialize(ImageSpecification _imageSpecs) {
    imageSpecs = _imageSpecs;
    aspectRatio = (f32)imageSpecs.width / (f32)imageSpecs.height;
    countPixels = imageSpecs.width * imageSpecs.height;
    imageSizeof = countPixels * sizeof(color);
    CUDA_CHECK(cudaMallocManaged((void**)&pPixels, imageSizeof));
}

void Image::free() {
    CUDA_CHECK(cudaFree(pPixels));
}


static void writePixelToFile(std::ostream& out, vec3 pixel) {
    out << (i32)(pixel.r) << ' '
        << (i32)(pixel.g) << ' '
        << (i32)(pixel.b) << '\n';
}


void writeImageToFile(const char* outputPath, Image* pImage) {
    const i32 width{ (i32)pImage->getWidth() };
    const i32 height{ (i32)pImage->getHeight() };

    std::ofstream file;
    file.open(outputPath);
    file << "P3\n" << width << ' ' << height << "\n255\n";

    for (i32 j = (i32)height - 1; j >= 0; j--) {
        for (i32 i = 0; i < width; i++) {
            const i32 index{ j * width + i };
            writePixelToFile(file, pImage->getPixels()[index]);
        }
    }

    file.close();
}


#endif
