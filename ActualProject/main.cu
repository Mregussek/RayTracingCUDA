
#include "defines.h"
#include "Platform.h"
#include "vec3.h"
#include "Image.h"
#include "Blocks.h"


__global__ void render(color* pPixels, i32 imageWidth, i32 imageHeight) {
    i32 i = threadIdx.x + blockIdx.x * blockDim.x;
    i32 j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    i32 index = j * imageWidth + i;
    pPixels[index] = 255.99f * vec3((f32)i / imageWidth, (f32)j / imageHeight, 0.2f);
}


void printCrucialInfoAboutRendering(Image* pImage, Blocks* pBlocks) {
    std::cerr << "Rendering a " << pImage->getWidth() << "x" << pImage->getHeight() << " image ";
    std::cerr << "in " << pBlocks->getWidth() << "x" << pBlocks->getHeight() << " blocks.\n";
}


auto main() -> i32 {

    ImageSpecification imageSpecs{};
    imageSpecs.width = 720;
    imageSpecs.height = 405;
    imageSpecs.samplesPerPixel = 25;
    imageSpecs.recursionDepth = 100;
    
    Image image{};
    image.initialize(imageSpecs);

    BlocksSpecification blockSpecs{};
    blockSpecs.x = 8;
    blockSpecs.y = 8;

    Blocks blocks{};
    blocks.initialize(blockSpecs, &image);

    printCrucialInfoAboutRendering(&image, &blocks);

    clock_t start, stop;
    start = clock();
    // Render our buffer
    render<<<blocks.getBlocks(), blocks.getThreads()>>>(image.getPixels(), image.getWidth(), image.getHeight());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    writeImageToFile("output_image.ppm", &image);

    image.free();
}
