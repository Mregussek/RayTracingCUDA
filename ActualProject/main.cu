
#include "Kernels.h"


static void printCrucialInfoAboutRendering(Image* pImage, Blocks* pBlocks) {
    std::cerr << "Rendering a " << pImage->getWidth() << "x" << pImage->getHeight() << " image "
              << "with " << pImage->getSamples() << " samples per pixel in " << pBlocks->getWidth() << "x"
              << pBlocks->getHeight() << " blocks.\n";
}


auto main() -> i32 {
    FilesystemSpecification filesystemSpecs;
    Filesystem filesystem;
    filesystem.load("resources/third.json", &filesystemSpecs);

    ImageSpecification imageSpecs{};
    imageSpecs.width = 1920;
    imageSpecs.height = 1080;
    imageSpecs.samplesPerPixel = 20;
    imageSpecs.recursionDepth = 50;
    
    Image image{};
    image.initialize(imageSpecs);

    BlocksSpecification blockSpecs{};
    blockSpecs.x = 8;
    blockSpecs.y = 8;

    Blocks blocks{};
    blocks.initialize(blockSpecs, &image);

    printCrucialInfoAboutRendering(&image, &blocks);

    curandState* pRandState;
    CUDA_CHECK( cudaMalloc((void**)&pRandState, image.getCount() * sizeof(curandState)));

    u32 itemsCount{ (u32)filesystemSpecs.materials.size() };
    HittableObject** pList;
    CUDA_CHECK( cudaMalloc((void**)&pList, itemsCount * sizeof(HittableObject*)) );
    HittableObject** pWorld;
    CUDA_CHECK( cudaMalloc((void**)&pWorld, 1 * sizeof(HittableObject*)) );
    Camera** pCamera;
    CUDA_CHECK( cudaMalloc((void**)&pCamera, sizeof(Camera*)) );

    f32* positionsGPU;
    CUDA_MALLOC_AND_MEMCPY(positionsGPU, filesystemSpecs.positions);
    f32* colorsGPU;
    CUDA_MALLOC_AND_MEMCPY(colorsGPU, filesystemSpecs.colors);
    i32* materialsGPU;
    CUDA_MALLOC_AND_MEMCPY(materialsGPU, filesystemSpecs.materials);
    f32* radiusGPU;
    CUDA_MALLOC_AND_MEMCPY(radiusGPU, filesystemSpecs.radius);

    RTX_CALL_KERNEL_AND_VALIDATE(
        worldCreate<<<1, 1>>>(pList, pWorld, pCamera, image.getAspectRatio(), itemsCount,
                              positionsGPU, colorsGPU,
                              materialsGPU, radiusGPU) );
    RTX_CALL_KERNEL_AND_VALIDATE(
        renderInit<<<blocks.getBlocks(), blocks.getThreads()>>>(image.getWidth(), image.getHeight(), pRandState)
    );

    Timer<TimerType::MILISECONDS> timer;
    timer.start();

    RTX_CALL_KERNEL_AND_VALIDATE(
        render<<<blocks.getBlocks(), blocks.getThreads()>>>(image.getPixels(),
                                                            image.getWidth(),
                                                            image.getHeight(),
                                                            image.getSamples(),
                                                            image.getDepth(),
                                                            pCamera,
                                                            pWorld,
                                                            pRandState)
    );

    timer.stop();

    RTX_CALL_KERNEL_AND_VALIDATE( worldFree<<<1, 1>>>(pList, pWorld, pCamera, itemsCount) );

    CUDA_CHECK( cudaFree(pRandState) );
    CUDA_CHECK( cudaFree(pCamera) );
    CUDA_CHECK( cudaFree(pList) );
    CUDA_CHECK( cudaFree(pWorld) );

    writeImageToFile("output_image.ppm", &image);
    image.free();

    cudaFree(positionsGPU);
    cudaFree(colorsGPU);
    cudaFree(materialsGPU);
    cudaFree(radiusGPU);

    return 0;
}
