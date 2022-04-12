
#include "defines.h"
#include "Platform.h"
#include "vec3.h"
#include "Image.h"
#include "Blocks.h"
#include "Timer.h"
#include "Ray.h"
#include "Camera.h"


RTX_GLOBAL void initRender(Camera** pCamera, u32 imageWidth, u32 imageHeight, f32 aspectRatio) {
    const u32 i{ threadIdx.x + blockIdx.x * blockDim.x };
    const u32 j{ threadIdx.y + blockIdx.y * blockDim.y };
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    const u32 index{ j * imageWidth + i };

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        CameraSpecification camSpecs;
        camSpecs.height = 2.f;
        camSpecs.width = camSpecs.height * aspectRatio;
        camSpecs.focalLength = 1.f;
        camSpecs.origin = point3{ 0.f, 0.f, 0.f };

        *pCamera = new Camera();
        (*pCamera)->initialize(camSpecs);
    }
}


RTX_DEVICE vec3 colorRay(const Ray& ray) {
    const color white{ 1.f, 1.f, 1.f };
    const color blue{ 0.5f, 0.7f, 1.f };

    const vec3 unitRayDirection{ vec3::normalize(ray.direction) };
    const f32 t{ 0.5f * (unitRayDirection.y + 1.f) };
    return (1.f - t) * white + t * blue;
}


RTX_GLOBAL void render(color* pPixels, u32 imageWidth, u32 imageHeight, Camera** pCamera) {
    const u32 i{ threadIdx.x + blockIdx.x * blockDim.x };
    const u32 j{ threadIdx.y + blockIdx.y * blockDim.y };
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    const u32 index{ j * imageWidth + i };

    const f32 u{ (f32)i / (f32)imageWidth };
    const f32 v{ (f32)j / (f32)imageHeight };
    const Ray ray{ (*pCamera)->origin(), (*pCamera)->calculateRayDirection(u, v) };
    pPixels[index] = 255.99f * colorRay(ray);
}


RTX_GLOBAL void closeRender(Camera** pCamera) {
    delete *pCamera;
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

    Camera** pCamera;
    CUDA_CHECK( cudaMalloc((void**)&pCamera, sizeof(Camera*)) );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    initRender<<<1, 1>>>(pCamera, image.getWidth(), image.getHeight(), image.getAspectRatio());
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    Timer<TimerType::MICROSECONDS> timer;
    timer.start();

    render<<<blocks.getBlocks(), blocks.getThreads()>>>(image.getPixels(),
                                                        image.getWidth(),
                                                        image.getHeight(),
                                                        pCamera);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );
    
    timer.stop();

    writeImageToFile("output_image.ppm", &image);

    image.free();

    closeRender<<<1, 1>>>(pCamera);
    CUDA_CHECK( cudaFree(pCamera) );

    return 0;
}
