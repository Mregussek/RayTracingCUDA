
#include "defines.h"
#include "Platform.h"
#include "vec3.h"
#include "Image.h"
#include "Blocks.h"
#include "Timer.h"
#include "Ray.h"
#include "Camera.h"
#include "HittableList.h"
#include "HittableObject.h"
#include "HittableSphere.h"


RTX_GLOBAL void renderInit(u32 imageWidth, u32 imageHeight, curandState* pRandState) {
    const u32 i{ threadIdx.x + blockIdx.x * blockDim.x };
    const u32 j{ threadIdx.y + blockIdx.y * blockDim.y };
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    const u32 index{ j * imageWidth + i };
    curand_init(1984, index, 0, &pRandState[index]);
}


RTX_DEVICE vec3 colorRay(const Ray& ray, HittableObject** pWorld, curandState* pRandState, i32 recursionDepth) {
    const color white{ 1.f, 1.f, 1.f };
    const color blue{ 0.5f, 0.7f, 1.f };

    Ray currentRay{ ray };
    f32 currentAttenuation{ 1.f };

    for (i32 i = 0; i < recursionDepth; i++) {
        HitSpecification hitSpecs;
        if ((*pWorld)->hit(currentRay, { 0.001f, FLT_MAX }, &hitSpecs)) {
            const vec3 target{ hitSpecs.point + hitSpecs.normal + HittableSphere::isRandomInUnitSphere(pRandState) };
            currentAttenuation *= 0.5f;
            currentRay = Ray{ hitSpecs.point, target - hitSpecs.point };
        }
        else {
            const vec3 unitRayDirection{ vec3::normalize(currentRay.direction) };
            const f32 t{ 0.5f * (unitRayDirection.y + 1.f) };
            const vec3 heavenColor{ (1.f - t) * white + t * blue };
            return currentAttenuation * heavenColor;
        }
    }

    return { 0.f, 0.f, 0.f };
}


RTX_GLOBAL void render(color* pPixels, u32 imageWidth, u32 imageHeight, u32 samples, i32 recursionDepth,
                       Camera** pCamera, HittableObject** pWorld, curandState* pRandState) {
    const u32 i{ threadIdx.x + blockIdx.x * blockDim.x };
    const u32 j{ threadIdx.y + blockIdx.y * blockDim.y };
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    const u32 index{ j * imageWidth + i };

    curandState localRandState = pRandState[index];
    vec3 localPixel{ 0.f, 0.f, 0.f };

    for (u32 s = 0; s < samples; s++) {
        const f32 u{ ((f32)i + curand_uniform(&localRandState)) / (f32)imageWidth };
        const f32 v{ ((f32)j + curand_uniform(&localRandState)) / (f32)imageHeight };
        const Ray ray{ (*pCamera)->origin(), (*pCamera)->calculateRayDirection(u, v) };
        localPixel = localPixel + colorRay(ray, pWorld, &localRandState, recursionDepth);
    }
    pRandState[index] = localRandState;
    localPixel = localPixel / (f32)samples;
    localPixel = vec3::square(localPixel);
    pPixels[index] = localPixel * 255.99f;
}


RTX_GLOBAL void renderClose(Camera** pCamera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *pCamera;
    }
}


RTX_GLOBAL void worldCreate(HittableObject** pList, HittableObject** pWorld, Camera** pCamera, f32 aspectRatio) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(pList + 0) = new HittableSphere({ 0.f, 0.f, -1.f }, 0.5f);
        *(pList + 1) = new HittableSphere({ 0.f, -100.5f, -1.f }, 100.f);
        *pWorld = new HittableList(pList, 2);

        CameraSpecification camSpecs;
        camSpecs.height = 2.f;
        camSpecs.width = camSpecs.height * aspectRatio;
        camSpecs.focalLength = 1.f;
        camSpecs.origin = point3{ 0.f, 0.f, 0.f };

        *pCamera = new Camera();
        (*pCamera)->initialize(camSpecs);
    }
}


RTX_GLOBAL void worldFree(HittableObject** pList, HittableObject** pWorld) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *(pList + 0);
        delete *(pList + 1);
        delete *pWorld;
    }
}


void printCrucialInfoAboutRendering(Image* pImage, Blocks* pBlocks) {
    std::cerr << "Rendering a " << pImage->getWidth() << "x" << pImage->getHeight() << " image ";
    std::cerr << "in " << pBlocks->getWidth() << "x" << pBlocks->getHeight() << " blocks.\n";
}


auto main() -> i32 {

    ImageSpecification imageSpecs{};
    imageSpecs.width = 720;
    imageSpecs.height = 405;
    imageSpecs.samplesPerPixel = 100;
    imageSpecs.recursionDepth = 100;
    
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

    HittableObject** pList;
    CUDA_CHECK( cudaMalloc((void**)&pList, 2 * sizeof(HittableObject*)) );
    HittableObject** pWorld;
    CUDA_CHECK( cudaMalloc((void**)&pWorld, 1 * sizeof(HittableObject*)) );
    Camera** pCamera;
    CUDA_CHECK( cudaMalloc((void**)&pCamera, sizeof(Camera*)) );

    RTX_CALL_KERNEL_AND_VALIDATE( worldCreate<<<1, 1>>>(pList, pWorld, pCamera, image.getAspectRatio()) );
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

    RTX_CALL_KERNEL_AND_VALIDATE( renderClose<<<1, 1>>>(pCamera) );
    RTX_CALL_KERNEL_AND_VALIDATE( worldFree<<<1, 1>>>(pList, pWorld) );

    CUDA_CHECK( cudaFree(pRandState) );
    CUDA_CHECK( cudaFree(pCamera) );
    CUDA_CHECK( cudaFree(pList) );
    CUDA_CHECK( cudaFree(pWorld) );

    writeImageToFile("output_image.ppm", &image);
    image.free();

    return 0;
}
