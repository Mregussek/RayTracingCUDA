
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


RTX_GLOBAL void renderInit(Camera** pCamera, u32 imageWidth, u32 imageHeight, f32 aspectRatio) {
    const u32 i{ threadIdx.x + blockIdx.x * blockDim.x };
    const u32 j{ threadIdx.y + blockIdx.y * blockDim.y };
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    // const u32 index{ j * imageWidth + i };

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


RTX_DEVICE b8 hitSphere(const vec3& center, f32 radius, const Ray& ray) {
    const vec3 oc = ray.origin - center;
    const f32 a = vec3::dot(ray.direction, ray.direction);
    const f32 b = 2.f * vec3::dot(oc, ray.direction);
    const f32 c = vec3::dot(oc, oc) - radius * radius;
    const f32 discriminant = b * b - 4.f * a * c;
    return (discriminant > 0.f);
}


RTX_DEVICE vec3 colorRay(const Ray& ray, HittableObject** pWorld) {
    const color white{ 1.f, 1.f, 1.f };
    const color blue{ 0.5f, 0.7f, 1.f };

    HitSpecification hitSpecs;
    if ((*pWorld)->hit(ray, { 0.f, FLT_MAX }, &hitSpecs)) {
        return 0.5f * (hitSpecs.normal + 1.f);
    }

    const vec3 unitRayDirection{ vec3::normalize(ray.direction) };
    const f32 t{ 0.5f * (unitRayDirection.y + 1.f) };
    return (1.f - t) * white + t * blue;
}


RTX_GLOBAL void render(color* pPixels, u32 imageWidth, u32 imageHeight, Camera** pCamera, HittableObject** pWorld) {
    const u32 i{ threadIdx.x + blockIdx.x * blockDim.x };
    const u32 j{ threadIdx.y + blockIdx.y * blockDim.y };
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    const u32 index{ j * imageWidth + i };

    const f32 u{ (f32)i / (f32)imageWidth };
    const f32 v{ (f32)j / (f32)imageHeight };
    const Ray ray{ (*pCamera)->origin(), (*pCamera)->calculateRayDirection(u, v) };
    pPixels[index] = 255.99f * colorRay(ray, pWorld);
}


RTX_GLOBAL void renderClose(Camera** pCamera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (*pCamera) {
            delete *pCamera;
        }
    }
}


RTX_GLOBAL void worldCreate(HittableObject** pList, HittableObject** pWorld) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(pList + 0) = new HittableSphere({ 0.f, 0.f, -1.f }, 0.5f);
        *(pList + 1) = new HittableSphere({ 0.f, -100.5f, -1.f }, 100.f);
        *pWorld = new HittableList(pList, 2);
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

    HittableObject** pList;
    CUDA_CHECK( cudaMalloc((void**)&pList, 2 * sizeof(HittableObject*)) );
    HittableObject** pWorld;
    CUDA_CHECK( cudaMalloc((void**)&pWorld, 1 * sizeof(HittableObject*)) );
    Camera** pCamera;
    CUDA_CHECK( cudaMalloc((void**)&pCamera, sizeof(Camera*)) );

    RTX_CALL_KERNEL_AND_VALIDATE( worldCreate<<<1, 1>>>(pList, pWorld) );
    RTX_CALL_KERNEL_AND_VALIDATE( renderInit<<<1, 1>>> (pCamera, image.getWidth(), image.getHeight(), image.getAspectRatio()) );

    Timer<TimerType::MICROSECONDS> timer;
    timer.start();

    RTX_CALL_KERNEL_AND_VALIDATE(
        render<<<blocks.getBlocks(), blocks.getThreads()>>>(image.getPixels(), image.getWidth(), image.getHeight(), pCamera, pWorld)
    );

    timer.stop();

    RTX_CALL_KERNEL_AND_VALIDATE( renderClose<<<1, 1>>>(pCamera) );
    RTX_CALL_KERNEL_AND_VALIDATE( worldFree<<<1, 1>>>(pList, pWorld) );

    CUDA_CHECK( cudaFree(pCamera) );
    CUDA_CHECK( cudaFree(pList) );
    CUDA_CHECK( cudaFree(pWorld) );

    writeImageToFile("output_image.ppm", &image);
    image.free();

    return 0;
}
