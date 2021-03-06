
#ifndef KERNELS_H
#define KERNELS_H


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
#include "Material.h"
#include "Filesystem.h"


RTX_GLOBAL void worldCreate(HittableObject** pList, HittableObject** pWorld, Camera** pCamera,
    f32 aspectRatio, u32 listCount,
    f32* positions, f32* colors, i32* materials, f32* radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (u32 i = 0; i < listCount; i++) {
            Material* pMaterial = materials[i] == 0 ? (Material*)new Metal{} : (Material*)new Lambertian{};
            pMaterial->setAlbedo(colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]);
            *(pList + i) = new HittableSphere{
                point3{ positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2] },
                f32{ radius[i] },
                pMaterial
            };
        }
        *pWorld = new HittableList(pList, listCount);

        CameraSpecification camSpecs;
        camSpecs.height = 2.f;
        camSpecs.width = camSpecs.height * aspectRatio;
        camSpecs.focalLength = 1.f;
        camSpecs.origin = point3{ 0.f, 0.f, 0.f };

        *pCamera = new Camera();
        (*pCamera)->initialize(camSpecs);
    }
}


RTX_GLOBAL void worldFree(HittableObject** pList, HittableObject** pWorld, Camera** pCamera, u32 listCount) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (u32 i = 0; i < listCount; i++) {
            (*(pList + i))->deleteMaterial();
            delete* (pList + i);
        }
        delete* pWorld;
        delete* pCamera;
    }
}


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
    vec3 currentAttenuation{ 1.f, 1.f, 1.f };

    for (i32 i = 0; i < recursionDepth; i++) {
        HitSpecification hitSpecs;
        if ((*pWorld)->hit(currentRay, { 0.001f, FLT_MAX }, &hitSpecs)) {
            Ray scatteredRay;
            vec3 attenuation;
            if (hitSpecs.pMaterial->scatter(currentRay, hitSpecs, &attenuation, &scatteredRay, pRandState)) {
                currentAttenuation = currentAttenuation * attenuation;
                currentRay = scatteredRay;
            }
            else {
                return vec3{};
            }
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


RTX_DEVICE f32 clamp(f32 x, f32 min, f32 max) {
    if (x < min) {
        return min;
    }
    if (x > max) {
        return max;
    }
    return x;
}


RTX_DEVICE color applyPostProcessing(color pixel, i32 samplesPerPixel) {
    const f32 scale{ 1.f / (f32)samplesPerPixel };
    color sampledPixel{ pixel * scale };
    if constexpr (ENABLE_GAMMA_CORRECTION) {
        sampledPixel = vec3::square(sampledPixel);
    }
    return {
        255.999f * clamp(sampledPixel.r, 0.f, 0.999f),
        255.999f * clamp(sampledPixel.g, 0.f, 0.999f),
        255.999f * clamp(sampledPixel.b, 0.f, 0.999f)
    };
}


RTX_GLOBAL void render(color* pPixels, u32 imageWidth, u32 imageHeight, u32 samples, i32 recursionDepth,
    Camera** pCamera, HittableObject** pWorld, curandState* pRandState) {
    const u32 i{ threadIdx.x + blockIdx.x * blockDim.x };
    const u32 j{ threadIdx.y + blockIdx.y * blockDim.y };
    if ((i >= imageWidth) || (j >= imageHeight)) { return; }
    const u32 index{ j * imageWidth + i };

    curandState localRandState = pRandState[index];
    color localPixel{ 0.f, 0.f, 0.f };

    for (u32 s = 0; s < samples; s++) {
        const f32 u{ ((f32)i + curand_uniform(&localRandState)) / (f32)imageWidth };
        const f32 v{ ((f32)j + curand_uniform(&localRandState)) / (f32)imageHeight };
        const Ray ray{ (*pCamera)->origin(), (*pCamera)->calculateRayDirection(u, v) };
        localPixel = localPixel + colorRay(ray, pWorld, &localRandState, recursionDepth);
    }
    pRandState[index] = localRandState;

    pPixels[index] = applyPostProcessing(localPixel, samples);
}


#endif
