
#include <iostream>
#include <fstream>
#include <random>
#include <limits>
#include "vec3.h"
#include "Ray.h"
#include "defines.h"
#include "HittableList.h"
#include "HittableObjects.h"
#include "HittableSphere.h"
#include "Material.h"
#include "Camera.h"
#include "Framebuffer.h"


auto main() -> i32 {

    ImageSpecification imageSpecs{};
    imageSpecs.width = 720;
    imageSpecs.height = 405;
    imageSpecs.aspectRatio = (f32)imageSpecs.width / (f32)imageSpecs.height;
    imageSpecs.samplesPerPixel = 25;
    imageSpecs.recursionDepth = 100;

    CameraSpecification cameraSpecification{};
    cameraSpecification.height = 2.f;
    cameraSpecification.width = cameraSpecification.height * imageSpecs.aspectRatio;
    cameraSpecification.focalLength = 1.f;
    cameraSpecification.origin = point3{ 0.f, 0.f, 0.f };

    Camera camera;
    camera.initialize(cameraSpecification);

    HittableList world{
        new HittableSphere{ point3{  0.0f,    0.0f,  -1.f}, radius{  0.5f }, new Metal{ color{ 0.8f, 0.8f, 0.8f } }},
        new HittableSphere{ point3{  1.5f,    0.0f,  -1.f}, radius{  0.5f }, new Lambertian{ color{ 0.7f, 0.3f, 0.3f } }},
        new HittableSphere{ point3{ -1.5f,    0.0f,  -2.f}, radius{  0.5f }, new Lambertian{ color{ 0.2f, 0.3f, 0.7f } }},
        new HittableSphere{ point3{ -1.0f,   -0.2f,  -1.f}, radius{  0.3f }, new Metal{ color{ 0.8f, 0.6f, 0.2f } }},
        new HittableSphere{ point3{  0.0f, -100.5f,  -1.f}, radius{ 100.f }, new Lambertian{ color{ 0.8f, 0.8f, 0.f } }}
    };

    Image image;
    image.initialize(imageSpecs);

    image.render(&camera, &world);
    writeImageToFile("output_image.ppm", &image);

    image.free();
    world.clear();
    std::cerr << "\nDone.\n";
	return 0;
}
