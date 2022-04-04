
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

    ImageSpecification image{};
    image.width = 720;
    image.height = 405;
    image.aspectRatio = (f32)image.width / (f32)image.height;
    image.samplesPerPixel = 25;
    image.recursionDepth = 100;
    
    CameraSpecification cameraSpecification{};
    cameraSpecification.height = 2.f;
    cameraSpecification.width = cameraSpecification.height * image.aspectRatio;
    cameraSpecification.focalLength = 1.f;
    cameraSpecification.origin = point3{ 0.f, 0.f, 0.f };
    
    Camera camera{ cameraSpecification };

    HittableList world{
        new HittableSphere{ point3{  0.0f,    0.0f,  -1.f}, radius{  0.5f }, new Metal{ color{ 0.8f, 0.8f, 0.8f } }},
        new HittableSphere{ point3{  1.5f,    0.0f,  -1.f}, radius{  0.5f }, new Lambertian{ color{ 0.7f, 0.3f, 0.3f } }},
        new HittableSphere{ point3{ -1.5f,    0.0f,  -2.f}, radius{  0.5f }, new Lambertian{ color{ 0.2f, 0.3f, 0.7f } }},
        new HittableSphere{ point3{ -1.0f,   -0.2f,  -1.f}, radius{  0.3f }, new Metal{ color{ 0.8f, 0.6f, 0.2f } }},
        new HittableSphere{ point3{  0.0f, -100.5f,  -1.f}, radius{ 100.f }, new Lambertian{ color{ 0.8f, 0.8f, 0.f } }}
    };

    render(image, &camera, &world);

    world.clear();
    std::cerr << "\nDone.\n";
	return 0;
}
