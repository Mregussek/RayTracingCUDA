
#include <iostream>
#include <fstream>
#include "vec3.h"
#include "ray.h"
#include "defines.h"


struct ImageSpecification {
    i32 width{ 256 };
    i32 height{ 256 };
};


void print_remaining_scanlines_with_info(ImageSpecification image, i32 remaining) {
    std::cerr << "\rImage " << image.height << "x" << image.width << " Scanlines remaining : " << remaining << ' ' << std::flush;
}


auto main() -> i32 {
	
    constexpr ImageSpecification image{};

    std::ofstream file;
    file.open("output_filename.ppm");
    file << "P3\n" << image.width << ' ' << image.height << "\n255\n";
    for (i32 j = image.height - 1; j >= 0; j--) {
        print_remaining_scanlines_with_info(image, j);
        for (i32 i = 0; i < image.width; i++) {
            const f32 r = (f32)i / ((f32)image.width - 1.f);
            const f32 g = (f32)j / ((f32)image.height - 1.f);
            const f32 b = 0.25f;
            writeColor(file, color{ r, g, b });
        }
    }
    file.close();
    std::cerr << "\nDone.\n";
	return 0;
}
