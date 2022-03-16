
#include <iostream>
#include <fstream>
#include <cstdint>


using i8 = int_fast8_t;
using i16 = int_least16_t;
using i32 = int_fast32_t;
using i64 = int_fast64_t;

using u8 = uint_fast8_t;
using u16 = uint_least16_t;
using u32 = uint_fast32_t;
using u64 = uint_fast64_t;

using f32 = float;
using f64 = double;

using b8 = i8;
using b32 = i32;


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
            const i32 ir = (i32)(255.999f * r);
            const i32 ig = (i32)(255.999f * g);
            const i32 ib = (i32)(255.999f * b);
            file << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    file.close();
    std::cerr << "\nDone.\n";
	return 0;
}
