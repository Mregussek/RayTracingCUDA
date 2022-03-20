
#ifndef RAY_H
#define RAY_H


#include "defines.h"
#include "vec3.h"


struct Ray {

	point3 origin{};
	vector3 direction{};

	constexpr Ray() = default;

	constexpr Ray(point3 _origin, vector3 _direction) :
		origin{ _origin },
		direction{ _direction }
	{ }

	template<typename val = f32>
	constexpr point3 at(val t) {
		return origin + t * direction;
	}

};


#endif
