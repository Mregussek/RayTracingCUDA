
#ifndef RAY_H
#define RAY_H


#include "defines.h"
#include "vec3.h"


struct ray {

	point3 origin{};
	vector3 direction{};


	ray() = default;

	ray(point3 _origin, vector3 _direction) :
		origin{ _origin },
		direction{ _direction }
	{ }

	template<typename val = f32>
	point3 at(val t) {
		return origin + t * direction;
	}

};


#endif
