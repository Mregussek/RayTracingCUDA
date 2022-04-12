
#ifndef RAY_H
#define RAY_H


#include "defines.h"
#include "vec3.h"


struct Ray {

	RTX_DEVICE Ray() { };
	RTX_DEVICE Ray(point3 _origin, vector3 _direction) :
		origin{ _origin },
		direction{ _direction }
	{ }

	RTX_DEVICE point3 at(f32 t) const {
		return origin + t * direction;
	}

	point3 origin{};
	vector3 direction{};

};


#endif
