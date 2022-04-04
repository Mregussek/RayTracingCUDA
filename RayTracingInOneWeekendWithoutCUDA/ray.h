
#ifndef RAY_H
#define RAY_H


#include "defines.h"
#include "vec3.h"


struct Ray {

	point3 origin{};
	vector3 direction{};

	Ray() = default;
	
	Ray(point3 _origin, vector3 _direction) :
		origin{ _origin },
		direction{ _direction }
	{ }

	point3 at(f32 t) const {
		return origin + t * direction;
	}

};


#endif
