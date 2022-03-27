
#ifndef HITTABLEOBJECTS_H
#define HITTABLEOBJECTS_H


#include "defines.h"
#include "vec3.h"
#include "Ray.h"


struct Ray;


struct HitSpecification {

	point3 point{ 0.f, 0.f, 0.f };
	vector3 normal{ 0.f, 0.f, 0.f };
	f32 t{ 0.f };
	b8 frontFace{ 0 };

	constexpr HitSpecification() = default;
	constexpr HitSpecification(point3 _p, vector3 _n, f32 _t) :
		point(_p),
		normal(_n),
		t(_t)
	{ }

};


struct HitInterval {

	f32 max{ 0.f };
	f32 min{ 0.f };

	constexpr HitInterval(f32 _min, f32 _max) :
		max(_max),
		min(_min)
	{ }

};


class HittableObject {
public:

	virtual b8 hit(const Ray& ray, HitInterval interval, HitSpecification* hitSpecs) const { return false; }

};


#endif
