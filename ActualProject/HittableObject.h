
#ifndef HITTABLE_H
#define HITTABLE_H


#include "vec3.h"


struct Ray;
struct Material;


struct HitSpecification {

	point3 point{ 0.f, 0.f, 0.f };
	vector3 normal{ 0.f, 0.f, 0.f };
	Material* pMaterial{ nullptr };
	f32 t{ 0.f };
	b8 frontFace{ 0 };

	HitSpecification() = default;
	RTX_DEVICE HitSpecification(point3 _p, vector3 _n, f32 _t) :
		point(_p),
		normal(_n),
		t(_t)
	{ }

};


struct HitInterval {

	f32 max{ 0.f };
	f32 min{ 0.f };

	RTX_DEVICE HitInterval(f32 _min, f32 _max) :
		max(_max),
		min(_min)
	{ }

};


class HittableObject {
public:

	RTX_DEVICE virtual b8 hit(const Ray& ray, HitInterval interval, HitSpecification* hitSpecs) const { return RTX_FALSE; }
	RTX_DEVICE virtual b8 deleteMaterial() { return RTX_FALSE; }

};

#endif
