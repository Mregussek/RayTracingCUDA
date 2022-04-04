
#ifndef HITTABLESPHERE_H
#define HITTABLESPHERE_H


#include "HittableObjects.h"
#include "defines.h"
#include "vec3.h"


struct Ray;


class HittableSphere : public HittableObject {
public:

	HittableSphere() = default;
	HittableSphere(point3 _center, f32 _radius, Material* _pMaterial);

	b8 hit(const Ray& ray, HitInterval interval, HitSpecification* hitSpecs) const override;

	b8 deleteMaterial();

	static vector3 isRandomInUnitSphere();

private:

	point3 center{ 0.f, 0.f, 0.f };
	f32 radius{ 0.f };
	Material* pMaterial{ nullptr };

};


#endif
