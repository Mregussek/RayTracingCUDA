
#ifndef HITTABLESPHERE_H
#define HITTABLESPHERE_H


#include "HittableObject.h"
#include "Ray.h"
#include "defines.h"
#include "vec3.h"


struct Ray;
struct Material;


class HittableSphere : public HittableObject {
public:

	RTX_DEVICE HittableSphere(point3 _center, f32 _radius, Material* _pMaterial);

	RTX_DEVICE b8 hit(const Ray& ray, HitInterval interval, HitSpecification* hitSpecs) const override;

	RTX_DEVICE b8 deleteMaterial();

	RTX_DEVICE static vector3 isRandomInUnitSphere(curandState* pRandState);

private:

	point3 center{ 0.f, 0.f, 0.f };
	Material* pMaterial{ nullptr };
	f32 radius{ 0.f };

};


RTX_DEVICE HittableSphere::HittableSphere(point3 _center, f32 _radius, Material* _pMaterial) :
	center(_center),
	radius(_radius),
	pMaterial(_pMaterial)
{ }

RTX_DEVICE b8 HittableSphere::hit(const Ray& ray, HitInterval interval, HitSpecification* hitSpecs) const {
	if (!hitSpecs) {
		return RTX_FALSE;
	}

	// Sphere Equation Calcation, looking for discriminant (delta). If no roots, then delta < 0,
	// delta = 0 is 1 root (ray only touches one sphere's point, delta > 0 is 2 roots (Ray pierces the sphere).
	const vector3 oc{ ray.origin - center };
	const f32 a{ vector3::dot(ray.direction, ray.direction) };
	const f32 b{ 2.f * vector3::dot(oc, ray.direction) };
	const f32 c{ vector3::dot(oc, oc) - radius * radius };
	const f32 delta{ b * b - 4 * a * c };
	if (delta < 0) {
		return RTX_FALSE;
	}

	// Checking if first root is between interval, if not validating second root. Applicable root stays in "root" variable.
	f32 root = (-b - sqrt(delta)) / (2.f * a);
	if (root < interval.min || interval.max < root) {
		root = (-b + sqrt(delta)) / (2.f * a);
		if (root < interval.min || interval.max < root) {
			return RTX_FALSE;
		}
	}

	// Saving hit specification to pointer.
	const point3 hitPoint{ ray.at(root) };
	const vector3 outwardNormal{ (hitPoint - center) / radius };
	hitSpecs->t = root;
	hitSpecs->point = hitPoint;
	hitSpecs->frontFace = vector3::dot(ray.direction, outwardNormal) < 0 ? RTX_TRUE : RTX_FALSE;
	hitSpecs->normal = hitSpecs->frontFace ? outwardNormal : (-1.f * outwardNormal);
	hitSpecs->pMaterial = pMaterial;
	return RTX_TRUE;
}

RTX_DEVICE b8 HittableSphere::deleteMaterial() {
	delete pMaterial;
	return RTX_TRUE;
}

RTX_DEVICE vector3 HittableSphere::isRandomInUnitSphere(curandState* pRandState) {
	vec3 p;
	do {
		p = 2.f * vec3::random(pRandState) - 1.f;
	} while (vec3::dot(p, p) >= 1.0f);
	return p;
}


#endif
