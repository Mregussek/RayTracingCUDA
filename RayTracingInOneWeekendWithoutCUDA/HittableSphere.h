
#ifndef HITTABLESPHERE_H
#define HITTABLESPHERE_H


#include <iostream>
#include "HittableObjects.h"
#include "defines.h"
#include "vec3.h"
#include "Ray.h"


class Material;


class HittableSphere : public HittableObject {
public:

	constexpr HittableSphere() = default;
	constexpr HittableSphere(point3 _center, f32 _radius, Material* _pMaterial) :
		center(_center),
		radius(_radius),
		pMaterial(_pMaterial)
	{ }

	constexpr b8 hit(const Ray& ray, HitInterval interval, HitSpecification* hitSpecs) const override {
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
		hitSpecs->normal = hitSpecs->frontFace ? outwardNormal : -outwardNormal;
		hitSpecs->pMaterial = pMaterial;
		return RTX_TRUE;
	}

	b8 deleteMaterial() {
		if (pMaterial) {
			delete pMaterial;
			return RTX_TRUE;
		}
		return RTX_FALSE;
	}

	static vector3 isRandomInUnitSphere() {
		while (true) {
			const vector3 r{ vector3::random(-1.f, 1.f) };
			if (vector3::dot(r, r) >= 1) {
				continue;
			}
			return r;
		}
	}

private:

	point3 center{ 0.f, 0.f, 0.f };
	f32 radius{ 0.f };
	Material* pMaterial{ nullptr };

};


#endif
