
#ifndef HITTABLESPHERE_H
#define HITTABLESPHERE_H


#include <iostream>
#include "HittableObjects.h"
#include "defines.h"
#include "vec3.h"
#include "Ray.h"


class HittableSphere : public HittableObject {
public:

	constexpr HittableSphere() = default;
	constexpr HittableSphere(point3 _center, f32 _radius) :
		center(_center),
		radius(_radius)
	{ }

	constexpr b8 hit(const Ray& ray, HitInterval interval, HitSpecification* hitSpecs) const override {
		if (!hitSpecs) {
			return false;
		}

		// Sphere Equation Calcation, looking for discriminant (delta). If no roots, then delta < 0,
		// delta = 0 is 1 root (ray only touches one sphere's point, delta > 0 is 2 roots (Ray pierces the sphere).
		const vector3 oc{ ray.origin - center };
		const f32 a{ vector3::dot(ray.direction, ray.direction) };
		const f32 b{ 2.f * vector3::dot(oc, ray.direction) };
		const f32 c{ vector3::dot(oc, oc) - radius * radius };
		const f32 delta{ b * b - 4 * a * c };
		if (delta < 0) {
			return false;
		}

		// Checking if first root is between interval, if not validating second root. Applicable root stays in "root" variable.
		f32 root = (-b - sqrt(delta)) / (2.f * a);
		if (root < interval.min || interval.max < root) {
			root = (-b + sqrt(delta)) / (2.f * a);
			if (root < interval.min || interval.max < root) {
				return false;
			}
		}

		// Saving hit specification to pointer.
		const point3 hitPoint{ ray.at(root) };
		const vector3 outwardNormal{ (hitPoint - center) / radius };
		hitSpecs->t = root;
		hitSpecs->point = hitPoint;
		hitSpecs->frontFace = vector3::dot(ray.direction, outwardNormal) < 0;
		hitSpecs->normal = hitSpecs->frontFace ? outwardNormal : -outwardNormal;
		return true;
	}

private:

	point3 center{ 0.f, 0.f, 0.f };
	f32 radius{ 0.f };

};


#endif
