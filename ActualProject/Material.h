
#ifndef MATERIAL_H
#define MATERIAL_H


#include "defines.h"
#include "Ray.h"
#include "HittableObject.h"
#include "HittableSphere.h"


class Material {
public:

	RTX_DEVICE virtual b8 scatter(const Ray& ray, const HitSpecification& hitSpecs, color* pAttenuation,
								  Ray* pScattered, curandState* pRandState) const { return RTX_FALSE; }

	RTX_DEVICE virtual void setAlbedo(f32 r, f32 g, f32 b) {
		albedo = { r, g, b };
	}

protected:

	color albedo{ 0.f, 0.f, 0.f };

};


class Lambertian : public Material {
public:

	RTX_DEVICE b8 scatter(const Ray& ray, const HitSpecification& hitSpecs, color* pAttenuation, Ray* pScattered,
						  curandState* pRandState) const override {
		const vector3 scatterDirection{ hitSpecs.normal + vector3::normalize(HittableSphere::isRandomInUnitSphere(pRandState)) };
		*pScattered = Ray(hitSpecs.point, vector3::nearZero(scatterDirection) ? hitSpecs.normal : scatterDirection);
		*pAttenuation = albedo;
		return RTX_TRUE;
	}

};


class Metal : public Material {
public:

	RTX_DEVICE b8 scatter(const Ray& ray, const HitSpecification& hitSpecs, color* pAttenuation, Ray* pScattered,
						  curandState* pRandState) const override {
		const vector3 reflected{ vector3::reflect(vector3::normalize(ray.direction), hitSpecs.normal) };
		*pScattered = Ray(hitSpecs.point, reflected);
		*pAttenuation = albedo;
		return (vector3::dot(pScattered->direction, hitSpecs.normal) > 0) ? RTX_TRUE : RTX_FALSE;
	}

};


RTX_DEVICE b8 HittableSphere::deleteMaterial() {
	delete pMaterial;
	return RTX_TRUE;
}


#endif
