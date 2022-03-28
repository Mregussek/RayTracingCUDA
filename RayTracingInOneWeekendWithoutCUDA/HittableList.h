
#ifndef HITTABLELIST_H
#define HITTABLELIST_H


#include "HittableObjects.h"
#include "defines.h"
#include <initializer_list>
#include <vector>


class HittableList : public HittableObject {
public:

	HittableList() = default;
	HittableList(std::initializer_list<HittableObject*> _objects) :
		objects(_objects)
	{ }


	void clear() {
		for (HittableObject* obj : objects) {
			obj->deleteMaterial();
			delete obj;
		}

		objects.clear();
	}

	void add(HittableObject* object) {
		objects.push_back(object);
	}

	b8 hit(const Ray& ray, HitInterval interval, HitSpecification* specs) const override {
		HitSpecification tmpSpecs;
		b8 hitAnything{ RTX_FALSE };
		f32 closestSoFar{ interval.max };

		for (const HittableObject* object : objects) {
			if (object->hit(ray, HitInterval{ interval.min, closestSoFar }, &tmpSpecs)) {
				hitAnything = RTX_TRUE;
				closestSoFar = tmpSpecs.t;
				*specs = tmpSpecs;
			}
		}

		return hitAnything;
	}

private:

	std::vector<HittableObject*> objects;

};


#endif
