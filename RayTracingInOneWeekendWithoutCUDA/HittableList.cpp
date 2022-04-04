
#include "HittableList.h"


HittableList::HittableList(std::initializer_list<HittableObject*> _objects) :
	objects(_objects)
{ }


void HittableList::clear() {
	for (HittableObject* obj : objects) {
		obj->deleteMaterial();
		delete obj;
	}

	objects.clear();
}

void HittableList::add(HittableObject* object) {
	objects.push_back(object);
}

b8 HittableList::hit(const Ray& ray, HitInterval interval, HitSpecification* specs) const {
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