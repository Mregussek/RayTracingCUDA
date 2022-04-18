
#ifndef HITTABLELIST_H
#define HITTABLELIST_H


#include "HittableObject.h"
#include "defines.h"
#include "Ray.h"


class HittableList : public HittableObject {
public:

	RTX_DEVICE HittableList(HittableObject** _pList, u32 count);

	RTX_DEVICE void clear();

	RTX_DEVICE b8 hit(const Ray& ray, HitInterval interval, HitSpecification* specs) const override;

private:

	HittableObject** pList;
	u32 countList;

};


RTX_DEVICE HittableList::HittableList(HittableObject** _pList, u32 count) :
	pList(_pList),
	countList(count)
{ }


RTX_DEVICE void HittableList::clear() {
	for (u32 i = 0; i < countList; i++) {
		delete (*pList + i);
	}
}


RTX_DEVICE b8 HittableList::hit(const Ray& ray, HitInterval interval, HitSpecification* specs) const {
	HitSpecification tmpSpecs{};
	b8 hitAnything{ RTX_FALSE };
	f32 closestSoFar{ interval.max };

	for (u32 i = 0; i < countList; i++) {
		if (pList[i]->hit(ray, HitInterval{interval.min, closestSoFar}, &tmpSpecs)) {
			hitAnything = RTX_TRUE;
			closestSoFar = tmpSpecs.t;
			*specs = tmpSpecs;
		}
	}

	return hitAnything;
}


#endif
