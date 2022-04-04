
#ifndef HITTABLELIST_H
#define HITTABLELIST_H


#include "HittableObjects.h"
#include "defines.h"
#include <initializer_list>
#include <vector>


struct Ray;


class HittableList : public HittableObject {
public:

	HittableList(std::initializer_list<HittableObject*> _objects);

	void add(HittableObject* object);
	void clear();

	b8 hit(const Ray& ray, HitInterval interval, HitSpecification* specs) const override;

private:

	std::vector<HittableObject*> objects;

};


#endif
