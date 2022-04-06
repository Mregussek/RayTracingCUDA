
#ifndef BLOCKS_H
#define BLOCKS_H


#include "defines.h"
#include "Image.h"


struct BlocksSpecification {

    u32 x{ 0 };
    u32 y{ 0 };

};


class Blocks {
public:

    void initialize(BlocksSpecification _blockSpecs, Image* pImage) {
        blockSpecs = _blockSpecs;
        blocks = dim3{ pImage->getWidth() / blockSpecs.x + 1, pImage->getHeight() / blockSpecs.y + 1 };
        threads = dim3{ blockSpecs.x, blockSpecs.y };
    }

    i32 getWidth() const { return blockSpecs.x; }
    i32 getHeight() const { return blockSpecs.y; }
    dim3 getBlocks() const { return blocks; }
    dim3 getThreads() const { return threads; }

private:

    BlocksSpecification blockSpecs;
    dim3 blocks{ 0, 0 };
    dim3 threads{ 0, 0 };

};


#endif
