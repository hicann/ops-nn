#ifndef _TEST_GROUP_QUANT_H_
#define _TEST_GROUP_QUANT_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct GroupQuantTilingData {
    int64_t dimS{0};
    int64_t dimE{0};
    int64_t dimH{0};
    int64_t hasOffset{0};
    int64_t needCoreNum{0};
    int64_t preCoreNum{0};
    int64_t xRowNumPreCore{0};
    int64_t xRowNumPostCore{0};
};

#pragma pack()

#define DTYPE_X float
#define DTYPE_SCALE float
#define DTYPE_GROUP_INDEX int32_t
#define DTYPE_OFFSET float
#define DTYPE_Y int8_t
#define ORIG_DTYPE_X 0
#define ORIG_DTYPE_SCALE 0
#define ORIG_DTYPE_GROUP_INDEX 3
#define ORIG_DTYPE_OFFSET 0
#define ORIG_DTYPE_Y 2

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
    GroupQuantTilingData tilingData;                                          \
    INIT_TILING_DATA(GroupQuantTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).dimS = tilingDataPointer->dimS;                              \
    (tilingData).dimE = tilingDataPointer->dimE;                              \
    (tilingData).dimH = tilingDataPointer->dimH;                              \
    (tilingData).hasOffset = tilingDataPointer->hasOffset;                    \
    (tilingData).needCoreNum = tilingDataPointer->needCoreNum;                \
    (tilingData).preCoreNum = tilingDataPointer->preCoreNum;                  \
    (tilingData).xRowNumPreCore = tilingDataPointer->xRowNumPreCore;          \
    (tilingData).xRowNumPostCore = tilingDataPointer->xRowNumPostCore;
#endif