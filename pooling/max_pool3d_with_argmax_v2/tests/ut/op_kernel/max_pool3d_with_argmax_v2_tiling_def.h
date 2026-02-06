/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_with_argmax_v2_tiling.h
 * \brief
 */

#ifndef _GE_MAX_POOL3D_WITH_ARGMAX_V2_TILING_DEF_H_
#define _GE_MAX_POOL3D_WITH_ARGMAX_V2_TILING_DEF_H_
#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"
#include "securec.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define DTYPE_X float
#define DTYPE_ARGMAX int32_t

#pragma pack(1)

struct MaxPool3DWithArgmaxV2TilingData {
    uint32_t inputShapes[3] = {0, 0, 0};
    uint32_t outShapes[3] = {0, 0, 0};
    uint32_t kD;
    uint32_t kH;
    uint32_t kW;
    uint32_t sD;
    uint32_t sH;
    uint32_t sW;
    uint32_t pD;
    uint32_t pH;
    uint32_t pW;
    uint32_t dD;
    uint32_t dH;
    uint32_t dW;
    float minFloat;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2NoSplitTilingData {
    uint32_t inputShapes[3] = {0, 0, 0};
    uint32_t outShapes[3] = {0, 0, 0};
    uint32_t kD;
    uint32_t kH;
    uint32_t kW;
    uint32_t sD;
    uint32_t sH;
    uint32_t sW;
    uint32_t pD;
    uint32_t pH;
    uint32_t pW;
    uint32_t dD;
    uint32_t dH;
    uint32_t dW;
    uint32_t batchesPerCore;
    uint32_t leftOverBatches;
    uint32_t partD;
    uint32_t partH;
    uint32_t partW;
    float minFloat;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2SplitDTilingData {
    uint32_t inputShapes[3] = {0, 0, 0};
    uint32_t outShapes[3] = {0, 0, 0};
    uint32_t kD;
    uint32_t kH;
    uint32_t kW;
    uint32_t sD;
    uint32_t sH;
    uint32_t sW;
    uint32_t pD;
    uint32_t pH;
    uint32_t pW;
    uint32_t dD;
    uint32_t dH;
    uint32_t dW;
    uint32_t batchesPerCore;
    uint32_t leftOverBatches;
    uint32_t partsPerCore;
    uint32_t leftOverParts;
    uint32_t partD;
    uint32_t partH;
    uint32_t partW;
    float minFloat;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
    uint32_t splitPointDIn[50] = {0};
    uint32_t splitPointDOut[50] = {0};
    uint32_t sizeIn[50] = {0};
    uint32_t batchStart[50] = {0};
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2SplitHTilingData {
    uint32_t inputShapes[3] = {0, 0, 0};
    uint32_t outShapes[3] = {0, 0, 0};
    uint32_t kD;
    uint32_t kH;
    uint32_t kW;
    uint32_t sD;
    uint32_t sH;
    uint32_t sW;
    uint32_t pD;
    uint32_t pH;
    uint32_t pW;
    uint32_t dD;
    uint32_t dH;
    uint32_t dW;
    uint32_t batchesPerCore;
    uint32_t leftOverBatches;
    uint32_t partsPerCore;
    uint32_t leftOverParts;
    uint32_t partD;
    uint32_t partH;
    uint32_t partW;
    float minFloat;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
    uint32_t splitPointDIn[50] = {0};
    uint32_t splitPointHIn[50] = {0};
    uint32_t splitPointDOut[50] = {0};
    uint32_t splitPointHOut[50] = {0};
    uint32_t sizeIn[50] = {0};
    uint32_t batchStart[50] = {0};
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2SplitWTilingData {
    uint32_t inputShapes[3] = {0, 0, 0};
    uint32_t outShapes[3] = {0, 0, 0};
    uint32_t kD;
    uint32_t kH;
    uint32_t kW;
    uint32_t sD;
    uint32_t sH;
    uint32_t sW;
    uint32_t pD;
    uint32_t pH;
    uint32_t pW;
    uint32_t dD;
    uint32_t dH;
    uint32_t dW;
    uint32_t batchesPerCore;
    uint32_t leftOverBatches;
    uint32_t partsPerCore;
    uint32_t leftOverParts;
    uint32_t partD;
    uint32_t partH;
    uint32_t partW;
    float minFloat;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
    uint32_t splitPointDIn[50] = {0};
    uint32_t splitPointHIn[50] = {0};
    uint32_t splitPointWIn[50] = {0};
    uint32_t splitPointDOut[50] = {0};
    uint32_t splitPointHOut[50] = {0};
    uint32_t splitPointWOut[50] = {0};
    uint32_t sizeIn[50] = {0};
    uint32_t batchStart[50] = {0};
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2HugeKernelTilingData {
    uint32_t inputShapes[3] = {0, 0, 0};
    uint32_t outShapes[3] = {0, 0, 0};
    uint32_t kD;
    uint32_t kH;
    uint32_t kW;
    uint32_t sD;
    uint32_t sH;
    uint32_t sW;
    uint32_t pD;
    uint32_t pH;
    uint32_t pW;
    uint32_t dD;
    uint32_t dH;
    uint32_t dW;
    uint32_t batchesPerCore;
    uint32_t leftOverBatches;
    uint32_t partD;
    uint32_t partH;
    uint32_t partW;
    float minFloat;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2NoExpandIndicesTilingData {
    uint64_t nc;
    uint64_t dx;
    uint64_t hx;
    uint64_t wx;
    uint64_t kd;
    uint64_t kh;
    uint64_t kw;
    uint64_t sd;
    uint64_t sh;
    uint64_t sw;
    uint64_t pf;
    uint64_t pb;
    uint64_t pt;
    uint64_t pd;
    uint64_t pl;
    uint64_t pr;
    uint64_t dy;
    uint64_t hy;
    uint64_t wy;
    uint64_t ncFactor;
    uint64_t ncTail;
    uint64_t ncOuter;
    uint64_t dyFactor;
    uint64_t dyTail;
    uint64_t dyOuter;
    uint64_t hyFactor;
    uint64_t hyTail;
    uint64_t hyOuter;
    uint64_t wyFactor;
    uint64_t wyTail;
    uint64_t wyOuter;
    uint64_t blockFactor;
    uint64_t blockTail;
    uint64_t totalIdx;
    uint64_t coreNums;
    uint64_t inputBufferSize;
    uint64_t outputMaxBufferSize;
    uint64_t outputIndiceBufferSize;
    uint64_t indiceTempBufferSize;
    uint64_t maskBufferSize;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2BigKernelTilingData {
    uint32_t inputShapes[3] = {0, 0, 0};
    uint32_t outShapes[3] = {0, 0, 0};
    uint32_t kD;
    uint32_t kH;
    uint32_t kW;
    uint32_t sD;
    uint32_t sH;
    uint32_t sW;
    uint32_t pD;
    uint32_t pH;
    uint32_t pW;
    uint32_t dD;
    uint32_t dH;
    uint32_t dW;
    uint64_t blockFactor;
    uint64_t blockTail;
    uint64_t totalIdx;
    uint64_t coreNums;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2GatherTilingData {
    uint64_t dInput = 0;
    uint64_t hInput = 0;
    uint64_t wInput = 0;
    uint64_t dOutput = 0;
    uint64_t hOutput = 0;
    uint64_t wOutput = 0;
    uint64_t dKernel = 0;
    uint64_t hKernel = 0;
    uint64_t wKernel = 0;
    uint64_t dStride = 0;
    uint64_t hStride = 0;
    uint64_t wStride = 0;
    uint64_t padFront = 0;
    uint64_t padLeft = 0;
    uint64_t padTop = 0;
    uint64_t highAxisInner = 0;
    uint64_t highAxisTail = 0;
    uint64_t highAxisOuter = 0;
    uint64_t dOutputInner = 0;
    uint64_t dOutputTail = 0;
    uint64_t dOutputOuter = 0;
    uint64_t hOutputInner = 0;
    uint64_t hOutputTail = 0;
    uint64_t hOutputOuter = 0;
    uint64_t wOutputInner = 0;
    uint64_t wOutputTail = 0;
    uint64_t wOutputOuter = 0;
    uint64_t normalCoreProcessNum = 0;
    uint64_t tailCoreProcessNum = 0;
    uint64_t usedCoreNum = 0;
    uint64_t inputBufferSize = 0;
    uint64_t maxValueBufferSize = 0;
    uint64_t argmaxBufferSize = 0;
    uint64_t isPad = 0;
    uint64_t dDilation = 0;
    uint64_t hDilation = 0;
    uint64_t wDilation = 0;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData {
    int64_t dInDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t dOutDim = 0;
    int64_t hOutDim = 0;
    int64_t wOutDim = 0;
    int64_t kD = 0;
    int64_t kH = 0;
    int64_t kW = 0;
    int64_t sD = 0;
    int64_t sH = 0;
    int64_t sW = 0;
    int64_t pD = 0;
    int64_t pH = 0;
    int64_t pW = 0;
    int64_t dD = 0;
    int64_t dH = 0;
    int64_t dW = 0;
    int64_t blockFactor = 0;
    int64_t blockTail = 0;
    int64_t totalIdx = 0;
    int64_t coreNums = 0;
    int64_t maxCount = 0;
    int64_t isSigOut = 0;
};
#pragma pack()

#pragma pack(1)
struct MaxPool3DWithArgmaxV2SimtTilingData {
    uint64_t threadNums = 1;
    uint64_t blockNums = 1;
    uint64_t nDim = 0;
    uint64_t cDim = 0;
    uint64_t dInDim = 0;
    uint64_t hInDim = 0;
    uint64_t wInDim = 0;
    uint64_t dOutDim = 0;
    uint64_t hOutDim = 0;
    uint64_t wOutDim = 0;
    uint64_t kSizeD = 0;
    uint64_t kSizeH = 0;
    uint64_t kSizeW = 0;
    uint64_t stridesD = 0;
    uint64_t stridesH = 0;
    uint64_t stridesW = 0;
    uint64_t padD = 0;
    uint64_t padH = 0;
    uint64_t padW = 0;
    uint64_t dilationD = 0;
    uint64_t dilationH = 0;
    uint64_t dilationW = 0;
    uint64_t ceilMode = 0;
};
#pragma pack()

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2TilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2TilingData), tiling, sizeof(MaxPool3DWithArgmaxV2TilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2NoSplitTilingData* constData)
{
    memcpy_s(constData,sizeof(MaxPool3DWithArgmaxV2NoSplitTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2NoSplitTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2SplitDTilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2SplitDTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2SplitDTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2SplitHTilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2SplitHTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2SplitHTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2SplitWTilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2SplitWTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2SplitWTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2HugeKernelTilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2HugeKernelTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2HugeKernelTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2BigKernelTilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2GatherTilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2GatherTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2GatherTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* const_data)
{
    memcpy_s(const_data, sizeof(MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DWithArgmaxV2SimtTilingData* constData)
{
    memcpy_s(constData, sizeof(MaxPool3DWithArgmaxV2SimtTilingData), tiling, sizeof(MaxPool3DWithArgmaxV2SimtTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)  \
    MaxPool3DWithArgmaxV2TilingData tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)



#endif