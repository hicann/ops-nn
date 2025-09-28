/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_grad_with_argmax_tiling.h
 * \brief
 */

#ifndef _GE_MAX_POOL3D_GRAD_WITH_ARGMAX_TILING_DEF_H_
#define _GE_MAX_POOL3D_GRAD_WITH_ARGMAX_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct MaxPool3DGradWithArgmaxTilingData {
    uint64_t ncDim;
    uint64_t diDim;
    uint64_t hiDim;
    uint64_t wiDim;
    uint64_t doDim;
    uint64_t hoDim;
    uint64_t woDim;
    uint64_t kd;
    uint64_t kh;
    uint64_t kw;
    uint64_t sd;
    uint64_t sh;
    uint64_t sw;
    uint64_t padDTop;
    uint64_t padHTop;
    uint64_t padWTop;
    uint64_t padDBottom;
    uint64_t padHBottom;
    uint64_t padWBottom;
    uint64_t baseNc;
    uint64_t baseDo;
    uint64_t baseHo;
    uint64_t baseWo;
    uint64_t ncTail;
    uint64_t doTail;
    uint64_t hoTail;
    uint64_t woTail;
    uint64_t ncCnt;
    uint64_t doCnt;
    uint64_t hoCnt;
    uint64_t woCnt;
    uint64_t totalCnt;
    uint64_t usedCoreNum;
    uint64_t totalUBSize;

    // normal
    uint64_t singleCoreNc;
    uint64_t singleCoreDo;
    uint64_t singleCoreHo;
    uint64_t singleCoreWo;

    // scatter
    uint64_t ncRound;
    uint64_t ncRoundTail;
    uint64_t totalRound;
    uint64_t preCoreNum;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DGradWithArgmaxNoSplitTilingData {
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
    uint32_t partOutD;
    uint32_t partOutH;
    uint32_t partOutW;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DGradWithArgmaxSplitDTilingData {
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
    uint32_t partOutD;
    uint32_t partOutH;
    uint32_t partOutW;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DGradWithArgmaxSplitHTilingData {
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
    uint32_t partOutD;
    uint32_t partOutH;
    uint32_t partOutW;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
};

#pragma pack()

#pragma pack(1)

struct MaxPool3DGradWithArgmaxSplitWTilingData {
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
    uint32_t partOutD;
    uint32_t partOutH;
    uint32_t partOutW;
    uint32_t ceilD;
    uint32_t ceilH;
    uint32_t ceilW;
    uint32_t sizeUb1;
    uint32_t sizeUb2;
    uint32_t sizeValues;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                         \
    MaxPool3DGradWithArgmaxTilingData tilingData;                                          \
    INIT_TILING_DATA(MaxPool3DGradWithArgmaxTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).ncDim = tilingDataPointer->ncDim;                                         \
    (tilingData).diDim = tilingDataPointer->diDim;                                         \
    (tilingData).hiDim = tilingDataPointer->hiDim;                                         \
    (tilingData).wiDim = tilingDataPointer->wiDim;                                         \
    (tilingData).doDim = tilingDataPointer->doDim;                                         \
    (tilingData).hoDim = tilingDataPointer->hoDim;                                         \
    (tilingData).woDim = tilingDataPointer->woDim;                                         \
    (tilingData).kd = tilingDataPointer->kd;                                               \
    (tilingData).kh = tilingDataPointer->kh;                                               \
    (tilingData).kw = tilingDataPointer->kw;                                               \
    (tilingData).sd = tilingDataPointer->sd;                                               \
    (tilingData).sh = tilingDataPointer->sh;                                               \
    (tilingData).sw = tilingDataPointer->sw;                                               \
    (tilingData).padDTop = tilingDataPointer->padDTop;                                     \
    (tilingData).padHTop = tilingDataPointer->padHTop;                                     \
    (tilingData).padWTop = tilingDataPointer->padWTop;                                     \
    (tilingData).padDBottom = tilingDataPointer->padDBottom;                               \
    (tilingData).padHBottom = tilingDataPointer->padHBottom;                               \
    (tilingData).padWBottom = tilingDataPointer->padWBottom;                               \
    (tilingData).baseNc = tilingDataPointer->baseNc;                                       \
    (tilingData).baseDo = tilingDataPointer->baseDo;                                       \
    (tilingData).baseHo = tilingDataPointer->baseHo;                                       \
    (tilingData).baseWo = tilingDataPointer->baseWo;                                       \
    (tilingData).ncTail = tilingDataPointer->ncTail;                                       \
    (tilingData).doTail = tilingDataPointer->doTail;                                       \
    (tilingData).hoTail = tilingDataPointer->hoTail;                                       \
    (tilingData).woTail = tilingDataPointer->woTail;                                       \
    (tilingData).ncCnt = tilingDataPointer->ncCnt;                                         \
    (tilingData).doCnt = tilingDataPointer->doCnt;                                         \
    (tilingData).hoCnt = tilingDataPointer->hoCnt;                                         \
    (tilingData).woCnt = tilingDataPointer->woCnt;                                         \
    (tilingData).totalCnt = tilingDataPointer->totalCnt;                                   \
    (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                             \
    (tilingData).totalUBSize = tilingDataPointer->totalUBSize;                             \
    (tilingData).singleCoreNc = tilingDataPointer->singleCoreNc;                           \
    (tilingData).singleCoreDo = tilingDataPointer->singleCoreDo;                           \
    (tilingData).singleCoreHo = tilingDataPointer->singleCoreHo;                           \
    (tilingData).singleCoreWo = tilingDataPointer->singleCoreWo;                           \
    (tilingData).ncRound = tilingDataPointer->ncRound;                                     \
    (tilingData).ncRoundTail = tilingDataPointer->ncRoundTail;                             \
    (tilingData).totalRound = tilingDataPointer->totalRound;                               \
    (tilingData).preCoreNum = tilingDataPointer->preCoreNum;

inline void InitTilingData(uint8_t* tiling, MaxPool3DGradWithArgmaxNoSplitTilingData* constData)
{
    memcpy(constData, tiling, sizeof(MaxPool3DGradWithArgmaxNoSplitTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DGradWithArgmaxSplitDTilingData* constData)
{
    memcpy(constData, tiling, sizeof(MaxPool3DGradWithArgmaxSplitDTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DGradWithArgmaxSplitHTilingData* constData)
{
    memcpy(constData, tiling, sizeof(MaxPool3DGradWithArgmaxSplitHTilingData));
}

inline void InitTilingData(uint8_t* tiling, MaxPool3DGradWithArgmaxSplitWTilingData* constData)
{
    memcpy(constData, tiling, sizeof(MaxPool3DGradWithArgmaxSplitWTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float
#define DTYPE_GRAD float
#define DTYPE_ARGMAX int32_t
#define DTYPE_Y float

#endif