/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deformable_conv2d_tiling.h
 * \brief
 */

#ifndef DEFORMABLE_CONV2D_TILING_DEFH
#define DEFORMABLE_CONV2D_TILING_DEFH

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __aicore__
#define DTYPE_X float_t

struct DeformableConv2dTilingData {
    int64_t n;
    int64_t inC;
    int64_t inH;
    int64_t inW;
    int64_t outC;
    int64_t outH;
    int64_t outW;
    int64_t kH;
    int64_t kW;

    int64_t padTop;
    int64_t padLeft;
    int64_t strideH;
    int64_t strideW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t deformableGroups;
    int64_t groups;
    bool hasBias;

    int64_t slideSizeW;
    int64_t groupLen;
    int64_t singleVecNum;
    int64_t tailVecNum;

    TCubeTiling mmTilingData;
};

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                  \
    DeformableConv2dTilingData tilingData;                                          \
    INIT_TILING_DATA(DeformableConv2dTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).n = tilingDataPointer->n;                                          \
    (tilingData).inC = tilingDataPointer->inC;                                      \
    (tilingData).inH = tilingDataPointer->inH;                                      \
    (tilingData).inW = tilingDataPointer->inW;                                      \
    (tilingData).outC = tilingDataPointer->outC;                                    \
    (tilingData).outH = tilingDataPointer->outH;                                    \
    (tilingData).outW = tilingDataPointer->outW;                                    \
    (tilingData).kH = tilingDataPointer->kH;                                        \
    (tilingData).kW = tilingDataPointer->kW;                                        \
    (tilingData).padTop = tilingDataPointer->padTop;                                \
    (tilingData).padLeft = tilingDataPointer->padLeft;                              \
    (tilingData).strideH = tilingDataPointer->strideH;                              \
    (tilingData).strideW = tilingDataPointer->strideW;                              \
    (tilingData).dilationH = tilingDataPointer->dilationH;                          \
    (tilingData).dilationW = tilingDataPointer->dilationW;                          \
    (tilingData).deformableGroups = tilingDataPointer->deformableGroups;            \
    (tilingData).groups = tilingDataPointer->groups;                                \
    (tilingData).hasBias = tilingDataPointer->hasBias;                              \
    (tilingData).slideSizeW = tilingDataPointer->slideSizeW;                        \
    (tilingData).groupLen = tilingDataPointer->groupLen;                            \
    (tilingData).singleVecNum = tilingDataPointer->singleVecNum;                    \
    (tilingData).tailVecNum = tilingDataPointer->tailVecNum;                        \
    (tilingData).mmTilingData = tilingDataPointer->mmTilingData;

#endif  // DEFORMABLE_CONV2D_TILING_DEFH
