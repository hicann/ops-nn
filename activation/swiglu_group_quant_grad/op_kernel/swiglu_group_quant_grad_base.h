/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_grad_base.h
 * \brief SwiGLU Group Dynamic Quant Backward base class
 */

#ifndef SWIGLU_GROUP_QUANT_GRAD_BASE_H
#define SWIGLU_GROUP_QUANT_GRAD_BASE_H

#include "kernel_operator.h"

namespace SwigluGroupQuantGradOp {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SPLIT_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t FP32_32B_ALIGN_NUM = 8;
constexpr uint32_t CLAMP_BUFFER_INDEX = 8;
constexpr uint32_t SILU_GRAD_BUFFER_INDEX = 2;
constexpr uint32_t TMP_BUFFER_INDEX = 5;
constexpr uint32_t GRAD_WEIGHT_ACCUM_BUFFER_INDEX = 1;
constexpr uint32_t HAS_CLAMP_SCENE_TILE_NUM = 10;
constexpr uint32_t BASE_SCENE_TILE_NUM = 6;

class SwigluGroupQuantGradBase {
public:
    __aicore__ inline SwigluGroupQuantGradBase() {}

    __aicore__ inline void ParseTilingData(GM_ADDR tiling)
    {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantGradTilingData, tilingData, tiling);
        
        blockIdx = GetBlockIdx();
        coreNumAll = tilingData.coreNumAll;
        totalTokens = tilingData.totalTokens;
        dimH = tilingData.dimH;
        dim2H = tilingData.dim2H;
        groupNum = tilingData.groupNum;
        tileTokens = tilingData.tileTokens;
        tileH = tilingData.tileH;
        numHTiles = tilingData.numHTiles;
        hasWeight = tilingData.hasWeight;
        hasYOrigin = tilingData.hasYOrigin;
        hasGroupIndex = tilingData.hasGroupIndex;
        hasClampLimit = tilingData.hasClampLimit;
        clampLimit = tilingData.clampLimit;
        
        tileLength = tileTokens * tileH;
        tileDataSize = tileLength * sizeof(float);
    }

    __aicore__ inline void ComputeTruncRelatedParams()
    {
        usedCoreNum = (truncValue < coreNumAll) ? truncValue : coreNumAll;
        tokensPerCore = CeilDiv(truncValue, usedCoreNum);
        tokenStart = blockIdx * tokensPerCore;
    }

    __aicore__ inline void InitBuffer()
    {
        pipe.InitBuffer(gradYQueue, BUFFER_NUM, tileLength * sizeof(float));
        if (hasClampLimit) {
            pipe.InitBuffer(xQueue, BUFFER_NUM, tileLength * HAS_CLAMP_SCENE_TILE_NUM * sizeof(float));
        } else {
            pipe.InitBuffer(xQueue, BUFFER_NUM, tileLength * BASE_SCENE_TILE_NUM * sizeof(float));
        }
        
        if (hasWeight) {
            pipe.InitBuffer(weightQueue, BUFFER_NUM, AlignUp(tileTokens, FP32_32B_ALIGN_NUM) * sizeof(float) +
                                                     tileLength * sizeof(float));
            pipe.InitBuffer(yOriginQueue, BUFFER_NUM, tileLength * sizeof(float));
        }
    }

    __aicore__ inline void InitZeroOutTruncBuffer()
    {
        if (truncValue < totalTokens) {
            pipe.Reset();
            uint32_t zeroOutBufSize = (AlignUp(dim2H, FP32_32B_ALIGN_NUM) + FP32_32B_ALIGN_NUM) * sizeof(float);
            pipe.InitBuffer(zeroQueue, BUFFER_NUM, zeroOutBufSize);
        }
    }

    template <typename T>
    __aicore__ inline T CeilDiv(T x, T y)
    {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    template <typename T>
    __aicore__ inline T AlignUp(T num, T div)
    {
        return (div == 0) ? 0 : (num + div - 1) / div * div;
    }

    template <typename T>
    __aicore__ inline T AlignDown(T num, T div)
    {
        return (div == 0) ? 0 : num / div * div;
    }

protected:
    TPipe pipe;
    
    TQue<TPosition::VECIN, BUFFER_NUM> gradYQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> xQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> weightQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> yOriginQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> zeroQueue;
    
    uint32_t blockIdx = 0;
    uint32_t coreNumAll = 0;
    uint32_t totalTokens = 0;
    uint32_t dimH = 0;
    uint32_t dim2H = 0;
    uint32_t groupNum = 0;
    uint32_t truncValue = 0;
    uint32_t tileTokens = 0;
    uint32_t tileH = 0;
    uint32_t numHTiles = 0;
    uint32_t usedCoreNum = 0;
    uint32_t tokensPerCore = 0;
    uint32_t tokenStart = 0;
    uint32_t hasWeight = 0;
    uint32_t hasYOrigin = 0;
    uint32_t hasGroupIndex = 0;
    uint32_t hasClampLimit = 0;
    float clampLimit = -1.0f;
    
    uint32_t tileLength = 0;
    uint32_t tileDataSize = 0;
};

} // namespace SwigluGroupQuantGradOp

#endif // SWIGLU_GROUP_QUANT_GRAD_BASE_H