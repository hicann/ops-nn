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
 * \file avg_pool_struct.h
 * \brief tiling base data
 */

#ifndef AVG_POOL_STRUCT_H
#define AVG_POOL_STRUCT_H

namespace AvgPool {
class AvgPoolTilingData {
public:
    int64_t nDim;
    int64_t cDim;
    int64_t hInDim;
    int64_t wInDim;
    int64_t hOutDim;
    int64_t wOutDim;
    int64_t kH; // kernel
    int64_t kW;
    int64_t sH; // stride
    int64_t sW;
    int64_t dH;
    int64_t dW;
    int64_t tPad;
    int64_t bPad;
    int64_t lPad;
    int64_t rPad;
    int64_t divisorOverride;
    int64_t countIncludePad;
};

class AvgPoolBigKernelNhwcTilingData {
public:
    int64_t hInDim;
    int64_t wInDim;
    int64_t hOutDim;
    int64_t wOutDim;
    int64_t kH;
    int64_t kW;
    int64_t sH;
    int64_t sW;
    int64_t tPad;
    int64_t bPad;
    int64_t lPad;
    int64_t rPad;
    int64_t divisorOverride;
    int64_t countIncludePad;
    int64_t channel;
    int64_t blockFactor;
    int64_t blockTail;
    int64_t totalIdx;
    int64_t coreNums;
    int64_t inUbSize;
    int64_t outUbSize;
    int64_t isSigOut;
    int64_t tilingMode;
    int64_t onceOutNum;
};

class AvgPoolBigKernelTilingData{
public:
    int64_t hInDim;
    int64_t wInDim;
    int64_t hOutDim;
    int64_t wOutDim;
    int64_t kH;
    int64_t kW;
    int64_t sH;
    int64_t sW;
    int64_t tPad;
    int64_t bPad;
    int64_t lPad;
    int64_t rPad;
    int64_t divisorOverride;
    int64_t countIncludePad;
    int64_t blockFactor;
    int64_t blockTail;
    int64_t totalIdx;   
    int64_t coreNums;   
    int64_t maxCount;
    int64_t isSigOut;
};

class AvgPoolNHWCSmallKernelTilingData {
public:
    int64_t hInDim;
    int64_t wInDim;
    int64_t nOutDim;
    int64_t hOutDim;
    int64_t wOutDim;
    int64_t kW;
    int64_t kH;
    int64_t sW;
    int64_t sH;
    int64_t tPad;
    int64_t bottomPad;
    int64_t lPad;
    int64_t rPad;
    int64_t blockFactor;
    int64_t blockTail;
    int64_t ubFactorN;
    int64_t outUbFactorH;
    int64_t outUbFactorW;
    int64_t nLoop;
    int64_t hLoop;
    int64_t wLoop;
    int64_t channels;
    int64_t inUbSize;
    int64_t outUbSize;
    int64_t gatherMode;
    int64_t copyMode;
    int64_t onceCopyRow;
    int64_t splitMode;
    int64_t divisor;
    int64_t divisorUbSize;
    int32_t divisorMode;
    int32_t realCalcDivisor;
};

class AvgPoolSimtTilingData {
public:
    int64_t nDim;
    int64_t cDim;
    int64_t hInDim;
    int64_t wInDim;
    int64_t hOutDim;
    int64_t wOutDim;
    int64_t kH; // kernel
    int64_t kW;
    int64_t sH; // stride
    int64_t sW;
    int64_t dH;
    int64_t dW;
    int64_t tPad;
    int64_t bPad;
    int64_t lPad;
    int64_t rPad;
    int64_t divisorOverride;
    int64_t countIncludePad;
};

class AvgPoolNCHWSmallKernelTilingData {
public:
    int64_t hInDim;
    int64_t wInDim;
    int64_t nOutDim;
    int64_t hOutDim;
    int64_t wOutDim;
    int64_t kW;
    int64_t kH;
    int64_t sW;
    int64_t sH;
    int64_t tPad;
    int64_t bottomPad;
    int64_t lPad;
    int64_t rPad;
    int64_t blockFactor;
    int64_t blockTail;
    int64_t ubFactorN;
    int64_t outUbFactorH;
    int64_t outUbFactorW;
    int64_t nLoop;
    int64_t hLoop;
    int64_t wLoop;
    int64_t inUbSize;
    int64_t outUbSize;
    int64_t indiceUbSize;
    int64_t gatherMode;
    int64_t copyMode;
    int64_t onceCopyRow;
    int64_t splitMode;
    int64_t divisor;
    int64_t divisorMode;
    int64_t realCalcDivisor;
    int64_t divisorUbSize;
};

} // namespace AvgPool
#endif //AVG_POOL_STRUCT_H