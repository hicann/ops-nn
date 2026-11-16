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
 * \file quantize_add_layer_norm_regbase_stats.h
 * \brief ascend950 (arch35/regbase) mean/var statistics helpers for QuantizeAddLayerNorm,
 *        split from quantize_add_layer_norm_regbase_helper.h (header-size compliance).
 *        The VFCalcMeanVar binaryAdd two-pass pipeline is decomposed into per-row VF phases
 *        plus callee helpers. All LocalMemBar fences stay inside the VF they synchronize.
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_REGBASE_STATS_H
#define QUANTIZE_ADD_LAYER_NORM_REGBASE_STATS_H

#include "quantize_add_layer_norm_regbase_helper.h"

namespace NsQuantizeAddLayerNorm {

template <typename X1_TYPE>
struct MeanVarLocalAddr {
    __ubuf__ X1_TYPE* x1Addr;
    __ubuf__ X1_TYPE* x2Addr;
    __ubuf__ X1_TYPE* biasAddr;
    __ubuf__ X1_TYPE* xOutAddr;
    __ubuf__ float* x32Addr;
    __ubuf__ float* meanAddr;
    __ubuf__ float* varAddr;
};

template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline MeanVarLocalAddr<X1_TYPE> GetMeanVarLocalAddr(
    LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local, LocalTensor<X1_TYPE>& biasLocal,
    LocalTensor<X1_TYPE>& xOutLocal, LocalTensor<float>& x32Local, LocalTensor<float>& meanLocal,
    LocalTensor<float>& varLocal)
{
    MeanVarLocalAddr<X1_TYPE> localAddr;
    localAddr.x1Addr = (__ubuf__ X1_TYPE*)x1Local[0].GetPhyAddr();
    localAddr.x2Addr = (__ubuf__ X1_TYPE*)x2Local[0].GetPhyAddr();
    localAddr.biasAddr = nullptr;
    if constexpr ((IS_BIAS_ELEWISE || IS_BIAS_BROADCAST)) {
        localAddr.biasAddr = (__ubuf__ X1_TYPE*)biasLocal[0].GetPhyAddr();
    }
    localAddr.xOutAddr = (__ubuf__ X1_TYPE*)xOutLocal[0].GetPhyAddr();
    localAddr.x32Addr = (__ubuf__ float*)x32Local[0].GetPhyAddr();
    localAddr.meanAddr = (__ubuf__ float*)meanLocal[0].GetPhyAddr();
    localAddr.varAddr = (__ubuf__ float*)varLocal[0].GetPhyAddr();
    return localAddr;
}

// One rowsCount iteration of VFCalcMeanVarFast: add-bias row load, xOut/x32 stores, then the
// Two-pass mean/variance reduction. xFactor/mean/y/yFactor/var live entirely inside the iteration.
template <typename X1_TYPE, int32_t TILING_KEY>
__simd_callee__ inline void MeanVarFastChunk(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X1_TYPE* x2Addr,
                                             __ubuf__ X1_TYPE* biasAddr, __ubuf__ X1_TYPE* xOutAddr,
                                             __ubuf__ float* x32Addr, __ubuf__ float* meanAddr, __ubuf__ float* varAddr,
                                             RegTensor<float>& x, RegTensor<float>& colsNum, MaskReg& pregMain,
                                             MaskReg& pregMerge, MaskReg& pregLoop, uint16_t row, uint32_t rowStride)
{
    RegTensor<float> xFactor;
    RegTensor<float> mean;
    RegTensor<float> y;
    RegTensor<float> yFactor;
    RegTensor<float> var;
    if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(x1Addr, x2Addr, biasAddr, x, pregLoop,
                                                                            static_cast<uint32_t>(row) * rowStride,
                                                                            static_cast<uint32_t>(row) * rowStride, 0);
    } else {
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, x, pregLoop, static_cast<uint32_t>(row) * rowStride,
            static_cast<uint32_t>(row) * rowStride, static_cast<uint32_t>(row) * rowStride);
    }
    StoreFp32ToOutput(xOutAddr, x, pregLoop, static_cast<uint32_t>(row) * rowStride);
    StoreAlign(x32Addr + static_cast<uint32_t>(row) * rowStride, x, pregLoop);
    Div<float, &DIV_HIGH_PREC_MODE>(xFactor, x, colsNum, pregLoop);
    ReduceSum(mean, xFactor, pregLoop);

    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanAddr + row, mean, pregMerge);

    Duplicate(mean, mean, pregMain);
    Muls(mean, mean, (float)-1.0, pregMain);
    // xDelta = x - mean.
    Add(x, x, mean, pregLoop);
    Mul(y, x, x, pregLoop);
    Div<float, &DIV_HIGH_PREC_MODE>(yFactor, y, colsNum, pregLoop);
    ReduceSum(var, yFactor, pregLoop);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varAddr + row, var, pregMerge);
}

template <typename X1_TYPE, int32_t TILING_KEY>
__simd_vf__ inline void MeanVarFastVf(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X1_TYPE* x2Addr, __ubuf__ X1_TYPE* biasAddr,
                                      __ubuf__ X1_TYPE* xOutAddr, __ubuf__ float* x32Addr, __ubuf__ float* meanAddr,
                                      __ubuf__ float* varAddr, uint16_t rowsCount, uint32_t colsPerLoop,
                                      uint32_t rowStride, float colsNumFp)
{
    RegTensor<float> x;
    RegTensor<float> colsNum;
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
    uint32_t remaining = colsPerLoop;
    MaskReg pregLoop = UpdateMask<float>(remaining);

    Duplicate(colsNum, colsNumFp, pregMain);
    for (uint16_t row = 0; row < rowsCount; row++) {
        MeanVarFastChunk<X1_TYPE, TILING_KEY>(x1Addr, x2Addr, biasAddr, xOutAddr, x32Addr, meanAddr, varAddr, x,
                                              colsNum, pregMain, pregMerge, pregLoop, row, rowStride);
    }
}

// AddLayerNormCommon
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFCalcMeanVarFast(LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local,
                                         LocalTensor<X1_TYPE>& biasLocal, LocalTensor<X1_TYPE>& xOutLocal,
                                         LocalTensor<float>& x32Local, LocalTensor<float>& meanLocal,
                                         LocalTensor<float>& varLocal, uint16_t rowsCount, uint32_t colsPerLoop,
                                         uint32_t colsPerLoopAlign)
{
    // Set phy addr
    MeanVarLocalAddr<X1_TYPE> localAddr = GetMeanVarLocalAddr<X1_TYPE, TILING_KEY>(
        x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal, varLocal);

    float colsNumFp = static_cast<float>(colsPerLoop);
    asc_vf_call<MeanVarFastVf<X1_TYPE, TILING_KEY>>(
        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, localAddr.xOutAddr, localAddr.x32Addr,
        localAddr.meanAddr, localAddr.varAddr, rowsCount, colsPerLoop, colsPerLoopAlign, colsNumFp);
}

// Mean pass, non-last binaryAddRemainder iteration: dual half-chunks (Q at offset, R at
// offset + binaryAddOffset), both masked with pregLoop.
template <typename X1_TYPE, int32_t TILING_KEY>
__simd_callee__ inline void MeanRemainderChunk(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X1_TYPE* x2Addr,
                                               __ubuf__ X1_TYPE* biasAddr, __ubuf__ X1_TYPE* xOutAddr,
                                               __ubuf__ float* x32Addr, __ubuf__ float* binaryAddAddr,
                                               RegTensor<float>& binaryAddQ, RegTensor<float>& binaryAddR,
                                               RegTensor<float>& vlMean, RegTensor<float>& colsNum, MaskReg& pregLoop,
                                               MaskReg& pregMerge, uint16_t i, uint16_t row, uint32_t rowStride,
                                               uint32_t vlFp32, uint32_t binaryAddOffset)
{
    if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, binaryAddQ, pregLoop,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride,
            static_cast<uint32_t>(i) * vlFp32);
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride + binaryAddOffset,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride + binaryAddOffset,
            static_cast<uint32_t>(i) * vlFp32 + binaryAddOffset);
    } else {
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, binaryAddQ, pregLoop,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride);
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride + binaryAddOffset,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride + binaryAddOffset,
            static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride + binaryAddOffset);
    }
    uint32_t baseOffset = static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride;
    StoreFp32ToOutput(xOutAddr, binaryAddQ, pregLoop, baseOffset);
    StoreFp32ToOutput(xOutAddr, binaryAddR, pregLoop, baseOffset + binaryAddOffset);
    StoreAlign(x32Addr + baseOffset, binaryAddQ, pregLoop);
    StoreAlign(x32Addr + baseOffset + binaryAddOffset, binaryAddR, pregLoop);

    Div<float, &DIV_HIGH_PREC_MODE>(binaryAddQ, binaryAddQ, colsNum, pregLoop);
    Div<float, &DIV_HIGH_PREC_MODE>(binaryAddR, binaryAddR, colsNum, pregLoop);

    Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
    ReduceSum(vlMean, binaryAddQ, pregLoop);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + i), vlMean, pregMerge);
}

// Mean pass, last binaryAddRemainder iteration: Q side runs full-lane (pregMain), R side keeps pregLoop.
template <typename X1_TYPE, int32_t TILING_KEY>
__simd_callee__ inline void MeanRemainderTailChunk(
    __ubuf__ X1_TYPE* x1Addr, __ubuf__ X1_TYPE* x2Addr, __ubuf__ X1_TYPE* biasAddr, __ubuf__ X1_TYPE* xOutAddr,
    __ubuf__ float* x32Addr, __ubuf__ float* binaryAddAddr, RegTensor<float>& binaryAddQ, RegTensor<float>& binaryAddR,
    RegTensor<float>& vlMean, RegTensor<float>& colsNum, MaskReg& pregMain, MaskReg& pregLoop, MaskReg& pregMerge,
    uint16_t row, uint32_t binaryAddRemainder, uint16_t binaryAddRemainderLoop, uint32_t rowStride, uint32_t vlFp32,
    uint32_t binaryAddOffset)
{
    // Tail chunk of the R (remainder) stream: only the lanes below the row end are valid.
    // Explicit count instead of re-using binaryAddRemainder: UpdateMask(count>=vlFp32) yields a
    // full-lane mask, so the tail chunk would read/store [cols, cols+pad) lanes out of bounds.
    uint32_t tailCnt = binaryAddRemainder - static_cast<uint32_t>(binaryAddRemainderLoop - 1) * vlFp32;
    pregLoop = UpdateMask<float>(tailCnt);
    uint32_t baseOffset = static_cast<uint32_t>(binaryAddRemainderLoop - 1) * vlFp32 +
                          static_cast<uint32_t>(row) * rowStride;
    if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, binaryAddQ, pregMain, baseOffset, baseOffset,
            static_cast<uint32_t>(binaryAddRemainderLoop - 1) * vlFp32);
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop, baseOffset + binaryAddOffset, baseOffset + binaryAddOffset,
            static_cast<uint32_t>(binaryAddRemainderLoop - 1) * vlFp32 + binaryAddOffset);
    } else {
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, binaryAddQ, pregMain, baseOffset, baseOffset, baseOffset);
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop, baseOffset + binaryAddOffset, baseOffset + binaryAddOffset,
            baseOffset + binaryAddOffset);
    }
    StoreFp32ToOutput(xOutAddr, binaryAddQ, pregMain, baseOffset);
    StoreFp32ToOutput(xOutAddr, binaryAddR, pregLoop, baseOffset + binaryAddOffset);
    StoreAlign(x32Addr + baseOffset, binaryAddQ, pregMain);
    StoreAlign(x32Addr + baseOffset + binaryAddOffset, binaryAddR, pregLoop);

    Div<float, &DIV_HIGH_PREC_MODE>(binaryAddQ, binaryAddQ, colsNum, pregMain);
    Div<float, &DIV_HIGH_PREC_MODE>(binaryAddR, binaryAddR, colsNum, pregLoop);

    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
    ReduceSum(vlMean, binaryAddQ, pregMain);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop - 1),
                                                         vlMean, pregMerge);
}

// Mean pass, quotient iterations beyond the remainder region: single stream, full-lane mask.
template <typename X1_TYPE, int32_t TILING_KEY>
__simd_callee__ inline void MeanQuotientChunk(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X1_TYPE* x2Addr,
                                              __ubuf__ X1_TYPE* biasAddr, __ubuf__ X1_TYPE* xOutAddr,
                                              __ubuf__ float* x32Addr, __ubuf__ float* binaryAddAddr,
                                              RegTensor<float>& vlMean, RegTensor<float>& colsNum, MaskReg& pregMain,
                                              MaskReg& pregMerge, uint16_t i, uint16_t row,
                                              uint16_t binaryAddRemainderLoop, uint32_t rowStride, uint32_t vlFp32)
{
    RegTensor<float> x;
    uint32_t baseOffset = static_cast<uint32_t>(i + binaryAddRemainderLoop) * vlFp32 +
                          static_cast<uint32_t>(row) * rowStride;
    if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            x1Addr, x2Addr, biasAddr, x, pregMain, baseOffset, baseOffset,
            static_cast<uint32_t>(i + binaryAddRemainderLoop) * vlFp32);
    } else {
        LoadInputsToRegResidualFirst<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(x1Addr, x2Addr, biasAddr, x, pregMain,
                                                                            baseOffset, baseOffset, baseOffset);
    }
    StoreFp32ToOutput(xOutAddr, x, pregMain, baseOffset);
    StoreAlign(x32Addr + baseOffset, x, pregMain);

    Div<float, &DIV_HIGH_PREC_MODE>(x, x, colsNum, pregMain);
    ReduceSum(vlMean, x, pregMain);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i),
                                                         vlMean, pregMerge);
}

// Mean-pass dichotomy tree reduction. This callee runs inside the owning mean VF.
__simd_callee__ inline void DichotomyHalfMean(__ubuf__ float* binaryAddAddr, RegTensor<float>& binaryAddQ,
                                              RegTensor<float>& binaryAddR, MaskReg& pregMain,
                                              uint16_t binaryAddLoopMean, uint16_t binaryAddKLoop, uint32_t vlFp32)
{
    uint16_t curBinaryAddLoopMean = binaryAddLoopMean;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddLoopMean = curBinaryAddLoopMean / 2;
        for (uint16_t j = 0; j < curBinaryAddLoopMean; j++) {
            LoadAlign(binaryAddQ, ((__ubuf__ float*)binaryAddAddr + j * vlFp32));
            LoadAlign(binaryAddR, ((__ubuf__ float*)binaryAddAddr + (j + curBinaryAddLoopMean) * vlFp32));
            Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
            StoreAlign(((__ubuf__ float*)binaryAddAddr + j * vlFp32), binaryAddQ, pregMain);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

// Var pass, non-last binaryAddRemainder iteration: re-loads the centered x32 halves.
__simd_callee__ inline void VarRemainderChunk(__ubuf__ float* x32Addr, __ubuf__ float* binaryAddAddr,
                                              RegTensor<float>& binaryAddQ, RegTensor<float>& binaryAddR,
                                              RegTensor<float>& binaryAddQPow, RegTensor<float>& binaryAddRPow,
                                              RegTensor<float>& vlVar, RegTensor<float>& mean,
                                              RegTensor<float>& colsNum, MaskReg& pregLoop, MaskReg& pregMerge,
                                              uint16_t i, uint16_t row, uint32_t rowStride, uint32_t vlFp32,
                                              uint32_t binaryAddOffset)
{
    uint32_t baseOffset = static_cast<uint32_t>(i) * vlFp32 + static_cast<uint32_t>(row) * rowStride;
    LoadAlign(binaryAddQ, x32Addr + baseOffset);
    LoadAlign(binaryAddR, x32Addr + baseOffset + binaryAddOffset);
    Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
    Sub(binaryAddR, binaryAddR, mean, pregLoop);
    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);

    Div<float, &DIV_HIGH_PREC_MODE>(binaryAddQPow, binaryAddQPow, colsNum, pregLoop);
    Div<float, &DIV_HIGH_PREC_MODE>(binaryAddRPow, binaryAddRPow, colsNum, pregLoop);

    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
    ReduceSum(vlVar, binaryAddQPow, pregLoop);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + i), vlVar, pregMerge);
}

// Var pass, last binaryAddRemainder iteration: Q side full-lane (pregMain), R side pregLoop.
__simd_callee__ inline void VarRemainderTailChunk(__ubuf__ float* x32Addr, __ubuf__ float* binaryAddAddr,
                                                  RegTensor<float>& binaryAddQ, RegTensor<float>& binaryAddR,
                                                  RegTensor<float>& binaryAddQPow, RegTensor<float>& binaryAddRPow,
                                                  RegTensor<float>& vlVar, RegTensor<float>& mean,
                                                  RegTensor<float>& colsNum, MaskReg& pregMain, MaskReg& pregLoop,
                                                  MaskReg& pregMerge, uint16_t row, uint32_t binaryAddRemainder,
                                                  uint16_t binaryAddRemainderLoop, uint32_t rowStride, uint32_t vlFp32,
                                                  uint32_t binaryAddOffset)
{
    // Same tail-mask correction as MeanRemainderTailChunk: R stream covers only
    // remainder-(remainderLoop-1)*vlFp32 valid lanes in its last chunk.
    uint32_t tailCnt = binaryAddRemainder - static_cast<uint32_t>(binaryAddRemainderLoop - 1) * vlFp32;
    pregLoop = UpdateMask<float>(tailCnt);
    uint32_t baseOffset = static_cast<uint32_t>(binaryAddRemainderLoop - 1) * vlFp32 +
                          static_cast<uint32_t>(row) * rowStride;
    LoadAlign(binaryAddQ, x32Addr + baseOffset);
    LoadAlign(binaryAddR, x32Addr + baseOffset + binaryAddOffset);
    Sub(binaryAddQ, binaryAddQ, mean, pregMain);
    Sub(binaryAddR, binaryAddR, mean, pregLoop);
    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregMain);
    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);

    Div<float, &DIV_HIGH_PREC_MODE>(binaryAddQPow, binaryAddQPow, colsNum, pregMain);
    Div<float, &DIV_HIGH_PREC_MODE>(binaryAddRPow, binaryAddRPow, colsNum, pregLoop);

    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregMain);
    ReduceSum(vlVar, binaryAddQPow, pregMain);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop - 1),
                                                         vlVar, pregMerge);
}

// Var pass, quotient iterations beyond the remainder region.
__simd_callee__ inline void VarQuotientChunk(__ubuf__ float* x32Addr, __ubuf__ float* binaryAddAddr,
                                             RegTensor<float>& vlVar, RegTensor<float>& mean, RegTensor<float>& colsNum,
                                             MaskReg& pregMain, MaskReg& pregMerge, uint16_t i, uint16_t row,
                                             uint16_t binaryAddRemainderLoop, uint32_t rowStride, uint32_t vlFp32)
{
    RegTensor<float> x1;
    RegTensor<float> y1;
    RegTensor<float> y1Pow;
    uint32_t baseOffset = static_cast<uint32_t>(i + binaryAddRemainderLoop) * vlFp32 +
                          static_cast<uint32_t>(row) * rowStride;
    LoadAlign(x1, x32Addr + baseOffset);
    Sub(y1, x1, mean, pregMain);
    Mul(y1Pow, y1, y1, pregMain);
    Div<float, &DIV_HIGH_PREC_MODE>(y1Pow, y1Pow, colsNum, pregMain);
    ReduceSum(vlVar, y1Pow, pregMain);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i),
                                                         vlVar, pregMerge);
}

// Var-pass dichotomy tree reduction. This callee runs inside the owning variance VF.
__simd_callee__ inline void DichotomyHalfVar(__ubuf__ float* binaryAddAddr, RegTensor<float>& binaryAddQ,
                                             RegTensor<float>& binaryAddR, MaskReg& pregMain, uint16_t binaryAddLoopVar,
                                             uint16_t binaryAddKLoop, uint32_t vlFp32)
{
    uint16_t curBinaryAddLoopVar = binaryAddLoopVar;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddLoopVar = curBinaryAddLoopVar / 2;
        for (uint16_t j = 0; j < curBinaryAddLoopVar; j++) {
            LoadAlign(binaryAddQ, ((__ubuf__ float*)binaryAddAddr + j * vlFp32));
            LoadAlign(binaryAddR, ((__ubuf__ float*)binaryAddAddr + (j + curBinaryAddLoopVar) * vlFp32));
            Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
            StoreAlign(((__ubuf__ float*)binaryAddAddr + j * vlFp32), binaryAddQ, pregMain);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

// Per-row mean phase of VFCalcMeanVar: partial sums -> dichotomy tree -> final mean store.
template <typename X1_TYPE, int32_t TILING_KEY>
__simd_vf__ inline void MeanPhaseVf(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X1_TYPE* x2Addr, __ubuf__ X1_TYPE* biasAddr,
                                    __ubuf__ X1_TYPE* xOutAddr, __ubuf__ float* x32Addr, __ubuf__ float* meanAddr,
                                    __ubuf__ float* binaryAddAddr, uint16_t row, float colsNumFp,
                                    uint32_t binaryAddRemainder, uint16_t binaryAddRemainderLoop,
                                    uint16_t binaryAddQuotientLoop, uint16_t binaryAddLoopMean, uint16_t binaryAddKLoop,
                                    uint32_t binaryAddOffset, uint32_t binaryAddLastNum, uint32_t rowStride,
                                    uint32_t vlFp32)
{
    RegTensor<float> meanTemp;
    RegTensor<float> mean;
    RegTensor<float> binaryAddQ;
    RegTensor<float> binaryAddR;
    RegTensor<float> vlMean;
    RegTensor<float> colsNum;

    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
    MaskReg pregLoop;
    uint32_t remaining = binaryAddRemainder;

    Duplicate(colsNum, colsNumFp, pregMain);
    for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
        pregLoop = UpdateMask<float>(remaining);
        MeanRemainderChunk<X1_TYPE, TILING_KEY>(x1Addr, x2Addr, biasAddr, xOutAddr, x32Addr, binaryAddAddr, binaryAddQ,
                                                binaryAddR, vlMean, colsNum, pregLoop, pregMerge, i, row, rowStride,
                                                vlFp32, binaryAddOffset);
    }
    MeanRemainderTailChunk<X1_TYPE, TILING_KEY>(
        x1Addr, x2Addr, biasAddr, xOutAddr, x32Addr, binaryAddAddr, binaryAddQ, binaryAddR, vlMean, colsNum, pregMain,
        pregLoop, pregMerge, row, binaryAddRemainder, binaryAddRemainderLoop, rowStride, vlFp32, binaryAddOffset);
    for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
        MeanQuotientChunk<X1_TYPE, TILING_KEY>(x1Addr, x2Addr, biasAddr, xOutAddr, x32Addr, binaryAddAddr, vlMean,
                                               colsNum, pregMain, pregMerge, i, row, binaryAddRemainderLoop, rowStride,
                                               vlFp32);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    DichotomyHalfMean(binaryAddAddr, binaryAddQ, binaryAddR, pregMain, binaryAddLoopMean, binaryAddKLoop, vlFp32);
    uint32_t meanLastMaskLen = binaryAddLastNum;
    pregLoop = UpdateMask<float>(meanLastMaskLen);
    LoadAlign(meanTemp, binaryAddAddr);
    ReduceSum(mean, meanTemp, pregLoop);

    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanAddr + row, mean, pregMerge);
    Duplicate(mean, mean, pregMain);
    LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
}

template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFCalcMeanPhase(const MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr,
                                       uint16_t row, float colsNumFp, int64_t binaryAddRemainder,
                                       uint16_t binaryAddRemainderLoop, uint16_t binaryAddQuotientLoop,
                                       uint16_t binaryAddLoopMean, uint16_t binaryAddKLoop, uint32_t binaryAddOffset,
                                       uint64_t binaryAddLastNum, uint32_t colsPerLoopAlign, uint32_t vlFp32)
{
    asc_vf_call<MeanPhaseVf<X1_TYPE, TILING_KEY>>(
        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, localAddr.xOutAddr, localAddr.x32Addr,
        localAddr.meanAddr, binaryAddAddr, row, colsNumFp, static_cast<uint32_t>(binaryAddRemainder),
        binaryAddRemainderLoop, binaryAddQuotientLoop, binaryAddLoopMean, binaryAddKLoop, binaryAddOffset,
        static_cast<uint32_t>(binaryAddLastNum), colsPerLoopAlign, vlFp32);
}

// Per-row variance phase of VFCalcMeanVar. mean is broadcast back from the mean phase's
// lane-0 store (DIST_BRC_B32 load, same mechanism as the full_load VFCalcYQuant mean/rstd loads).
__simd_vf__ inline void VarPhaseVf(__ubuf__ float* x32Addr, __ubuf__ float* meanAddr, __ubuf__ float* varAddr,
                                   __ubuf__ float* binaryAddAddr, uint16_t row, float colsNumFp,
                                   uint32_t binaryAddRemainder, uint16_t binaryAddRemainderLoop,
                                   uint16_t binaryAddQuotientLoop, uint16_t binaryAddLoopVar, uint16_t binaryAddKLoop,
                                   uint32_t binaryAddOffset, uint32_t binaryAddLastNum, uint32_t rowStride,
                                   uint32_t vlFp32)
{
    RegTensor<float> binaryAddQ;
    RegTensor<float> binaryAddR;
    RegTensor<float> binaryAddQPow;
    RegTensor<float> binaryAddRPow;
    RegTensor<float> vlVar;
    RegTensor<float> varTemp;
    RegTensor<float> var;
    RegTensor<float> mean;
    RegTensor<float> colsNum;

    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
    MaskReg pregLoop;

    Duplicate(colsNum, colsNumFp, pregMain);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, meanAddr + row);

    uint32_t remaining = binaryAddRemainder;
    for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
        pregLoop = UpdateMask<float>(remaining);
        VarRemainderChunk(x32Addr, binaryAddAddr, binaryAddQ, binaryAddR, binaryAddQPow, binaryAddRPow, vlVar, mean,
                          colsNum, pregLoop, pregMerge, i, row, rowStride, vlFp32, binaryAddOffset);
    }
    VarRemainderTailChunk(x32Addr, binaryAddAddr, binaryAddQ, binaryAddR, binaryAddQPow, binaryAddRPow, vlVar, mean,
                          colsNum, pregMain, pregLoop, pregMerge, row, binaryAddRemainder, binaryAddRemainderLoop,
                          rowStride, vlFp32, binaryAddOffset);
    for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
        VarQuotientChunk(x32Addr, binaryAddAddr, vlVar, mean, colsNum, pregMain, pregMerge, i, row,
                         binaryAddRemainderLoop, rowStride, vlFp32);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    DichotomyHalfVar(binaryAddAddr, binaryAddQ, binaryAddR, pregMain, binaryAddLoopVar, binaryAddKLoop, vlFp32);
    uint32_t varLastMaskLen = binaryAddLastNum;
    pregLoop = UpdateMask<float>(varLastMaskLen);
    LoadAlign(varTemp, binaryAddAddr);
    ReduceSum(var, varTemp, pregLoop);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(varAddr + row, var, pregMerge);
    LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
}

template <typename X1_TYPE>
__aicore__ inline void VFCalcVarPhase(const MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr,
                                      uint16_t row, float colsNumFp, int64_t binaryAddRemainder,
                                      uint16_t binaryAddRemainderLoop, uint16_t binaryAddQuotientLoop,
                                      uint16_t binaryAddLoopVar, uint16_t binaryAddKLoop, uint32_t binaryAddOffset,
                                      uint64_t binaryAddLastNum, uint32_t colsPerLoopAlign, uint32_t vlFp32)
{
    asc_vf_call<VarPhaseVf>(localAddr.x32Addr, localAddr.meanAddr, localAddr.varAddr, binaryAddAddr, row, colsNumFp,
                            static_cast<uint32_t>(binaryAddRemainder), binaryAddRemainderLoop, binaryAddQuotientLoop,
                            binaryAddLoopVar, binaryAddKLoop, binaryAddOffset, static_cast<uint32_t>(binaryAddLastNum),
                            colsPerLoopAlign, vlFp32);
}

template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFCalcMeanVar(LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local,
                                     LocalTensor<X1_TYPE>& biasLocal, LocalTensor<X1_TYPE>& xOutLocal,
                                     LocalTensor<float>& x32Local, LocalTensor<float>& meanLocal,
                                     LocalTensor<float>& varLocal, LocalTensor<float> binaryAddLocal,
                                     uint16_t rowsCount, uint32_t colsPerLoop, uint32_t colsPerLoopAlign,
                                     uint32_t vlFp32, uint64_t binaryAddLastNum, uint32_t binaryAddOffset,
                                     uint16_t binaryAddKLoop)
{
    // Set phy addr
    MeanVarLocalAddr<X1_TYPE> localAddr = GetMeanVarLocalAddr<X1_TYPE, TILING_KEY>(
        x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal, varLocal);
    __ubuf__ float* binaryAddAddr = (__ubuf__ float*)binaryAddLocal.GetPhyAddr();

    // Compute VF params
    float colsNumFp = static_cast<float>(colsPerLoop);

    int64_t binaryAddRemainder = colsPerLoop - binaryAddOffset;
    uint16_t binaryAddRemainderLoop = CEIL_DIV(binaryAddRemainder, vlFp32);
    uint16_t binaryAddQuotientLoop = CEIL_DIV(binaryAddOffset, vlFp32);
    uint16_t binaryAddLoopMean = ((binaryAddOffset / vlFp32) / vlFp32);
    uint16_t binaryAddLoopVar = binaryAddLoopMean;

    for (uint16_t k = 0; k < rowsCount; k++) {
        VFCalcMeanPhase<X1_TYPE, TILING_KEY>(
            localAddr, binaryAddAddr, k, colsNumFp, binaryAddRemainder, binaryAddRemainderLoop, binaryAddQuotientLoop,
            binaryAddLoopMean, binaryAddKLoop, binaryAddOffset, binaryAddLastNum, colsPerLoopAlign, vlFp32);
        VFCalcVarPhase<X1_TYPE>(localAddr, binaryAddAddr, k, colsNumFp, binaryAddRemainder, binaryAddRemainderLoop,
                                binaryAddQuotientLoop, binaryAddLoopVar, binaryAddKLoop, binaryAddOffset,
                                binaryAddLastNum, colsPerLoopAlign, vlFp32);
    }
}

} // namespace NsQuantizeAddLayerNorm

#endif // QUANTIZE_ADD_LAYER_NORM_REGBASE_STATS_H
