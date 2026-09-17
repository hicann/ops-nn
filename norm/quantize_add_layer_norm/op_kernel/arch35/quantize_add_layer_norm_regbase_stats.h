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
 *        The VFCalcMeanVar binaryAdd two-pass pipeline is decomposed into per-row phase
 *        kernels (each owning its __VEC_SCOPE__) plus per-chunk helpers called from inside
 *        those scopes (rms_norm LoadSquareRemainTile pattern). All LocalMemBar fences stay
 *        inside the scope of the function they synchronize; DichotomyHalf* must only be
 *        called from inside a scope (NormCommon::DichotomyAdd precedent).
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_REGBASE_STATS_H
#define QUANTIZE_ADD_LAYER_NORM_REGBASE_STATS_H

#include "quantize_add_layer_norm_regbase_helper.h"

namespace QuantizeAddLayerNormRegbase {

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
// two-pass mean/var reduces. xFactor/mean/y/yFactor/var live entirely inside the iteration.
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void MeanVarFastChunk(MeanVarLocalAddr<X1_TYPE>& localAddr, RegTensor<float>& x,
                                        RegTensor<float>& colsNum, MaskReg& pregMain, MaskReg& pregMerge,
                                        MaskReg& pregLoop, uint16_t i, uint32_t colsPerLoopAlign)
{
    RegTensor<float> xFactor;
    RegTensor<float> mean;
    RegTensor<float> y;
    RegTensor<float> yFactor;
    RegTensor<float> var;
    if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr,
                                                               x, pregLoop, i * colsPerLoopAlign, i * colsPerLoopAlign,
                                                               0);
    } else {
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr,
                                                               x, pregLoop, i * colsPerLoopAlign, i * colsPerLoopAlign,
                                                               i * colsPerLoopAlign);
    }
    // save xOut
    StoreRegToOutput(localAddr.xOutAddr, x, pregLoop, i * colsPerLoopAlign);
    // save x32
    StoreAlign((__ubuf__ float*)localAddr.x32Addr + i * colsPerLoopAlign, x, pregLoop);
    Div<float, &divHighPrecMode>(xFactor, x, colsNum, pregLoop);
    ReduceSum(mean, xFactor, pregLoop);

    // save mean
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)localAddr.meanAddr + i, mean, pregMerge);

    Duplicate(mean, mean, pregMain);
    Muls(mean, mean, (float)-1.0, pregMain);
    // xDelta = x - mean
    Add(x, x, mean, pregLoop);
    Mul(y, x, x, pregLoop);
    Div<float, &divHighPrecMode>(yFactor, y, colsNum, pregLoop);
    ReduceSum(var, yFactor, pregLoop);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)localAddr.varAddr + i, var, pregMerge);
}

// AddLayerNormCommon
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFCalcMeanVarFast(LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local,
                                         LocalTensor<X1_TYPE>& biasLocal, LocalTensor<X1_TYPE>& xOutLocal,
                                         LocalTensor<float>& x32Local, LocalTensor<float>& meanLocal,
                                         LocalTensor<float>& varLocal, uint16_t rowsCount, int64_t powerOfTwo,
                                         uint32_t colsPerLoop, uint32_t colsPerLoopAlign, uint32_t vlFp32)
{
    // Set phy addr
    MeanVarLocalAddr<X1_TYPE> localAddr = GetMeanVarLocalAddr<X1_TYPE, TILING_KEY>(
        x1Local, x2Local, biasLocal, xOutLocal, x32Local, meanLocal, varLocal);

    // Compute VF params
    float colsNumFp = static_cast<float>(colsPerLoop);

    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> colsNum;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        uint32_t sreg0 = colsPerLoop;
        MaskReg pregLoop = UpdateMask<float>(sreg0);

        Duplicate(colsNum, colsNumFp, pregMain);
        for (uint16_t i = 0; i < rowsCount; i++) {
            MeanVarFastChunk<X1_TYPE, TILING_KEY>(localAddr, x, colsNum, pregMain, pregMerge, pregLoop, i,
                                                  colsPerLoopAlign);
        }
    }
}

// Mean pass, non-last binaryAddRemainder iteration: dual half-chunks (Q at offset, R at
// offset + binaryAddOffset), both masked with pregLoop.
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void MeanRemainderChunk(MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr,
                                          RegTensor<float>& binaryAddQ, RegTensor<float>& binaryAddR,
                                          RegTensor<float>& vlMean, RegTensor<float>& colsNum, MaskReg& pregLoop,
                                          MaskReg& pregMerge, uint16_t i, uint16_t k, uint32_t colsPerLoopAlign,
                                          uint32_t vlFp32, uint32_t binaryAddOffset)
{
    if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr,
                                                               binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign,
                                                               i * vlFp32 + k * colsPerLoopAlign, i * vlFp32);
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddR, pregLoop,
            i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset, i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
            i * vlFp32 + binaryAddOffset);
    } else {
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddQ, pregLoop,
            i * vlFp32 + k * colsPerLoopAlign, i * vlFp32 + k * colsPerLoopAlign, i * vlFp32 + k * colsPerLoopAlign);
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddR, pregLoop,
            i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset, i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
            i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
    }
    StoreRegToOutput(localAddr.xOutAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign);
    StoreRegToOutput(localAddr.xOutAddr, binaryAddR, pregLoop, i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
    StoreAlign((__ubuf__ float*)localAddr.x32Addr + i * vlFp32 + k * colsPerLoopAlign, binaryAddQ, pregLoop);
    StoreAlign((__ubuf__ float*)localAddr.x32Addr + i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset, binaryAddR,
               pregLoop);

    Div<float, &divHighPrecMode>(binaryAddQ, binaryAddQ, colsNum, pregLoop);
    Div<float, &divHighPrecMode>(binaryAddR, binaryAddR, colsNum, pregLoop);

    Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
    ReduceSum(vlMean, binaryAddQ, pregLoop);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + i), vlMean, pregMerge);
}

// Mean pass, last binaryAddRemainder iteration: Q side runs full-lane (pregMain), R side keeps pregLoop.
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void MeanRemainderTailChunk(MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr,
                                              RegTensor<float>& binaryAddQ, RegTensor<float>& binaryAddR,
                                              RegTensor<float>& vlMean, RegTensor<float>& colsNum, MaskReg& pregMain,
                                              MaskReg& pregLoop, MaskReg& pregMerge, uint16_t k,
                                              int64_t binaryAddRemainder, uint16_t binaryAddRemainderLoop,
                                              uint32_t colsPerLoopAlign, uint32_t vlFp32, uint32_t binaryAddOffset)
{
    uint32_t sreg0 = binaryAddRemainder;
    pregLoop = UpdateMask<float>(sreg0);
    if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddQ, pregMain,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign, (binaryAddRemainderLoop - 1) * vlFp32);
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddR, pregLoop,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
            (binaryAddRemainderLoop - 1) * vlFp32 + binaryAddOffset);
    } else {
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddQ, pregMain,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign);
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddR, pregLoop,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
            (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
    }
    StoreRegToOutput(localAddr.xOutAddr, binaryAddQ, pregMain,
                     (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign);
    StoreRegToOutput(localAddr.xOutAddr, binaryAddR, pregLoop,
                     (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
    StoreAlign((__ubuf__ float*)localAddr.x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
               binaryAddQ, pregMain);
    StoreAlign((__ubuf__ float*)localAddr.x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign +
                   binaryAddOffset,
               binaryAddR, pregLoop);

    Div<float, &divHighPrecMode>(binaryAddQ, binaryAddQ, colsNum, pregMain);
    Div<float, &divHighPrecMode>(binaryAddR, binaryAddR, colsNum, pregLoop);

    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
    ReduceSum(vlMean, binaryAddQ, pregMain);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop - 1),
                                                         vlMean, pregMerge);
}

// Mean pass, quotient iterations beyond the remainder region: single stream, full-lane mask.
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void MeanQuotientChunk(MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr,
                                         RegTensor<float>& vlMean, RegTensor<float>& colsNum, MaskReg& pregMain,
                                         MaskReg& pregMerge, uint16_t i, uint16_t k, uint16_t binaryAddRemainderLoop,
                                         uint32_t colsPerLoopAlign, uint32_t vlFp32)
{
    RegTensor<float> x;
    if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, x, pregMain,
            (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
            (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign, (i + binaryAddRemainderLoop) * vlFp32);
    } else {
        LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
            localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, x, pregMain,
            (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
            (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
            (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
    }
    StoreRegToOutput(localAddr.xOutAddr, x, pregMain, (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
    StoreAlign((__ubuf__ float*)localAddr.x32Addr + (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign, x,
               pregMain);

    Div<float, &divHighPrecMode>(x, x, colsNum, pregMain);
    ReduceSum(vlMean, x, pregMain);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i),
                                                         vlMean, pregMerge);
}

// Mean-pass dichotomy tree reduction. MUST only be called from inside a __VEC_SCOPE__:
// it contains a LocalMemBar and inlines into the caller's vector region.
__aicore__ inline void DichotomyHalfMean(__ubuf__ float* binaryAddAddr, RegTensor<float>& binaryAddQ,
                                         RegTensor<float>& binaryAddR, MaskReg& pregMain, uint16_t binaryAddLoopMean,
                                         uint16_t binaryAddKLoop, uint32_t vlFp32)
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
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VarRemainderChunk(MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr,
                                         RegTensor<float>& binaryAddQ, RegTensor<float>& binaryAddR,
                                         RegTensor<float>& binaryAddQPow, RegTensor<float>& binaryAddRPow,
                                         RegTensor<float>& vlVar, RegTensor<float>& mean, RegTensor<float>& colsNum,
                                         MaskReg& pregLoop, MaskReg& pregMerge, uint16_t i, uint16_t k,
                                         uint32_t colsPerLoopAlign, uint32_t vlFp32, uint32_t binaryAddOffset)
{
    LoadAlign(binaryAddQ, (__ubuf__ float*)localAddr.x32Addr + i * vlFp32 + k * colsPerLoopAlign);
    LoadAlign(binaryAddR, (__ubuf__ float*)localAddr.x32Addr + i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
    Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
    Sub(binaryAddR, binaryAddR, mean, pregLoop);
    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);

    Div<float, &divHighPrecMode>(binaryAddQPow, binaryAddQPow, colsNum, pregLoop);
    Div<float, &divHighPrecMode>(binaryAddRPow, binaryAddRPow, colsNum, pregLoop);

    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
    ReduceSum(vlVar, binaryAddQPow, pregLoop);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + i), vlVar, pregMerge);
}

// Var pass, last binaryAddRemainder iteration: Q side full-lane (pregMain), R side pregLoop.
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VarRemainderTailChunk(MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr,
                                             RegTensor<float>& binaryAddQ, RegTensor<float>& binaryAddR,
                                             RegTensor<float>& binaryAddQPow, RegTensor<float>& binaryAddRPow,
                                             RegTensor<float>& vlVar, RegTensor<float>& mean, RegTensor<float>& colsNum,
                                             MaskReg& pregMain, MaskReg& pregLoop, MaskReg& pregMerge, uint16_t k,
                                             int64_t binaryAddRemainder, uint16_t binaryAddRemainderLoop,
                                             uint32_t colsPerLoopAlign, uint32_t vlFp32, uint32_t binaryAddOffset)
{
    uint32_t sreg1 = binaryAddRemainder;
    pregLoop = UpdateMask<float>(sreg1);
    LoadAlign(binaryAddQ,
              (__ubuf__ float*)localAddr.x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign);
    LoadAlign(binaryAddR, (__ubuf__ float*)localAddr.x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                              k * colsPerLoopAlign + binaryAddOffset);
    Sub(binaryAddQ, binaryAddQ, mean, pregMain);
    Sub(binaryAddR, binaryAddR, mean, pregLoop);
    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregMain);
    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);

    Div<float, &divHighPrecMode>(binaryAddQPow, binaryAddQPow, colsNum, pregMain);
    Div<float, &divHighPrecMode>(binaryAddRPow, binaryAddRPow, colsNum, pregLoop);

    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregMain);
    ReduceSum(vlVar, binaryAddQPow, pregMain);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop - 1),
                                                         vlVar, pregMerge);
}

// Var pass, quotient iterations beyond the remainder region.
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VarQuotientChunk(MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr,
                                        RegTensor<float>& vlVar, RegTensor<float>& mean, RegTensor<float>& colsNum,
                                        MaskReg& pregMain, MaskReg& pregMerge, uint16_t i, uint16_t k,
                                        uint16_t binaryAddRemainderLoop, uint32_t colsPerLoopAlign, uint32_t vlFp32)
{
    RegTensor<float> x1;
    RegTensor<float> y1;
    RegTensor<float> y1Pow;
    LoadAlign(x1, (__ubuf__ float*)localAddr.x32Addr + (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
    Sub(y1, x1, mean, pregMain);
    Mul(y1Pow, y1, y1, pregMain);
    Div<float, &divHighPrecMode>(y1Pow, y1Pow, colsNum, pregMain);
    ReduceSum(vlVar, y1Pow, pregMain);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i),
                                                         vlVar, pregMerge);
}

// Var-pass dichotomy tree reduction. MUST only be called from inside a __VEC_SCOPE__.
__aicore__ inline void DichotomyHalfVar(__ubuf__ float* binaryAddAddr, RegTensor<float>& binaryAddQ,
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
__aicore__ inline void VFCalcMeanPhase(MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr, uint16_t k,
                                       float colsNumFp, int64_t binaryAddRemainder, uint16_t binaryAddRemainderLoop,
                                       uint16_t binaryAddQuotientLoop, uint16_t binaryAddLoopMean,
                                       uint16_t binaryAddKLoop, uint32_t binaryAddOffset, uint64_t binaryAddLastNum,
                                       uint32_t colsPerLoopAlign, uint32_t vlFp32)
{
    __VEC_SCOPE__
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
        uint32_t sreg0 = binaryAddRemainder;

        Duplicate(colsNum, colsNumFp, pregMain);
        for (uint16_t i = 0; i < (uint16_t)(binaryAddRemainderLoop - 1); i++) {
            pregLoop = UpdateMask<float>(sreg0);
            MeanRemainderChunk<X1_TYPE, TILING_KEY>(localAddr, binaryAddAddr, binaryAddQ, binaryAddR, vlMean, colsNum,
                                                    pregLoop, pregMerge, i, k, colsPerLoopAlign, vlFp32,
                                                    binaryAddOffset);
        }
        MeanRemainderTailChunk<X1_TYPE, TILING_KEY>(localAddr, binaryAddAddr, binaryAddQ, binaryAddR, vlMean, colsNum,
                                                    pregMain, pregLoop, pregMerge, k, binaryAddRemainder,
                                                    binaryAddRemainderLoop, colsPerLoopAlign, vlFp32, binaryAddOffset);
        for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
            MeanQuotientChunk<X1_TYPE, TILING_KEY>(localAddr, binaryAddAddr, vlMean, colsNum, pregMain, pregMerge, i, k,
                                                   binaryAddRemainderLoop, colsPerLoopAlign, vlFp32);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DichotomyHalfMean(binaryAddAddr, binaryAddQ, binaryAddR, pregMain, binaryAddLoopMean, binaryAddKLoop, vlFp32);
        {
            uint32_t meanLastMaskLen = binaryAddLastNum;
            pregLoop = UpdateMask<float>(meanLastMaskLen);
            LoadAlign(meanTemp, ((__ubuf__ float*)binaryAddAddr));
            ReduceSum(mean, meanTemp, pregLoop);
        }

        // batch mean
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)localAddr.meanAddr + k), mean,
                                                             pregMerge);
        Duplicate(mean, mean, pregMain);
        LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
    }
}

// Per-row variance phase of VFCalcMeanVar. mean is broadcast back from the mean phase's
// lane-0 store (DIST_BRC_B32 load, same mechanism as the full_load VFCalcYQuant mean/rstd loads).
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFCalcVarPhase(MeanVarLocalAddr<X1_TYPE>& localAddr, __ubuf__ float* binaryAddAddr, uint16_t k,
                                      float colsNumFp, int64_t binaryAddRemainder, uint16_t binaryAddRemainderLoop,
                                      uint16_t binaryAddQuotientLoop, uint16_t binaryAddLoopVar,
                                      uint16_t binaryAddKLoop, uint32_t binaryAddOffset, uint64_t binaryAddLastNum,
                                      uint32_t colsPerLoopAlign, uint32_t vlFp32)
{
    __VEC_SCOPE__
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
        LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, (__ubuf__ float*)localAddr.meanAddr + k);

        uint32_t sreg1 = binaryAddRemainder;
        for (uint16_t i = 0; i < (uint16_t)(binaryAddRemainderLoop - 1); i++) {
            pregLoop = UpdateMask<float>(sreg1);
            VarRemainderChunk<X1_TYPE, TILING_KEY>(localAddr, binaryAddAddr, binaryAddQ, binaryAddR, binaryAddQPow,
                                                   binaryAddRPow, vlVar, mean, colsNum, pregLoop, pregMerge, i, k,
                                                   colsPerLoopAlign, vlFp32, binaryAddOffset);
        }
        VarRemainderTailChunk<X1_TYPE, TILING_KEY>(localAddr, binaryAddAddr, binaryAddQ, binaryAddR, binaryAddQPow,
                                                   binaryAddRPow, vlVar, mean, colsNum, pregMain, pregLoop, pregMerge,
                                                   k, binaryAddRemainder, binaryAddRemainderLoop, colsPerLoopAlign,
                                                   vlFp32, binaryAddOffset);
        for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
            VarQuotientChunk<X1_TYPE, TILING_KEY>(localAddr, binaryAddAddr, vlVar, mean, colsNum, pregMain, pregMerge,
                                                  i, k, binaryAddRemainderLoop, colsPerLoopAlign, vlFp32);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DichotomyHalfVar(binaryAddAddr, binaryAddQ, binaryAddR, pregMain, binaryAddLoopVar, binaryAddKLoop, vlFp32);
        {
            uint32_t varLastMaskLen = binaryAddLastNum;
            pregLoop = UpdateMask<float>(varLastMaskLen);
            LoadAlign(varTemp, ((__ubuf__ float*)binaryAddAddr));
            ReduceSum(var, varTemp, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)localAddr.varAddr + k), var,
                                                                 pregMerge);
        }
        LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
    }
}

template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFCalcMeanVar(LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local,
                                     LocalTensor<X1_TYPE>& biasLocal, LocalTensor<X1_TYPE>& xOutLocal,
                                     LocalTensor<float>& x32Local, LocalTensor<float>& meanLocal,
                                     LocalTensor<float>& varLocal, LocalTensor<float> binaryAddLocal,
                                     uint16_t rowsCount, int64_t powerOfTwo, uint32_t colsPerLoop,
                                     uint32_t colsPerLoopAlign, uint32_t vlFp32, uint64_t binaryAddLastNum,
                                     uint32_t binaryAddOffset, uint16_t binaryAddKLoop)
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
        VFCalcVarPhase<X1_TYPE, TILING_KEY>(
            localAddr, binaryAddAddr, k, colsNumFp, binaryAddRemainder, binaryAddRemainderLoop, binaryAddQuotientLoop,
            binaryAddLoopVar, binaryAddKLoop, binaryAddOffset, binaryAddLastNum, colsPerLoopAlign, vlFp32);
    }
}

} // namespace QuantizeAddLayerNormRegbase

#endif // QUANTIZE_ADD_LAYER_NORM_REGBASE_STATS_H
