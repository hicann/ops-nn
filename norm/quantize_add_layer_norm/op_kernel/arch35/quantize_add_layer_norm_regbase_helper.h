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
 * \file quantize_add_layer_norm_regbase_helper.h
 * \brief ascend950 (arch35/regbase) helpers for QuantizeAddLayerNorm.
 *        Mirrors add_layer_norm_quant_regbase_helper.h, single-path static quant only
 *        (no dual scales/offsets, no dynamic quant).
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_REGBASE_COMMON_H
#define QUANTIZE_ADD_LAYER_NORM_REGBASE_COMMON_H

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../add_layer_norm/arch35/add_layer_norm_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace QuantizeAddLayerNormRegbase {
using namespace AddLayerNorm;
using namespace AscendC;
using AscendC::Reg::Truncate;

// OPT_CODE bits: scales is required (always present), zero_points optional
#define OFFSET_CODE 0x1

#define IS_OFFSET_EXIST ((OPT_CODE & OFFSET_CODE) > 0)

// quant mode digit of the regbase tiling key: 0 = mul, 1 = div(per_channel), 2 = per_tensor(scalar mul)
#define IS_DIV_SCALE (((TILING_KEY / 10) % 10) == 1)
#define IS_PER_TENSOR_SCALE (((TILING_KEY / 10) % 10) == 2)

#define CONST_CONDITIONAL_EXPR(_cond, _expr) \
    if constexpr (_cond) {                   \
        _expr;                               \
    }

#define CONST_CONDITIONAL_ASSIGN(_cond, _var, _expr) \
    if constexpr (_cond) {                           \
        _var = _expr;                                \
    }

// fake class members:
constexpr uint32_t vlFp32_ = GetVecLen() / sizeof(float);
constexpr uint32_t blockSize_ = GetDataBlockSizeInBytes();

constexpr AscendC::Reg::CastTrait quantCastTraitF32ToF16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait quantCastTraitF16ToS8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr AscendC::Reg::DivSpecificMode divHighPrecMode = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
};

template <typename T>
__aicore__ inline void LoadQuantParams(__ubuf__ T* ubAddr, RegTensor<float>& dstTensor, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dstTensor, (__ubuf__ float*)ubAddr + offset);
    } else {
        RegTensor<T> dstB16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(dstB16, (__ubuf__ T*)ubAddr + offset);
        Cast<float, T, castTraitB162B32>(dstTensor, dstB16, preg);
    }
}

// per_tensor quant: scales/zero_points are single scalars in GM; the kernel copies only
// element 0 into UB, then broadcasts it to the whole register via a DIST_BRC load
// (same mechanism as the mean/rstd broadcast loads).
template <typename T>
__aicore__ inline void LoadScalarQuantParam(__ubuf__ T* ubAddr, RegTensor<float>& dstTensor, MaskReg& preg)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstTensor, (__ubuf__ float*)ubAddr);
    } else {
        RegTensor<T> dstB16;
        LoadAlign<T, LoadDist::DIST_BRC_B16>(dstB16, (__ubuf__ T*)ubAddr);
        Cast<float, T, castTraitB162B32>(dstTensor, dstB16, preg);
    }
}

__aicore__ inline void Round2Int8(RegTensor<int8_t>& dstTensor, RegTensor<float>& srcTensor, MaskReg& preg)
{
    RegTensor<half> tmpFp16;
    RegTensor<float> tmpFp32;
    Truncate<float, RoundMode::CAST_RINT>(tmpFp32, srcTensor, preg);
    Cast<half, float, quantCastTraitF32ToF16>(tmpFp16, tmpFp32, preg);
    Cast<int8_t, half, quantCastTraitF16ToS8>(dstTensor, tmpFp16, preg);
}

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

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> MakeZeroPadParams(int32_t copyLen, int32_t blockSize)
{
    int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(T), blockSize) / sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.paddingValue = static_cast<T>(0.0);
    padParams.rightPadding = copyLenAlign - copyLen;
    return padParams;
}

template <typename T>
__aicore__ inline DataCopyExtParams MakeDataCopyParams(int32_t copyLen, uint16_t blockCount = 1, uint16_t dstStride = 0,
                                                       uint16_t srcStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = srcStride;
    dataCopyParams.dstStride = dstStride;
    return dataCopyParams;
}

template <typename X1_TYPE, typename GammaQueue, typename BetaQueue>
__aicore__ inline void CopyGammaAndBetaToUBCommon(LocalTensor<X1_TYPE> gammaLocal, LocalTensor<X1_TYPE> betaLocal,
                                                  GlobalTensor<X1_TYPE>& gammaGm, GlobalTensor<X1_TYPE>& betaGm,
                                                  GammaQueue& gammaQueue, BetaQueue& betaQueue, int64_t offset,
                                                  int32_t copyLen, int32_t blockSize)
{
    DataCopyPadExtParams<X1_TYPE> padParams = MakeZeroPadParams<X1_TYPE>(copyLen, blockSize);
    DataCopyExtParams dataCopyParams = MakeDataCopyParams<X1_TYPE>(copyLen);
    DataCopyPad(betaLocal, betaGm[offset], dataCopyParams, padParams);
    betaQueue.EnQue(betaLocal);
    DataCopyPad(gammaLocal, gammaGm[offset], dataCopyParams, padParams);
    gammaQueue.EnQue(gammaLocal);
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
        RegTensor<float> xFactor;
        RegTensor<float> mean;

        RegTensor<float> y;
        RegTensor<float> yFactor;
        RegTensor<float> var;

        RegTensor<float> colsNum;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        uint32_t sreg0 = colsPerLoop;
        MaskReg pregLoop = UpdateMask<float>(sreg0);

        Duplicate(colsNum, colsNumFp, pregMain);
        for (uint16_t i = 0; i < rowsCount; i++) {
            if constexpr (IS_BIAS_BROADCAST) {
                LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(localAddr.x1Addr, localAddr.x2Addr,
                                                                       localAddr.biasAddr, x, pregLoop,
                                                                       i * colsPerLoopAlign, i * colsPerLoopAlign, 0);
            } else {
                LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                    localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, x, pregLoop, i * colsPerLoopAlign,
                    i * colsPerLoopAlign, i * colsPerLoopAlign);
            }
            // save xOut
            StoreRegToOutput(localAddr.xOutAddr, x, pregLoop, i * colsPerLoopAlign);
            // save x32
            StoreAlign((__ubuf__ float*)localAddr.x32Addr + i * colsPerLoopAlign, x, pregLoop);
            Div<float, &divHighPrecMode>(xFactor, x, colsNum, pregLoop);
            ReduceSum(mean, xFactor, pregLoop);

            // save mean
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)localAddr.meanAddr + i, mean,
                                                                 pregMerge);

            Duplicate(mean, mean, pregMain);
            Muls(mean, mean, (float)-1.0, pregMain);
            // xDelta = x - mean
            Add(x, x, mean, pregLoop);
            Mul(y, x, x, pregLoop);
            Div<float, &divHighPrecMode>(yFactor, y, colsNum, pregLoop);
            ReduceSum(var, yFactor, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)localAddr.varAddr + i, var,
                                                                 pregMerge);
        }
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

    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> meanTemp;
        RegTensor<float> mean;

        RegTensor<float> x1;
        RegTensor<float> y1;
        RegTensor<float> y1Pow;
        RegTensor<float> varTemp;
        RegTensor<float> var;

        RegTensor<float> binaryAddQ;
        RegTensor<float> binaryAddR;
        RegTensor<float> vlMean;

        RegTensor<float> binaryAddQPow;
        RegTensor<float> binaryAddRPow;
        RegTensor<float> vlVar;

        RegTensor<float> colsNum;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        Duplicate(colsNum, colsNumFp, pregMain);

        for (uint16_t k = 0; k < rowsCount; k++) {
            uint32_t sreg0 = binaryAddRemainder;
            for (uint16_t i = 0; i < (uint16_t)(binaryAddRemainderLoop - 1); i++) {
                pregLoop = UpdateMask<float>(sreg0);
                if constexpr (IS_BIAS_BROADCAST) {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddQ, pregLoop,
                        i * vlFp32 + k * colsPerLoopAlign, i * vlFp32 + k * colsPerLoopAlign, i * vlFp32);
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddR, pregLoop,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset, i * vlFp32 + binaryAddOffset);
                } else {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddQ, pregLoop,
                        i * vlFp32 + k * colsPerLoopAlign, i * vlFp32 + k * colsPerLoopAlign,
                        i * vlFp32 + k * colsPerLoopAlign);
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddR, pregLoop,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                }
                StoreRegToOutput(localAddr.xOutAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign);
                StoreRegToOutput(localAddr.xOutAddr, binaryAddR, pregLoop,
                                 i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                StoreAlign((__ubuf__ float*)localAddr.x32Addr + i * vlFp32 + k * colsPerLoopAlign, binaryAddQ,
                           pregLoop);
                StoreAlign((__ubuf__ float*)localAddr.x32Addr + i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                           binaryAddR, pregLoop);

                Div<float, &divHighPrecMode>(binaryAddQ, binaryAddQ, colsNum, pregLoop);
                Div<float, &divHighPrecMode>(binaryAddR, binaryAddR, colsNum, pregLoop);

                Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
                ReduceSum(vlMean, binaryAddQ, pregLoop);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + i), vlMean,
                                                                     pregMerge);
            }
            {
                pregLoop = UpdateMask<float>(sreg0);
                if constexpr (IS_BIAS_BROADCAST) {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, binaryAddQ, pregMain,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                        (binaryAddRemainderLoop - 1) * vlFp32);
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
                StoreAlign(
                    (__ubuf__ float*)localAddr.x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                    binaryAddQ, pregMain);
                StoreAlign((__ubuf__ float*)localAddr.x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                               k * colsPerLoopAlign + binaryAddOffset,
                           binaryAddR, pregLoop);

                Div<float, &divHighPrecMode>(binaryAddQ, binaryAddQ, colsNum, pregMain);
                Div<float, &divHighPrecMode>(binaryAddR, binaryAddR, colsNum, pregLoop);

                Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                ReduceSum(vlMean, binaryAddQ, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop - 1), vlMean, pregMerge);
            }
            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                if constexpr (IS_BIAS_BROADCAST) {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, x, pregMain,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                        (i + binaryAddRemainderLoop) * vlFp32);
                } else {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        localAddr.x1Addr, localAddr.x2Addr, localAddr.biasAddr, x, pregMain,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
                }
                StoreRegToOutput(localAddr.xOutAddr, x, pregMain,
                                 (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
                StoreAlign(
                    (__ubuf__ float*)localAddr.x32Addr + (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                    x, pregMain);
                Div<float, &divHighPrecMode>(x, x, colsNum, pregMain);
                ReduceSum(vlMean, x, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i), vlMean, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
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

            uint32_t sreg1 = binaryAddRemainder;
            for (uint16_t i = 0; i < (uint16_t)(binaryAddRemainderLoop - 1); i++) {
                pregLoop = UpdateMask<float>(sreg1);
                LoadAlign(binaryAddQ, (__ubuf__ float*)localAddr.x32Addr + i * vlFp32 + k * colsPerLoopAlign);
                LoadAlign(binaryAddR,
                          (__ubuf__ float*)localAddr.x32Addr + i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
                Sub(binaryAddR, binaryAddR, mean, pregLoop);
                Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
                Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);

                Div<float, &divHighPrecMode>(binaryAddQPow, binaryAddQPow, colsNum, pregLoop);
                Div<float, &divHighPrecMode>(binaryAddRPow, binaryAddRPow, colsNum, pregLoop);

                Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
                ReduceSum(vlVar, binaryAddQPow, pregLoop);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)binaryAddAddr + i), vlVar,
                                                                     pregMerge);
            }
            {
                pregLoop = UpdateMask<float>(sreg1);
                LoadAlign(binaryAddQ, (__ubuf__ float*)localAddr.x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                          k * colsPerLoopAlign);
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
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop - 1), vlVar, pregMerge);
            }
            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                LoadAlign(x1, (__ubuf__ float*)localAddr.x32Addr + (i + binaryAddRemainderLoop) * vlFp32 +
                                  k * colsPerLoopAlign);
                Sub(y1, x1, mean, pregMain);
                Mul(y1Pow, y1, y1, pregMain);
                Div<float, &divHighPrecMode>(y1Pow, y1Pow, colsNum, pregMain);
                ReduceSum(vlVar, y1Pow, pregMain);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__ubuf__ float*)binaryAddAddr + binaryAddRemainderLoop + i), vlVar, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
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
}

template <bool INIT, typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFWelfordParallelUpdateCommon(LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local,
                                                     LocalTensor<X1_TYPE>& biasLocal, LocalTensor<X1_TYPE>& xLocal,
                                                     LocalTensor<float>& tmpMeanLocal, LocalTensor<float>& tmpVarLocal,
                                                     uint64_t calLen, uint16_t loopCount, float scale)
{
    __ubuf__ X1_TYPE* x1Addr = (__ubuf__ X1_TYPE*)x1Local[0].GetPhyAddr();
    __ubuf__ X1_TYPE* x2Addr = (__ubuf__ X1_TYPE*)x2Local[0].GetPhyAddr();
    __ubuf__ X1_TYPE* xOutAddr = (__ubuf__ X1_TYPE*)xLocal[0].GetPhyAddr();
    __ubuf__ float* tmpMeanAddr = (__ubuf__ float*)tmpMeanLocal.GetPhyAddr();
    __ubuf__ float* tmpVarAddr = (__ubuf__ float*)tmpVarLocal.GetPhyAddr();

    __ubuf__ X1_TYPE* biasAddr = nullptr;
    if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
        biasAddr = (__ubuf__ X1_TYPE*)biasLocal[0].GetPhyAddr();
    }

    AddLayerNorm::VFWelfordParallelUpdateCommon<INIT, X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
        x1Addr, x2Addr, biasAddr, xOutAddr, tmpMeanAddr, tmpVarAddr, calLen, loopCount, scale);
}

__aicore__ inline void VFWelfordParallelFinalizeNonAlign(
    LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal, LocalTensor<float>& tmpMeanLocal,
    LocalTensor<float>& tmpVarLocal, LocalTensor<float>& dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float reduceScaleCorrection, float cnt, float eps)
{
    __ubuf__ float* meanAddr = (__ubuf__ float*)meanLocal[0].GetPhyAddr();
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal[0].GetPhyAddr();
    __ubuf__ float* tmpMeanAddr = (__ubuf__ float*)tmpMeanLocal[0].GetPhyAddr();
    __ubuf__ float* tmpVarAddr = (__ubuf__ float*)tmpVarLocal.GetPhyAddr();
    __ubuf__ float* dichotomyAddAddr = (__ubuf__ float*)dichotomyAddLocal.GetPhyAddr();
    AddLayerNorm::VFWelfordParallelFinalizeNonAlign(meanAddr, rstdAddr, tmpMeanAddr, tmpVarAddr, dichotomyAddAddr,
                                                    reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum,
                                                    offset, tailSize, reduceScale, reduceScaleCorrection, cnt, eps);
}

/*
  Welford Finalize aligned-scenario formulas:
  finalize_mean = sum_fun(mean) / parallel_N
  finalize_delta = mean - finalize_mean
  finalize_delta_square = finalize_delta * finalize_delta
  M2_fixed = M2 + float(count) * finalize_delta_square
  finalize_std = sum_fun(M2_fixed) / float(parallel_N * count)

  welford computes mean and variance with dichotomy accumulation.
  scale / reduceScale are exact power-of-two reciprocals; scaleCorrection / reduceScaleCorrection
  restore the true 1/colsPerLoop / 1/cols factor in one Muls after DichotomyAdd (rounding-error fix).
*/
__aicore__ inline void VFWelfordParallelFinalizeAlign(LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal,
                                                      LocalTensor<float>& tmpMeanLocal, LocalTensor<float>& tmpVarLocal,
                                                      LocalTensor<float>& dichotomyAddLocal, uint32_t reduceCount,
                                                      uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                      uint32_t dichotomyAddLastNum, uint32_t offset, float reduceScale,
                                                      float reduceScaleCorrection, float scale, float scaleCorrection,
                                                      float cnt, float eps)
{
    __ubuf__ float* meanAddr = (__ubuf__ float*)meanLocal[0].GetPhyAddr();
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal[0].GetPhyAddr();
    __ubuf__ float* tmpMeanAddr = (__ubuf__ float*)tmpMeanLocal[0].GetPhyAddr();
    __ubuf__ float* tmpVarAddr = (__ubuf__ float*)tmpVarLocal.GetPhyAddr();
    __ubuf__ float* dichotomyAddAddr = (__ubuf__ float*)dichotomyAddLocal.GetPhyAddr();

    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        RegTensor<float> one;
        RegTensor<float> rstd;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t sreg0 = dichotomyAddReminder;
        // compute mean
        // PART1: merge tail+full blocks
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL, tmpMeanAddr + i * VL_FP32);
            LoadAlign(dichotomyAddMeanR, tmpMeanAddr + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, scale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddAddr + i, mean, pregMerge);
        }

        // PART2: vcadd remaining full blocks back to UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
             i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddAddr + dichotomyAddReminderLoopCount + i, mean, pregMerge);
        }

        NormCommon::DichotomyAdd(mean, dichotomyAddAddr, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        Muls(mean, mean, scaleCorrection, pregMerge);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanAddr + offset, mean, pregMerge);

        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        sreg0 = dichotomyAddReminder;
        // PART1: merge tail+full blocks
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL, tmpMeanAddr + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            LoadAlign(dichotomyAddVarL, tmpVarAddr + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

            LoadAlign(dichotomyAddMeanR, tmpMeanAddr + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            LoadAlign(dichotomyAddVarR, tmpVarAddr + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddAddr + i, var, pregMerge);
        }

        // PART2: vcadd remaining full blocks back to UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
             i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            LoadAlign(dichotomyAddVarL, tmpVarAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddAddr + dichotomyAddReminderLoopCount + i, var, pregMerge);
        }

        NormCommon::DichotomyAdd(var, dichotomyAddAddr, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        Muls(var, var, reduceScaleCorrection, pregMerge);
        NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + offset, rstd, pregMerge);
    }
}

} // namespace QuantizeAddLayerNormRegbase

#endif // QUANTIZE_ADD_LAYER_NORM_REGBASE_COMMON_H
