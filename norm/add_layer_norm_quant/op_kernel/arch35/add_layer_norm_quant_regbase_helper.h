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
 * \file add_layer_norm_quant_regbase_helper.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_QUANT_REGBASE_COMMON_H
#define ADD_LAYER_NORM_QUANT_REGBASE_COMMON_H

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../add_layer_norm/arch35/add_layer_norm_regbase_common.h"

namespace AddLayerNormQuantRegbase {
using namespace AddLayerNorm;
using namespace AscendC;
using AscendC::MicroAPI::Truncate;

#define SCALE1_CODE 0x8
#define SCALE2_CODE 0x4
#define OFFSET1_CODE 0x2
#define OFFSET2_CODE 0x1

#define IS_SCALE1_EXIST ((OPT_CODE & SCALE1_CODE) > 0)
#define IS_SCALE2_EXIST ((OPT_CODE & SCALE2_CODE) > 0)
#define IS_OFFSET1_EXIST ((OPT_CODE & OFFSET1_CODE) > 0)
#define IS_OFFSET2_EXIST ((OPT_CODE & OFFSET2_CODE) > 0)

#define IS_DIV_SCALE (((TILING_KEY / 10) % 10) == 1)

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

constexpr AscendC::MicroAPI::CastTrait quantCastTraitF32ToS32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait quantCastTraitS32ToF32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr AscendC::MicroAPI::CastTrait quantCastTraitF32ToF16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait quantCastTraitF16ToS8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr AscendC::MicroAPI::DivSpecificMode divHighPrecMode = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};

constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr float ZERO_ME = 0.0;

template <HardEvent ent>
__aicore__ inline void SimpleNativePipeSync()
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(ent));
    SetFlag<ent>(event);
    WaitFlag<ent>(event);
}

template <typename T>
__aicore__ inline void LoadQuantParams(
    __local_mem__ T* ubAddr, RegTensor<float>& dstTensor, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dstTensor, (__local_mem__ float*)ubAddr + offset);
    } else {
        RegTensor<T> dstB16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(dstB16, (__local_mem__ T*)ubAddr + offset);
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

__aicore__ inline void ComputeDynamicQuantWithSmooth(
    RegTensor<int8_t>& dst, RegTensor<float>& src, RegTensor<float>& smooth, RegTensor<float>& scale, MaskReg& preg)
{
    RegTensor<float> sx;
    Mul(sx, src, smooth, preg);
    Div(sx, sx, scale, preg);
    Round2Int8(dst, sx, preg);
}

__aicore__ inline void ComputeDynamicQuantWithOutSmooth(
    RegTensor<int8_t>& dst, RegTensor<float>& src, RegTensor<float>& scale, MaskReg& preg)
{
    RegTensor<float> sx;
    Div(sx, src, scale, preg);
    Round2Int8(dst, sx, preg);
}

// AddLayerNormCommon
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFCalcMeanVarFast(
    LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local, LocalTensor<X1_TYPE>& biasLocal,
    LocalTensor<X1_TYPE>& xOutLocal, LocalTensor<float>& x32Local, LocalTensor<float>& meanLocal,
    LocalTensor<float>& varLocal, uint16_t rowsCount, int64_t powerOfTwo, uint32_t colsPerLoop,
    uint32_t colsPerLoopAlign, uint32_t vlFp32)
{
    // Set phy addr
    __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
    __local_mem__ X1_TYPE* x2Addr = (__local_mem__ X1_TYPE*)x2Local[0].GetPhyAddr();
    __local_mem__ X1_TYPE* biasAddr;
    if constexpr ((IS_BIAS_ELEWISE || IS_BIAS_BROADCAST)) {
        biasAddr = (__local_mem__ X1_TYPE*)biasLocal[0].GetPhyAddr();
    }
    __local_mem__ X1_TYPE* xOutAddr = (__local_mem__ X1_TYPE*)xOutLocal[0].GetPhyAddr();
    __local_mem__ float* x32Addr = (__local_mem__ float*)x32Local[0].GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
    __local_mem__ float* varAddr = (__local_mem__ float*)varLocal[0].GetPhyAddr();

    // Compute VF params
    float colsNumFp = static_cast<float>(colsPerLoop);
    uint16_t rowsLoopCount = CEIL_DIV(rowsCount, vlFp32);

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
                LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                    x1Addr, x2Addr, biasAddr, x, pregLoop, i * colsPerLoopAlign, i * colsPerLoopAlign, 0);
            } else {
                LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                    x1Addr, x2Addr, biasAddr, x, pregLoop, i * colsPerLoopAlign, i * colsPerLoopAlign,
                    i * colsPerLoopAlign);
            }
            // save xOut
            StoreRegToOutput(xOutAddr, x, pregLoop, i * colsPerLoopAlign);
            // save x32
            DataCopy((__local_mem__ float*)x32Addr + i * colsPerLoopAlign, x, pregLoop);
            Div<float, &divHighPrecMode>(xFactor, x, colsNum, pregLoop);
            ReduceSum(mean, xFactor, pregLoop);

            // save mean
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)meanAddr + i, mean, pregMerge);

            Duplicate(mean, mean, pregMain);
            Muls(mean, mean, (float)-1.0, pregMain);
            // xDelta = x - mean
            Add(x, x, mean, pregLoop);
            Mul(y, x, x, pregLoop);
            Div<float, &divHighPrecMode>(yFactor, y, colsNum, pregLoop);
            ReduceSum(var, yFactor, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)varAddr + i, var, pregMerge);
        }
    }
}

template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFCalcMeanVar(
    LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local, LocalTensor<X1_TYPE>& biasLocal,
    LocalTensor<X1_TYPE>& xOutLocal, LocalTensor<float>& x32Local, LocalTensor<float>& meanLocal,
    LocalTensor<float>& varLocal, LocalTensor<float> binaryAddLocal, uint16_t rowsCount, int64_t powerOfTwo,
    uint32_t colsPerLoop, uint32_t colsPerLoopAlign, uint32_t vlFp32, uint64_t binaryAddLastNum,
    uint32_t binaryAddOffset, uint16_t binaryAddKLoop)
{
    // Set phy addr
    __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
    __local_mem__ X1_TYPE* x2Addr = (__local_mem__ X1_TYPE*)x2Local[0].GetPhyAddr();
    __local_mem__ X1_TYPE* biasAddr;
    if constexpr ((IS_BIAS_ELEWISE || IS_BIAS_BROADCAST)) {
        biasAddr = (__local_mem__ X1_TYPE*)biasLocal[0].GetPhyAddr();
    }
    __local_mem__ X1_TYPE* xOutAddr = (__local_mem__ X1_TYPE*)xOutLocal[0].GetPhyAddr();
    __local_mem__ float* x32Addr = (__local_mem__ float*)x32Local[0].GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
    __local_mem__ float* varAddr = (__local_mem__ float*)varLocal[0].GetPhyAddr();
    __local_mem__ float* binaryAddAddr = (__local_mem__ float*)binaryAddLocal.GetPhyAddr();

    // Compute VF params
    float colsNumFp = static_cast<float>(colsPerLoop);
    uint16_t rowsLoopCount = CEIL_DIV(rowsCount, vlFp32);

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
                        x1Addr, x2Addr, biasAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign,
                        i * vlFp32 + k * colsPerLoopAlign, i * vlFp32);
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset, i * vlFp32 + binaryAddOffset);
                } else {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign,
                        i * vlFp32 + k * colsPerLoopAlign, i * vlFp32 + k * colsPerLoopAlign);
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                }
                StoreRegToOutput(xOutAddr, binaryAddQ, pregLoop, i * vlFp32 + k * colsPerLoopAlign);
                StoreRegToOutput(xOutAddr, binaryAddR, pregLoop, i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                DataCopy((__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign, binaryAddQ, pregLoop);
                DataCopy(
                    (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset, binaryAddR,
                    pregLoop);

                Div<float, &divHighPrecMode>(binaryAddQ, binaryAddQ, colsNum, pregLoop);
                Div<float, &divHighPrecMode>(binaryAddR, binaryAddR, colsNum, pregLoop);

                Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
                ReduceSum(vlMean, binaryAddQ, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)binaryAddAddr + i), vlMean, pregMerge);
            }
            {
                pregLoop = UpdateMask<float>(sreg0);
                if constexpr (IS_BIAS_BROADCAST) {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, binaryAddQ, pregMain,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                        (binaryAddRemainderLoop - 1) * vlFp32);
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        (binaryAddRemainderLoop - 1) * vlFp32 + binaryAddOffset);
                } else {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, binaryAddQ, pregMain,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign);
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, binaryAddR, pregLoop,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset,
                        (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                }
                StoreRegToOutput(
                    xOutAddr, binaryAddQ, pregMain, (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign);
                StoreRegToOutput(
                    xOutAddr, binaryAddR, pregLoop,
                    (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                DataCopy(
                    (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign,
                    binaryAddQ, pregMain);
                DataCopy(
                    (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign +
                        binaryAddOffset,
                    binaryAddR, pregLoop);

                Div<float, &divHighPrecMode>(binaryAddQ, binaryAddQ, colsNum, pregMain);
                Div<float, &divHighPrecMode>(binaryAddR, binaryAddR, colsNum, pregLoop);

                Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                ReduceSum(vlMean, binaryAddQ, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)binaryAddAddr + binaryAddRemainderLoop - 1), vlMean, pregMerge);
            }
            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                if constexpr (IS_BIAS_BROADCAST) {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, x, pregMain,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                        (i + binaryAddRemainderLoop) * vlFp32);
                } else {
                    LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                        x1Addr, x2Addr, biasAddr, x, pregMain,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign,
                        (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
                }
                StoreRegToOutput(xOutAddr, x, pregMain, (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
                DataCopy(
                    (__local_mem__ float*)x32Addr + (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign, x,
                    pregMain);
                Div<float, &divHighPrecMode>(x, x, colsNum, pregMain);
                ReduceSum(vlMean, x, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)binaryAddAddr + binaryAddRemainderLoop + i), vlMean, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoopMean = binaryAddLoopMean;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoopMean = curBinaryAddLoopMean / 2;
                for (uint16_t j = 0; j < curBinaryAddLoopMean; j++) {
                    DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddAddr + j * vlFp32));
                    DataCopy(binaryAddR, ((__local_mem__ float*)binaryAddAddr + (j + curBinaryAddLoopMean) * vlFp32));
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    DataCopy(((__local_mem__ float*)binaryAddAddr + j * vlFp32), binaryAddQ, pregMain);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
            {
                uint32_t sreg2 = binaryAddLastNum;
                pregLoop = UpdateMask<float>(sreg2);
                DataCopy(meanTemp, ((__local_mem__ float*)binaryAddAddr));
                ReduceSum(mean, meanTemp, pregLoop);
            }

            // batch mean
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)meanAddr + k), mean, pregMerge);
            Duplicate(mean, mean, pregMain);
            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

            uint32_t sreg1 = binaryAddRemainder;
            for (uint16_t i = 0; i < (uint16_t)(binaryAddRemainderLoop - 1); i++) {
                pregLoop = UpdateMask<float>(sreg1);
                DataCopy(binaryAddQ, (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign);
                DataCopy(
                    binaryAddR, (__local_mem__ float*)x32Addr + i * vlFp32 + k * colsPerLoopAlign + binaryAddOffset);
                Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
                Sub(binaryAddR, binaryAddR, mean, pregLoop);
                Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
                Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);

                Div<float, &divHighPrecMode>(binaryAddQPow, binaryAddQPow, colsNum, pregLoop);
                Div<float, &divHighPrecMode>(binaryAddRPow, binaryAddRPow, colsNum, pregLoop);

                Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
                ReduceSum(vlVar, binaryAddQPow, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)binaryAddAddr + i), vlVar, pregMerge);
            }
            {
                pregLoop = UpdateMask<float>(sreg1);
                DataCopy(
                    binaryAddQ,
                    (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 + k * colsPerLoopAlign);
                DataCopy(
                    binaryAddR, (__local_mem__ float*)x32Addr + (binaryAddRemainderLoop - 1) * vlFp32 +
                                    k * colsPerLoopAlign + binaryAddOffset);
                Sub(binaryAddQ, binaryAddQ, mean, pregMain);
                Sub(binaryAddR, binaryAddR, mean, pregLoop);
                Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregMain);
                Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);

                Div<float, &divHighPrecMode>(binaryAddQPow, binaryAddQPow, colsNum, pregMain);
                Div<float, &divHighPrecMode>(binaryAddRPow, binaryAddRPow, colsNum, pregLoop);

                Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregMain);
                ReduceSum(vlVar, binaryAddQPow, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)binaryAddAddr + binaryAddRemainderLoop - 1), vlVar, pregMerge);
            }
            for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                DataCopy(
                    x1, (__local_mem__ float*)x32Addr + (i + binaryAddRemainderLoop) * vlFp32 + k * colsPerLoopAlign);
                Sub(y1, x1, mean, pregMain);
                Mul(y1Pow, y1, y1, pregMain);
                Div<float, &divHighPrecMode>(y1Pow, y1Pow, colsNum, pregMain);
                ReduceSum(vlVar, y1Pow, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)binaryAddAddr + binaryAddRemainderLoop + i), vlVar, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            uint16_t curBinaryAddLoopVar = binaryAddLoopVar;
            for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                curBinaryAddLoopVar = curBinaryAddLoopVar / 2;
                for (uint16_t j = 0; j < curBinaryAddLoopVar; j++) {
                    DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddAddr + j * vlFp32));
                    DataCopy(binaryAddR, ((__local_mem__ float*)binaryAddAddr + (j + curBinaryAddLoopVar) * vlFp32));
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    DataCopy(((__local_mem__ float*)binaryAddAddr + j * vlFp32), binaryAddQ, pregMain);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
            {
                uint32_t sreg2 = binaryAddLastNum;
                pregLoop = UpdateMask<float>(sreg2);
                DataCopy(varTemp, ((__local_mem__ float*)binaryAddAddr));
                ReduceSum(var, varTemp, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)varAddr + k), var, pregMerge);
            }
            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
        }
    }
}

__aicore__ inline void VFCalcRstd(LocalTensor<float>& varLocal, uint32_t rowsCount, uint32_t vlFp32, float eps)
{
    __local_mem__ float* varAddr = (__local_mem__ float*)varLocal[0].GetPhyAddr();

    uint16_t rowsLoopCount = CEIL_DIV(rowsCount, vlFp32);

    __VEC_SCOPE__
    {
        RegTensor<float> r;
        RegTensor<float> y;
        RegTensor<float> s;
        RegTensor<float> t;
        RegTensor<float> e;
        RegTensor<float> scalar1;
        RegTensor<float> scalarInf;
        RegTensor<float> scalarZero;
        RegTensor<float> t1;
        RegTensor<float> t2;
        RegTensor<float> t3;
        RegTensor<float> t4;
        RegTensor<float> var;
        RegTensor<float> rstd;

        RegTensor<float> one;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        Duplicate(one, (float)1.0, pregMain);

        MaskReg cmpRegZero;
        MaskReg cmpRegInf;
        MaskReg pregLoop;
        uint32_t sreg0 = rowsCount;
        for (uint16_t j = 0; j < rowsLoopCount; j++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(var, ((__local_mem__ float*)varAddr + j * vlFp32));
            Duplicate(scalar1, float(SCALAR3), pregLoop);
            Duplicate(scalarInf, POS_INF, pregLoop);
            Duplicate(scalarZero, ZERO_ME, pregLoop);
            Duplicate(t1, float(SCALAR2), pregLoop);
            Duplicate(s, float(1.0), pregLoop);

            // rstd
            Adds(var, var, eps, pregLoop);
            Div(r, one, var, pregLoop);
            Sqrt(y, r, pregLoop);
            Muls(t, var, float(SCALAR1), pregLoop);
            Mul(t, t, y, pregLoop);                // -0.5 * x * y
            Mula(t1, t, y, pregLoop);              // 1.5 + (-0.5 * x * y) * y
            Mul(rstd, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
            Muls(t3, var, float(-1.0), pregLoop);  // -1 * x
            Mula(s, t3, r, pregLoop);              // 1 + (-1) * x * r
            Muls(t4, rstd, float(-1.0), pregLoop); // (-1) * y
            Mula(r, t4, rstd, pregLoop);           // r + (-1) * y * y
            Mula(s, var, r, pregLoop);             // s + x * t
            Mul(s, s, rstd, pregLoop);             // e * y
            Mula(rstd, s, scalar1, pregLoop);      // y + y * e * 0.5
            CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
            Select(rstd, scalarZero, rstd, cmpRegZero);
            CompareScalar(cmpRegInf, var, ZERO_ME, pregLoop);
            Select(rstd, scalarInf, rstd, cmpRegInf);
            DataCopy(((__local_mem__ float*)varAddr + j * vlFp32), rstd, pregLoop);
        }
    }
}

template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFWelfordParallelUpdateWithInit(
    LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local, LocalTensor<X1_TYPE>& biasLocal,
    LocalTensor<X1_TYPE>& xLocal, LocalTensor<float>& tmpMeanLocal, LocalTensor<float>& tmpVarLocal, uint64_t calLen,
    uint16_t loopCount, float scale)
{
    __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
    __local_mem__ X1_TYPE* x2Addr = (__local_mem__ X1_TYPE*)x2Local[0].GetPhyAddr();
    __local_mem__ X1_TYPE* xOutAddr = (__local_mem__ X1_TYPE*)xLocal[0].GetPhyAddr();
    __local_mem__ float* tmpMeanAddr = (__local_mem__ float*)tmpMeanLocal.GetPhyAddr();
    __local_mem__ float* tmpVarAddr = (__local_mem__ float*)tmpVarLocal.GetPhyAddr();

    __local_mem__ X1_TYPE* biasAddr;
    if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
        biasAddr = (__local_mem__ X1_TYPE*)biasLocal[0].GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delta3;
        RegTensor<float> delat4;
        MaskReg pregLoop;
        uint32_t sreg0 = calLen;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                x1Addr, x2Addr, biasAddr, x1, pregLoop, i * VL_FP32, i * VL_FP32, i * VL_FP32);
            StoreRegToOutput(xOutAddr, x1, pregLoop, i * VL_FP32);
            Duplicate(tmpMean, 0.0, pregLoop);
            Sub(delta1, x1, tmpMean, pregLoop);
            Muls(delta2, delta1, scale, pregLoop);
            Add(tmpMean, tmpMean, delta2, pregLoop);
            DataCopy(tmpMeanAddr + i * VL_FP32, tmpMean, pregLoop);

            Duplicate(tmpVar, 0.0, pregLoop);
            Sub(delta3, x1, tmpMean, pregLoop);
            Mul(delat4, delta1, delta3, pregLoop);
            Add(tmpVar, tmpVar, delat4, pregLoop);
            DataCopy(tmpVarAddr + i * VL_FP32, tmpVar, pregLoop);
        }
    }
}

/*
  Welford update 阶段计算公式如下:
  count += 1
  delta = new_value - mean
  mean += (delta / count)
  delta2 = new_value - mean
  var += delta * delta2
  return count, mean, var
*/
template <typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFWelfordParallelUpdate(
    LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local, LocalTensor<X1_TYPE>& biasLocal,
    LocalTensor<X1_TYPE>& xLocal, LocalTensor<float>& tmpMeanLocal, LocalTensor<float>& tmpVarLocal, uint64_t calLen,
    uint16_t loopCount, float scale)
{
    __local_mem__ X1_TYPE* x1Addr = (__local_mem__ X1_TYPE*)x1Local[0].GetPhyAddr();
    __local_mem__ X1_TYPE* x2Addr = (__local_mem__ X1_TYPE*)x2Local[0].GetPhyAddr();
    __local_mem__ X1_TYPE* xOutAddr = (__local_mem__ X1_TYPE*)xLocal[0].GetPhyAddr();
    __local_mem__ float* tmpMeanAddr = (__local_mem__ float*)tmpMeanLocal.GetPhyAddr();
    __local_mem__ float* tmpVarAddr = (__local_mem__ float*)tmpVarLocal.GetPhyAddr();
    __local_mem__ X1_TYPE* biasAddr;
    if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
        biasAddr = (__local_mem__ X1_TYPE*)biasLocal[0].GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delta3;
        RegTensor<float> delat4;
        MaskReg pregLoop;
        uint32_t sreg0 = calLen;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputsToReg<X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
                x1Addr, x2Addr, biasAddr, x1, pregLoop, i * VL_FP32, i * VL_FP32, i * VL_FP32);
            StoreRegToOutput(xOutAddr, x1, pregLoop, i * VL_FP32);
            DataCopy(tmpMean, tmpMeanAddr + i * VL_FP32);
            Sub(delta1, x1, tmpMean, pregLoop);
            Muls(delta2, delta1, scale, pregLoop);
            Add(tmpMean, tmpMean, delta2, pregLoop);
            DataCopy(tmpMeanAddr + i * VL_FP32, tmpMean, pregLoop);

            DataCopy(tmpVar, tmpVarAddr + i * VL_FP32);
            Sub(delta3, x1, tmpMean, pregLoop);
            Mul(delat4, delta1, delta3, pregLoop);
            Add(tmpVar, tmpVar, delat4, pregLoop);
            DataCopy(tmpVarAddr + i * VL_FP32, tmpVar, pregLoop);
        }
    }
}

/*
  Welford Finalize对齐场景计算公式如下:
  finalize_mean = sum_fun(mean) / parallel_N
  finalize_delta = mean - finalize_mean
  finalize_delta_square = finalize_delta * finalize_delta
  M2_fixed = M2 + float(count) * finalize_delta_square
  finalize_std = sum_fun(M2_fixed) / float(parallel_N * count)

  welford采用二分累加计算mean和variance, 基本逻辑为:
  先将尾块折叠到整块上，整尾块vadd之后，做一次vcadd回刷到UB上，剩余整块直接vcadd回刷到UB上，最后对UB上的结果做完全二分对折
*/
__aicore__ inline void VFWelfordParallelFinalizeAlign(
    LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal, LocalTensor<float>& tmpMeanLocal,
    LocalTensor<float>& tmpVarLocal, LocalTensor<float>& dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    float reduceScale, float scale, float cnt, float eps)
{
    __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
    __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal[0].GetPhyAddr();
    __local_mem__ float* tmpMeanAddr = (__local_mem__ float*)tmpMeanLocal[0].GetPhyAddr();
    __local_mem__ float* tmpVarAddr = (__local_mem__ float*)tmpVarLocal.GetPhyAddr();
    __local_mem__ float* dichotomyAddAddr = (__local_mem__ float*)dichotomyAddLocal.GetPhyAddr();

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
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0 = dichotomyAddReminder;
        // 计算mean
        // PART1: 整尾块合并
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanAddr + i * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanAddr + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, scale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddAddr + i, mean, pregMerge);
        }

        // PART2: 整块剩余部分vcadd回刷UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount); i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddAddr + dichotomyAddReminderLoopCount + i, mean, pregMerge);
        }

        DichotomyAdd(mean, dichotomyAddAddr, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanAddr + offset, mean, pregMerge);

        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        sreg0 = dichotomyAddReminder;
        // PART1: 整尾块合并
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanAddr + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarAddr + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

            DataCopy(dichotomyAddMeanR, tmpMeanAddr + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            DataCopy(dichotomyAddVarR, tmpVarAddr + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddAddr + i, var, pregMerge);
        }

        // PART2: 整块剩余部分vcadd回刷UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount); i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddAddr + dichotomyAddReminderLoopCount + i, var, pregMerge);
        }

        DichotomyAdd(var, dichotomyAddAddr, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        CalRstdByHighPrecision(var, rstd, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + offset, rstd, pregMerge);
    }
}

/*
  // Welford Finalize非对齐场景计算公式如下:
  counts = np.ones(len(mean), dtype=np.float32)*count
  for i in range(tail_size):
      counts[i] += 1
  finalize_mean = np.sum(mean*counts) / np.sum(counts)
  finalize_delta = mean - finalize_mean
  finalize_delta_square = finalize_delta * finalize_delta
  M2_fixed = M2 + counts * finalize_delta_square
  finalize_std = np.sum(M2_fixed) / np.sum(counts)

  // Welford Finalize非对齐场景下，二分累加存在如下几种场景:
  首先,非对齐场景下存在两种尾块类型
  1. welford自身的整块和尾块，需要注意的是，welford自身的整块也可能非对齐，整块+尾块=并行度
  2. 二分累加的整块和尾块
  3.
  3.1 welford整块小于二分累加整块，这种场景下，可以进一步细分为两种场景
  3.1.1 welford整块小于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
  3.1.2 welford整块大于等于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现剩余整块回刷UB逻辑中
  3.2 welford整块大于等于二分累加整块，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
*/

// welford整块大于等于二分累加整块
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation1(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCnt = cnt + float(1.0);
    float coeff = tailCnt / cnt;
    float tailCountScale = tailCnt * reduceScale;
    float countScale = cnt * reduceScale;

    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderRealLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

    uint32_t welfordDiff = tailSize - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplit =
        dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 - welfordDiffReminderAlign;
    uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminderAfterSplit, VL_FP32);
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
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0;

        // 整块使用tailCountScale,尾块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, tailCountScale, pregMain);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, mean, pregMerge);
        }

        // 处理welford第一次非对齐点, 整块使用tailCountScale,尾块部分使用tailCountScale, 部分使用countScale
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        uint32_t sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Muls(tmp, dichotomyAddMeanR, coeff, pregLoop1);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanR, tmp, pregLoop1);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
        }

        // 整块使用tailCountScale,尾块使用countScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            DataCopy(
                dichotomyAddMeanR,
                tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
        }
        DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        // 计算rstd
        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
            Mul(deltaR, deltaR, deltaR, pregMain);
            Muls(deltaR, deltaR, tailCnt, pregMain);

            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMain);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, var, pregMerge);
        }
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            Muls(tmp, deltaR, coeff, pregLoop1);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaR, tmp, pregLoop1);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(
                dichotomyAddMeanR,
                tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(
                dichotomyAddVarR,
                tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);
            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
        }
        DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        CalRstdByHighPrecision(var, rstd, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

// welford整块小于二分累加整块，并且小于等于二分累加尾块向上对齐
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation2(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCnt = cnt + float(1.0);
    float coeff = tailCnt / cnt;
    float tailCountScale = tailCnt * reduceScale;
    float countScale = cnt * reduceScale;

    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = tailSize / VL_FP32;
    uint32_t welfordDiffReminder = tailSize - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint16_t dichotomyAddReminderRealLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplit =
        dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 - welfordDiffReminderAlign;
    uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminderAfterSplit, VL_FP32);
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
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0;
        uint32_t sreg1;

        // 整块使用tailCountScale,尾块使用countScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregMain);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, mean, pregMerge);
        }

        // 处理welford第一次非对齐点, 尾块使用countScale,整块部分使用tailCountScale, 部分使用countScale
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Muls(tmp, dichotomyAddMeanL, coeff, pregLoop1);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop1);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
        }

        // 整块使用countScale,尾块使用countScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            DataCopy(
                dichotomyAddMeanR,
                tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用countScale
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
        }
        DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        // 计算rstd
        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
            Mul(deltaR, deltaR, deltaR, pregMain);
            Muls(deltaR, deltaR, cnt, pregMain);

            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMain);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, var, pregMerge);
        }
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            Muls(tmp, deltaL, coeff, pregLoop1);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop1);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(
                dichotomyAddMeanR,
                tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(
                dichotomyAddVarR,
                tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);
            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
        }

        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount); i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
        }
        DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        CalRstdByHighPrecision(var, rstd, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

// 场景3：welford整块小于二分累加整块，并且大于二分累加尾块向上对齐
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation3(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCnt = cnt + float(1.0);
    float coeff = tailCnt / cnt;
    float tailCountScale = tailCnt * reduceScale;
    float countScale = cnt * reduceScale;

    // 二分累加
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
    uint32_t dichotomyAddReminderRoundUp = dichotomyAddReminderLoopCount * VL_FP32;

    uint32_t welfordDiff = tailSize - dichotomyAddReminderRoundUp;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;
    uint16_t dichotomyAddPowerRemainLoopCount =
        dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount - welfordDiffLoopCount - welfordReminderLoopCount;
    uint32_t dichotomyAddPowerOffset =
        dichotomyAddReminderRoundUp + welfordDiffLoopCount * VL_FP32 + welfordDiffReminderAlign;

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
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0 = dichotomyAddReminder;
        // 整块使用tailCountScale, 尾块使用CountScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, mean, pregMerge);
        }

        // 剩余整块需要拆分成多部分
        // 整块剩余部分回刷UB，整块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
        }

        sreg0 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(
                dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(tmp, dichotomyAddMeanL, coeff, pregLoop);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, mean, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + welfordReminderLoopCount + i,
                mean, pregMerge);
        }

        DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        // 计算rstd
        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        // 整块使用tailCountScale, 尾块使用CountScale
        sreg0 = dichotomyAddReminder;
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, var, pregMerge);
        }

        // 整块剩余部分回刷UB，整块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
        }

        sreg0 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(
                dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            Muls(tmp, deltaL, coeff, pregLoop);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop);
            DataCopy(
                dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, var, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + welfordReminderLoopCount + i,
                var, pregMerge);
        }

        DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        CalRstdByHighPrecision(var, rstd, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

__aicore__ inline void VFWelfordParallelFinalizeNonAlign(
    LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal, LocalTensor<float>& tmpMeanLocal,
    LocalTensor<float>& tmpVarLocal, LocalTensor<float>& dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal[0].GetPhyAddr();
    __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal[0].GetPhyAddr();
    __local_mem__ float* tmpMeanAddr = (__local_mem__ float*)tmpMeanLocal[0].GetPhyAddr();
    __local_mem__ float* tmpVarAddr = (__local_mem__ float*)tmpVarLocal.GetPhyAddr();
    __local_mem__ float* dichotomyAddAddr = (__local_mem__ float*)dichotomyAddLocal.GetPhyAddr();

    // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint32_t dichotomyAddReminderRoundUp = CEIL_DIV(dichotomyAddReminder, VL_FP32) * VL_FP32;
    if (tailSize >= dichotomyAddPower) {
        VFWelfordParallelFinalizeNonAlignSituation1(
            meanAddr, rstdAddr, tmpMeanAddr, tmpVarAddr, dichotomyAddAddr, reduceCount, dichotomyAddPower,
            dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
        return;
    }
    if (tailSize <= dichotomyAddReminderRoundUp) {
        VFWelfordParallelFinalizeNonAlignSituation2(
            meanAddr, rstdAddr, tmpMeanAddr, tmpVarAddr, dichotomyAddAddr, reduceCount, dichotomyAddPower,
            dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
        return;
    }
    VFWelfordParallelFinalizeNonAlignSituation3(
        meanAddr, rstdAddr, tmpMeanAddr, tmpVarAddr, dichotomyAddAddr, reduceCount, dichotomyAddPower, dichotomyAddK,
        dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
}

template <bool COUNT_OUT>
__aicore__ inline void MergeOutScales(
    LocalTensor<float>& tmpLocal, GlobalTensor<float>& ws1Gm, GlobalTensor<float>& ws2Gm,
    GlobalTensor<float>& outScale1Gm, GlobalTensor<float>& outScale2Gm)
{
    if (GetBlockIdx() == 0) {
        constexpr int64_t MAGIC_PAGE_BYTES = 128;
        PipeBarrier<PIPE_ALL>();

        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

        DataCopyExtParams cpInParam;
        cpInParam.blockCount = GetBlockNum();
        cpInParam.blockLen = sizeof(float) * 1;
        cpInParam.srcStride = MAGIC_PAGE_BYTES - sizeof(float);
        cpInParam.dstStride = 0;

        DataCopyPad<float, PaddingMode::Compact>(tmpLocal, ws1Gm, cpInParam, padParams);
        if constexpr (COUNT_OUT) {
            DataCopyPad<float, PaddingMode::Compact>(tmpLocal[256], ws2Gm, cpInParam, padParams);
        }

        event_t eeee = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eeee);
        WaitFlag<HardEvent::MTE2_MTE3>(eeee);

        DataCopyExtParams cpOutParam;
        cpOutParam.blockCount = 1;
        cpOutParam.blockLen = sizeof(float) * GetBlockNum();
        cpOutParam.srcStride = 0;
        cpOutParam.dstStride = 0;

        DataCopyPad(outScale1Gm, tmpLocal, cpOutParam);
        if constexpr (COUNT_OUT) {
            DataCopyPad(outScale2Gm, tmpLocal[256], cpOutParam);
        }
    }
}

} // namespace AddLayerNormQuantRegbase

#endif // ADD_LAYER_NORM_QUANT_REGBASE_COMMON_H
