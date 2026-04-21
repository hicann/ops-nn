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
 * \file add_rms_norm_dynamic_mx_quant_common.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H
#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "../inc/platform.h"
#include "op_kernel/platform_util.h"
#include "add_rms_norm_dynamic_mx_quant_tiling_data.h"

namespace AddRmsNormDynamicMxQuant {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::Add;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t OUT_ALL = 256;
constexpr int64_t MX_STEP_PROCESS_NUM = 256;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint16_t MX_FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t MX_FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t MX_ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MX_FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t MX_FP8_E4M3_MAX = 0x3b124925;

constexpr int32_t NEG_ZERO = 0x80000000;
constexpr float ONE_FOURTH = 0.25;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t NEG_ONE = -1;
constexpr float FOUR = 4.0;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr int64_t MODE_ROUND = 0;
constexpr int64_t MODE_FLOOR = 1;
constexpr int64_t MODE_RINT = 4;

constexpr static uint32_t VL_F32 = platform::GetVRegSize() / sizeof(float); // 64
constexpr static uint32_t VL_B16 = platform::GetVRegSize() / sizeof(half); // 128
constexpr static uint16_t ELEMENT_AFTER_REDUCE = platform::GetVRegSize() / Ops::Base::GetUbBlockSize(); // 8
constexpr static uint32_t BLOCK_F32_ALIGN_NUM = Ops::Base::GetUbBlockSize() / sizeof(float);  // 8
constexpr static uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize();

constexpr int64_t AR_RECOMPUTE_SUM_BUFFER_BYTES = 32;
constexpr int64_t AR_RECOMPUTE_SUM_LEN = AR_RECOMPUTE_SUM_BUFFER_BYTES / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

__aicore__ inline uint64_t CeilAlign(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y * y;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline int64_t GetCacheId(const int64_t idx)
{
    return AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
}

template <typename T_IN>
__aicore__ inline void LoadTensorForDtypeTIn(
    __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
    }
}

template <typename T_OUT>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_OUT, float>::value) {
        DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitB322B16>(xOut, src, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
    }
}

// ===================== Rstd Computation =====================

__aicore__ inline void CalculateRstd(
    LocalTensor<float>& xReduceLocal, LocalTensor<float>& rstdLocal,
    uint32_t curRows, float avgFactor, float epsilon)
{
    static constexpr float POS_INF = 3.40282366920938E+38;
    static constexpr float SCALAR1 = -0.5;
    static constexpr float SCALAR2 = 1.5;
    static constexpr float SCALAR3 = 0.5;
    static constexpr float SCALAR0 = -99.99;

    __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
    uint16_t loopRows = static_cast<uint16_t>((curRows + VL_F32 - 1) / VL_F32);
    __VEC_SCOPE__
    {
        RegTensor<float> var, rstd, r, y, s, t, one, scalar1;
        RegTensor<float> t1, t2, t3, t4, scalarInf, scalarZero;
        MaskReg cmpRegZero, cmpRegInf;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregLoop;

        uint32_t sreg = static_cast<uint32_t>(curRows);
        for (uint16_t i = 0; i < loopRows; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            AscendC::MicroAPI::Duplicate(scalarInf, POS_INF, pregLoop);
            AscendC::MicroAPI::Duplicate(scalarZero, float(0.0), pregLoop);
            AscendC::MicroAPI::Duplicate(one, float(1.0), pregLoop);
            AscendC::MicroAPI::Duplicate(scalar1, SCALAR3, pregLoop);
            AscendC::MicroAPI::Duplicate(t1, SCALAR2, pregLoop);
            AscendC::MicroAPI::Duplicate(s, float(1.0), pregLoop);
            // rstd = 1/sqrt(mean(x^2) + eps)
            AscendC::MicroAPI::DataCopy(var, xReduceUb + i * VL_F32);
            AscendC::MicroAPI::Muls(var, var, avgFactor, pregLoop);
            AscendC::MicroAPI::Adds(var, var, epsilon, pregLoop);
            AscendC::MicroAPI::Maxs(var, var, SCALAR0, pregLoop);
            AscendC::MicroAPI::Div(r, one, var, pregLoop);
            AscendC::MicroAPI::Sqrt(y, r, pregLoop);
            AscendC::MicroAPI::Muls(t, var, SCALAR1, pregLoop);
            AscendC::MicroAPI::Mul(t, t, y, pregLoop);
            AscendC::MicroAPI::Mula(t1, t, y, pregLoop);
            AscendC::MicroAPI::Mul(rstd, y, t1, pregLoop);
            AscendC::MicroAPI::Muls(t3, var, float(-1.0), pregLoop);
            AscendC::MicroAPI::Mula(s, t3, r, pregLoop);
            AscendC::MicroAPI::Muls(t4, rstd, float(-1.0), pregLoop);
            AscendC::MicroAPI::Mula(r, t4, rstd, pregLoop);
            AscendC::MicroAPI::Mula(s, var, r, pregLoop);
            AscendC::MicroAPI::Mul(s, s, rstd, pregLoop);
            AscendC::MicroAPI::Mula(rstd, s, scalar1, pregLoop);
            AscendC::MicroAPI::CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
            AscendC::MicroAPI::Select(rstd, scalarZero, rstd, cmpRegZero);
            AscendC::MicroAPI::CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
            AscendC::MicroAPI::Select(rstd, scalarInf, rstd, cmpRegInf);
            AscendC::MicroAPI::DataCopy(rstdInUb + i * VL_F32, rstd, pregLoop);
        }
    }
}

// ===================== Split-R Common Free Functions =====================

// (x1 + x2)² → xFp32Tmp
template <typename T_X>
__aicore__ inline void MainBlockSquareVF(
    LocalTensor<T_X>& x1Local, LocalTensor<T_X>& x2Local,
    LocalTensor<float>& xFp32Tmp, uint32_t count)
{
    __local_mem__ T_X* x1InUb = (__local_mem__ T_X*)x1Local.GetPhyAddr();
    __local_mem__ T_X* x2InUb = (__local_mem__ T_X*)x2Local.GetPhyAddr();
    __local_mem__ float* xFp32TmpBuf = (__local_mem__ float*)xFp32Tmp.GetPhyAddr();

    uint16_t loops = (count + VL_F32 - 1) / VL_F32;
    uint32_t sreg = count;
    __VEC_SCOPE__
    {
        RegTensor<float> x1Reg, x2Reg, xSum;
        MaskReg pregLoop;
        for (uint16_t vi = 0; vi < loops; ++vi) {
            uint32_t offset = vi * VL_F32;
            pregLoop = UpdateMask<float>(sreg);
            LoadTensorForDtypeTIn<T_X>(x1InUb, x1Reg, pregLoop, offset);
            LoadTensorForDtypeTIn<T_X>(x2InUb, x2Reg, pregLoop, offset);
            AscendC::MicroAPI::Add(xSum, x1Reg, x2Reg, pregLoop);
            AscendC::MicroAPI::Mul(xSum, xSum, xSum, pregLoop);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_NORM_B32>(
                xFp32TmpBuf + offset, xSum, pregLoop);
        }
    }
}

// xFp32Tmp += (x1Fold + x2Fold)²
template <typename T_X>
__aicore__ inline void FoldBlockSquareAddVF(
    LocalTensor<T_X>& x1FoldLocal, LocalTensor<T_X>& x2FoldLocal,
    LocalTensor<float>& xFp32Tmp, uint32_t tailCount)
{
    __local_mem__ T_X* x1FoldInUb = (__local_mem__ T_X*)x1FoldLocal.GetPhyAddr();
    __local_mem__ T_X* x2FoldInUb = (__local_mem__ T_X*)x2FoldLocal.GetPhyAddr();
    __local_mem__ float* xFp32TmpBuf = (__local_mem__ float*)xFp32Tmp.GetPhyAddr();
    uint16_t tailLoops = (tailCount + VL_F32 - 1) / VL_F32;
    uint32_t sregTail = tailCount;
    __VEC_SCOPE__
    {
        RegTensor<float> x1FoldReg, x2FoldReg, foldSquare, mainReg, sum;
        MaskReg pregLoop;
        for (uint16_t i = 0; i < tailLoops; ++i) {
            pregLoop = UpdateMask<float>(sregTail);
            uint32_t offset = i * VL_F32;
            // Fold tile: (x1Fold + x2Fold)²
            LoadTensorForDtypeTIn<T_X>(x1FoldInUb, x1FoldReg, pregLoop, offset);
            LoadTensorForDtypeTIn<T_X>(x2FoldInUb, x2FoldReg, pregLoop, offset);
            AscendC::MicroAPI::Add(x1FoldReg, x1FoldReg, x2FoldReg, pregLoop);
            AscendC::MicroAPI::Mul(foldSquare, x1FoldReg, x1FoldReg, pregLoop);
            // Read back main block result and accumulate
            AscendC::MicroAPI::DataCopy(mainReg, xFp32TmpBuf + offset);
            AscendC::MicroAPI::Add(sum, mainReg, foldSquare, pregLoop);
            AscendC::MicroAPI::Select(sum, sum, mainReg, pregLoop);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_NORM_B32>(
                xFp32TmpBuf + offset, sum, pregLoop);
        }
    }
}

__aicore__ inline void UpdateCache(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const int64_t cacheId, const int64_t stride)
{
    uint16_t innerLoopTimes = cacheId;
    uint32_t innerLoopStride = stride;
    __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
    __local_mem__ float* cache = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheId * stride;
    __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> aReg, bReg;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

        DataCopy(aReg, (__local_mem__ float*)src);
        for (uint16_t j = 0; j < innerLoopTimes; ++j) {
            DataCopy(bReg, dst + j * innerLoopStride);
            AscendC::MicroAPI::Add(aReg, aReg, bReg, pregOne);
        }
        DataCopy((__local_mem__ float*)cache, aReg, pregOne);
    }
}

template <typename T_X>
__aicore__ inline void CalculateXAdd(
    LocalTensor<T_X>& xLocal1, LocalTensor<T_X>& xLocal2,
    LocalTensor<T_X>& xOutLocal, LocalTensor<float>& xFp32Local, uint32_t count)
{
    __local_mem__ T_X* x1InUb = (__local_mem__ T_X*)xLocal1.GetPhyAddr();
    __local_mem__ T_X* x2InUb = (__local_mem__ T_X*)xLocal2.GetPhyAddr();
    __local_mem__ T_X* xOutInUb = (__local_mem__ T_X*)xOutLocal.GetPhyAddr();
    __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

    uint32_t sreg = count;
    uint16_t loopCount = (sreg + VL_F32 - 1) / VL_F32;

    __VEC_SCOPE__
    {
        RegTensor<float> x1, x2, xSum;
        MaskReg pregLoop;
        for (uint16_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * VL_F32;
            pregLoop = UpdateMask<float>(sreg);
            LoadTensorForDtypeTIn<T_X>(x1InUb, x1, pregLoop, offset);
            LoadTensorForDtypeTIn<T_X>(x2InUb, x2, pregLoop, offset);
            AscendC::MicroAPI::Add(xSum, x1, x2, pregLoop);
            StoreTensorForDtypeTOut<T_X>(xOutInUb, xSum, pregLoop, offset);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_NORM_B32>(xFp32Tmp + offset, xSum, pregLoop);
        }
    }
}
template <typename T_X>
__aicore__ inline void MxQuantComputeMaxExpOCP(
    __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T_X> vdExp0;
        AscendC::MicroAPI::RegTensor<T_X> vdExp1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> invalidMaskFP16;
        AscendC::MicroAPI::Duplicate(invalidMaskFP16, INVALID_FLOAT16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg Mask = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg invalidDataMask0;
        AscendC::MicroAPI::MaskReg invalidDataMask1;
        AscendC::MicroAPI::UnalignReg u1;
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::LoadAlign<
                T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            if constexpr (IsSame<T_X, half>::value) {
                AscendC::MicroAPI::And(
                    vdExpSelect0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16, Mask);
                AscendC::MicroAPI::And(
                    vdExpSelect1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16, Mask);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidMaskFP16, Mask);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidMaskFP16, Mask);
                AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, Mask);
                AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, Mask);
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, Mask);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, Mask);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16, Mask);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16, Mask);
            }
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            AscendC::MicroAPI::StoreUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, ELEMENT_AFTER_REDUCE);
        }
        AscendC::MicroAPI::StoreUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename T_Y>
__aicore__ inline void MxQuantComputeScaleOCP(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
    __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
    uint16_t loopNumScale)
{
    uint16_t emax;
    if constexpr (IsSame<T_Y, fp8_e4m3fn_t>::value) {
        emax = MX_FP8_E4M3_MAX_EXP;
    } else if constexpr (IsSame<T_Y, fp8_e5m2_t>::value) {
        emax = MX_FP8_E5M2_MAX_EXP;
    } else if constexpr (IsSame<T_Y, fp4x2_e2m1_t>::value) {
        emax = FP4_E2M1_BF16_MAX_EXP;
    } else {
        emax = FP4_E1M2_MAX_EXP;
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
        AscendC::MicroAPI::Duplicate(maxExpValue, emax);
        AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::RegTensor<uint16_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, VL_B16);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::StoreAlign<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, VL_F32, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, VL_B16, preMaskScale);
        }
    }
}

constexpr uint32_t CUBLAS_ZERO_FOR_ALL = 0x00000000;
constexpr uint32_t CUBLAS_EXP254 = 0x000000fe;
constexpr uint32_t CUBLAS_HALF_FOR_MAN = 0x00400000;

template <typename T_X>
__aicore__ inline void MxQuantComputeMaxExpcuBLAS(
    __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T_X> vdExp0;
        AscendC::MicroAPI::RegTensor<T_X> vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> absMask16Bit;
        AscendC::MicroAPI::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg Mask = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::LoadAlign<
                T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, absMask16Bit, Mask);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, absMask16Bit, Mask);
            AscendC::MicroAPI::Max(
                vdMaxExp, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, Mask);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            AscendC::MicroAPI::StoreUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, ELEMENT_AFTER_REDUCE);
        }
        AscendC::MicroAPI::StoreUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename T_X, typename T_Y>
__aicore__ inline void MxQuantComputeScalecuBLAS(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
    __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
    uint16_t loopNumScale4NV)
{
    uint32_t dtypeMax;
    if constexpr (IsSame<T_Y, fp8_e4m3fn_t>::value) {
        dtypeMax = MX_FP8_E4M3_MAX;
    } else {
        dtypeMax = MX_FP8_E5M2_MAX;
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> max16;
        AscendC::MicroAPI::RegTensor<uint32_t> max32;
        AscendC::MicroAPI::RegTensor<uint32_t> exp32;
        AscendC::MicroAPI::RegTensor<uint32_t> man32;
        AscendC::MicroAPI::RegTensor<uint32_t> normalExp32;
        AscendC::MicroAPI::RegTensor<uint32_t> expAddOne32;
        AscendC::MicroAPI::RegTensor<uint32_t> extractExp;
        AscendC::MicroAPI::RegTensor<uint16_t> expOut;
        AscendC::MicroAPI::RegTensor<uint32_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> recExpOut;
        AscendC::MicroAPI::RegTensor<uint32_t> invMax;
        AscendC::MicroAPI::Duplicate(invMax, dtypeMax);
        AscendC::MicroAPI::RegTensor<uint32_t> manMaskFP32;
        AscendC::MicroAPI::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        AscendC::MicroAPI::RegTensor<uint32_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_FP32);
        AscendC::MicroAPI::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0);
        AscendC::MicroAPI::RegTensor<uint32_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        AscendC::MicroAPI::RegTensor<uint32_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        AscendC::MicroAPI::MaskReg preMaskScale;
        AscendC::MicroAPI::MaskReg maskHalf;
        uint32_t SixtyFour = 64;
        uint32_t ThirtyTwo = 32;
        preMaskScale = AscendC::MicroAPI::CreateMask<uint32_t>();
        maskHalf = AscendC::MicroAPI::UpdateMask<uint16_t>(SixtyFour);
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Float = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < loopNumScale4NV; i++) {
            AscendC::MicroAPI::LoadAlign<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, VL_F32);
            AscendC::MicroAPI::Cast<float, T_X, castTraitHalf2Float>(
                (AscendC::MicroAPI::RegTensor<float>&)max32,
                (AscendC::MicroAPI::RegTensor<T_X>&)max16, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);
            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)max32,
                (AscendC::MicroAPI::RegTensor<float>&)max32,
                (AscendC::MicroAPI::RegTensor<float>&)invMax, preMaskScale);
            AscendC::MicroAPI::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            AscendC::MicroAPI::And(man32, max32, manMaskFP32, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, CUBLAS_ZERO_FOR_ALL, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, CUBLAS_EXP254, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, CUBLAS_ZERO_FOR_ALL, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p1, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p2, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, CUBLAS_ZERO_FOR_ALL, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, CUBLAS_HALF_FOR_MAN, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p1, p1, p2, preMaskScale);
            AscendC::MicroAPI::MaskOr(p0, p0, p1, preMaskScale);
            AscendC::MicroAPI::Adds(expAddOne32, exp32, 1, preMaskScale);
            AscendC::MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expOut, extractExp);
            AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr + i * ThirtyTwo, expOut, maskHalf);
            AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(recExpOut, halfScale);
            AscendC::MicroAPI::StoreAlign<uint16_t>(halfScaleLocalAddr + i * VL_F32, recExpOut, maskHalf);
        }
    }
}

template <AscendC::RoundMode roundMode, typename T_X, typename T_Y>
__aicore__ inline void MxQuantComputeData(
    __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    __ubuf__ int8_t* outLocalAddr, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1 = AscendC::MicroAPI::CreateMask<T_X>();
        AscendC::MicroAPI::MaskReg dataMask2 = AscendC::MicroAPI::CreateMask<T_X>();
        AscendC::MicroAPI::MaskReg dataMask3 = AscendC::MicroAPI::CreateMask<T_X>();
        AscendC::MicroAPI::MaskReg dataMask4 = AscendC::MicroAPI::CreateMask<T_X>();
        AscendC::MicroAPI::MaskReg dataMask5 = AscendC::MicroAPI::CreateMask<T_Y>();
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T_X> vdExp0;
        AscendC::MicroAPI::RegTensor<T_X> vdExp1;
        AscendC::MicroAPI::RegTensor<T_X> vdExp0Convert;
        AscendC::MicroAPI::RegTensor<T_X> vdExp1Convert;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<T_Y> vdExp0FP8Zero;
        AscendC::MicroAPI::RegTensor<T_Y> vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<T_Y> vdExp1FP8Zero;
        AscendC::MicroAPI::RegTensor<T_Y> vdExp1FP8One;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp0FP4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp1FP4;
        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to80 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to81 = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to82 = {
            AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to83 = {
            AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::LoadAlign<
                T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            AscendC::MicroAPI::LoadAlign<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, ELEMENT_AFTER_REDUCE);
            if constexpr (IsSame<T_X, half>::value) {
                AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                    floatScaleForMul, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                AscendC::MicroAPI::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);
                AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                AscendC::MicroAPI::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);
            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T_X, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                AscendC::MicroAPI::Cast<float, T_X, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
            }
            AscendC::MicroAPI::Cast<T_Y, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::MicroAPI::Cast<T_Y, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            AscendC::MicroAPI::Cast<T_Y, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::MicroAPI::Cast<T_Y, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMask4);

            AscendC::MicroAPI::Add(
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8One, dataMask5);
            AscendC::MicroAPI::Add(
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero, dataMask5);
            AscendC::MicroAPI::Add(
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8One, dataMask5);

            AscendC::MicroAPI::StoreAlign<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ALL, dataMask5);
        }
    }
}

template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, typename T_Y>
__aicore__ inline void ComputeFP4FromHalf(AscendC::MicroAPI::RegTensor<float>& Reg)
{
    AscendC::MicroAPI::MaskReg pregAll32 =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg zeroMask;
    AscendC::MicroAPI::MaskReg specialMask;
    AscendC::MicroAPI::MaskReg negInfMask;
    AscendC::MicroAPI::RegTensor<int32_t> negZero;
    AscendC::MicroAPI::RegTensor<int32_t> maxExpFP32;
    AscendC::MicroAPI::RegTensor<int32_t> exp0FP32;
    AscendC::MicroAPI::RegTensor<int32_t> exp1FP32;

    AscendC::MicroAPI::Duplicate(negZero, NEG_ZERO);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::EQ>(
        negInfMask, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, negZero, pregAll32);

    if constexpr (IsSame<T_Y, fp4x2_e1m2_t>::value) {
        AscendC::MicroAPI::Muls(Reg, Reg, FOUR, pregAll32);
        AscendC::MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        AscendC::MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        AscendC::MicroAPI::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        AscendC::MicroAPI::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        AscendC::MicroAPI::And(exp0FP32, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        AscendC::MicroAPI::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        AscendC::MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        AscendC::MicroAPI::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        AscendC::MicroAPI::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        AscendC::MicroAPI::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        AscendC::MicroAPI::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        AscendC::MicroAPI::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        AscendC::MicroAPI::Mul(Reg, Reg, (AscendC::MicroAPI::RegTensor<float>&)exp1FP32, pregAll32);
        AscendC::MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        AscendC::MicroAPI::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        AscendC::MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        AscendC::MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        AscendC::MicroAPI::Mul(Reg, Reg, (AscendC::MicroAPI::RegTensor<float>&)exp0FP32, pregAll32);
    }

    AscendC::MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    AscendC::MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    AscendC::MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    AscendC::MicroAPI::Select<int32_t>(
        (AscendC::MicroAPI::RegTensor<int32_t>&)Reg, negZero, (AscendC::MicroAPI::RegTensor<int32_t>&)Reg,
        zeroMask);
}

template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, typename T_X, typename T_Y>
__aicore__ inline void MxQuantComputeDataFP4(
    __ubuf__ T_X* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T_X> vdExp0;
        AscendC::MicroAPI::RegTensor<T_X> vdExp1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::MicroAPI::RegTensor<T_Y> vdExp0FP4;
        AscendC::MicroAPI::RegTensor<T_Y> vdExp1FP4;
        AscendC::MicroAPI::RegTensor<float> halfScaleForMulFP32;
        AscendC::MicroAPI::RegTensor<float> vdExp0ZeroFP32;
        AscendC::MicroAPI::RegTensor<float> vdExp0OneFP32;
        AscendC::MicroAPI::RegTensor<float> vdExp1ZeroFP32;
        AscendC::MicroAPI::RegTensor<float> vdExp1OneFP32;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0ZeroBF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0OneBF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1ZeroBF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1OneBF16;
        AscendC::MicroAPI::MaskReg dataMaskB16 = AscendC::MicroAPI::CreateMask<half>();
        AscendC::MicroAPI::MaskReg dataMaskB32 = AscendC::MicroAPI::CreateMask<float>();

        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitF16toFp32Zero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitF16toFp32One = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitFp32toBF16 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};

        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T_X>(totalCountInUB);
            AscendC::MicroAPI::LoadAlign<
                T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            AscendC::MicroAPI::LoadAlign<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, ELEMENT_AFTER_REDUCE);

            if constexpr (IsSame<T_X, half>::value) {
                if constexpr (roundMode == RoundMode::CAST_RINT || roundMode == RoundMode::CAST_ROUND) {
                    // tail_axis_optimize_fp16 -> fp4 (rint, round)
                    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitF16toFp32Zero>(
                        halfScaleForMulFP32, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul,
                        dataMaskB16);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitF16toFp32Zero>(
                        vdExp0ZeroFP32, vdExp0, dataMaskB16);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitF16toFp32One>(vdExp0OneFP32, vdExp0, dataMaskB16);
                    AscendC::MicroAPI::Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                    AscendC::MicroAPI::Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
                    ComputeFP4FromHalf<toBf16RoundMode, roundMode, T_Y>(vdExp0ZeroFP32);
                    ComputeFP4FromHalf<toBf16RoundMode, roundMode, T_Y>(vdExp0OneFP32);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                        vdExp0ZeroBF16, vdExp0ZeroFP32, dataMaskB32);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                        vdExp0OneBF16, vdExp0OneFP32, dataMaskB32);
                    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0ZeroBF16,
                        (AscendC::MicroAPI::RegTensor<uint32_t>&)vdExp0ZeroBF16);
                    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0OneBF16,
                        (AscendC::MicroAPI::RegTensor<uint32_t>&)vdExp0OneBF16);
                    AscendC::MicroAPI::Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitF16toFp32Zero>(
                        vdExp1ZeroFP32, vdExp1, dataMaskB16);
                    AscendC::MicroAPI::Cast<float, T_X, castTraitF16toFp32One>(vdExp1OneFP32, vdExp1, dataMaskB16);
                    AscendC::MicroAPI::Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                    AscendC::MicroAPI::Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
                    ComputeFP4FromHalf<toBf16RoundMode, roundMode, T_Y>(vdExp1ZeroFP32);
                    ComputeFP4FromHalf<toBf16RoundMode, roundMode, T_Y>(vdExp1OneFP32);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                        vdExp1ZeroBF16, vdExp1ZeroFP32, dataMaskB32);
                    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                        vdExp1OneBF16, vdExp1OneFP32, dataMaskB32);
                    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1ZeroBF16,
                        (AscendC::MicroAPI::RegTensor<uint32_t>&)vdExp1ZeroBF16);
                    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1OneBF16,
                        (AscendC::MicroAPI::RegTensor<uint32_t>&)vdExp1OneBF16);
                    AscendC::MicroAPI::Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);
                    AscendC::MicroAPI::Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
                    AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp0FP4, vdExp0ZeroBF16, dataMask1);
                    AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp1FP4, vdExp1ZeroBF16, dataMask1);
                } else {
                    // for fp16 -> fp4 (floor)
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                    AscendC::MicroAPI::Cast<bfloat16_t, T_X, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                    AscendC::MicroAPI::Mul(
                        vdExp0BF16, vdExp0BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul,
                        dataMask1);
                    AscendC::MicroAPI::Mul(
                        vdExp1BF16, vdExp1BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul,
                        dataMask1);
                    AscendC::MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                    AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                    AscendC::MicroAPI::Cast<T_Y, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
                }
            } else {
                // for bf16
                AscendC::MicroAPI::Mul(
                    vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(
                    vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T_X>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                AscendC::MicroAPI::Cast<T_Y, T_X, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<T_Y, T_X, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }

            AscendC::MicroAPI::StoreAlign<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            AscendC::MicroAPI::StoreAlign<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
        }
    }
}

} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H