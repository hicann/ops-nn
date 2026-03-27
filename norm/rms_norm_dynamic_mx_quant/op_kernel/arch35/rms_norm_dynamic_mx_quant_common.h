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
 * \file rms_norm_dynamic_mx_quant_common.h
 * \brief Common utilities for RmsNormDynamicMxQuant kernel (arch35)
 */

#ifndef RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H
#define RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../inc/kernel_utils.h"

namespace RmsNormDynamicMxQuantNs {

using namespace AscendC;
using namespace AscendC::MicroAPI;

#define FLOAT_OVERFLOW_MODE_CTRL 60

constexpr uint16_t VECTOR_LENGTH = static_cast<int64_t>(Ops::Base::GetVRegSize());
constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
constexpr uint32_t VL_B16 = VECTOR_LENGTH / sizeof(half);
constexpr int64_t UB_BLOCK_SIZE = static_cast<int64_t>(Ops::Base::GetUbBlockSize());
constexpr int64_t UB_BLOCK_SIZE_FP32 = UB_BLOCK_SIZE / sizeof(float);
constexpr uint16_t VL_BLOCK_NUM = VECTOR_LENGTH / UB_BLOCK_SIZE;

constexpr int64_t DOUBLE_BUFFER = 2;

constexpr float RMS_POS_INF = 3.40282366920938E+38;
constexpr float RMS_ZERO = 0.0f;
constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t OUT_ALL = 256;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int64_t MODE_ROUND = 0;
constexpr int64_t MODE_FLOOR = 1;
constexpr int64_t MODE_RINT = 4;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t NEG_ONE = -1;
constexpr float FOUR = 4.0;
constexpr float ONE_FOURTH = 0.25;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr uint16_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344的float32表示 57334是E5M2所能表示的最大值
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;

constexpr static CastTrait CAST_B16_TO_B32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr static CastTrait CAST_B32_TO_B16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

__aicore__ inline void ComputeRstd(
    LocalTensor<float> xReduceTmpLocal, LocalTensor<float> rstdLocal, uint64_t curUbFactor, float epsilon,
    float avgFactor)
{
    __local_mem__ float* rstdLocalUbAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();
    uint16_t aLoop = static_cast<uint16_t>((curUbFactor + VL_FP32 - 1) / VL_FP32);
    __VEC_SCOPE__
    {
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> var;
        RegTensor<float> one;
        RegTensor<float> r;
        RegTensor<float> y;
        RegTensor<float> s;
        RegTensor<float> t;
        RegTensor<float> scalar1;
        RegTensor<float> scalarInf;
        RegTensor<float> scalarZero;
        RegTensor<float> t1;
        RegTensor<float> t2;
        RegTensor<float> t3;
        RegTensor<float> t4;
        RegTensor<float> rstd;

        MaskReg cmpRegZero;
        MaskReg cmpRegInf;
        MaskReg pregLoop;

        Duplicate(one, 1.0, pregMain);
        uint32_t sreg0 = static_cast<uint32_t>(curUbFactor);
        for (uint16_t a = 0; a < aLoop; a++) {
            pregLoop = UpdateMask<float>(sreg0);
            Duplicate(scalar1, float(0.5), pregLoop);
            Duplicate(scalarInf, RMS_POS_INF, pregLoop);
            Duplicate(scalarZero, RMS_ZERO, pregLoop);
            Duplicate(t1, float(1.5), pregLoop);
            Duplicate(s, float(1.0), pregLoop);

            // rstd
            DataCopy(var, xReduceTmpLocalUbAddr + a * VL_FP32);
            Muls(var, var, avgFactor, pregLoop);
            Adds(var, var, epsilon, pregLoop);
            Div(r, one, var, pregLoop);
            Sqrt(y, r, pregLoop);
            Muls(t, var, float(-0.5), pregLoop);
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
            CompareScalar(cmpRegZero, var, RMS_POS_INF, pregLoop);
            Select(rstd, scalarZero, rstd, cmpRegZero);
            CompareScalar(cmpRegInf, var, RMS_ZERO, pregLoop);
            Select(rstd, scalarInf, rstd, cmpRegInf);
            DataCopy(rstdLocalUbAddr + a * VL_FP32, rstd, pregLoop);
        }
    }
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, CAST_B16_TO_B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeT(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T> xOut;
        Cast<T, float, CAST_B32_TO_B16>(xOut, src, preg);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
    }
}

template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, typename T1, typename T2>
__aicore__ inline void ComputeData(
    __ubuf__ T1* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB,
    uint16_t loopNum)
{
    uint32_t totalCountInUB2 = totalCountInUB * DIGIT_TWO;
    uint16_t elementAfterReduce = VL_BLOCK_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::MaskReg dataMask2;
        AscendC::MicroAPI::MaskReg dataMask3;
        AscendC::MicroAPI::MaskReg dataMask4;
        AscendC::MicroAPI::MaskReg dataMask5;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T1> vdExp0;
        AscendC::MicroAPI::RegTensor<T1> vdExp1;
        AscendC::MicroAPI::RegTensor<T1> vdExp0Convert;
        AscendC::MicroAPI::RegTensor<T1> vdExp1Convert;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<T2> vdExp0FP8Zero;
        AscendC::MicroAPI::RegTensor<T2> vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<T2> vdExp1FP8Zero;
        AscendC::MicroAPI::RegTensor<T2> vdExp1FP8One;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp0FP4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp1FP4;
        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
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
        dataMask1 = AscendC::MicroAPI::CreateMask<T1>();
        dataMask2 = AscendC::MicroAPI::CreateMask<T1>();
        dataMask3 = AscendC::MicroAPI::CreateMask<T1>();
        dataMask4 = AscendC::MicroAPI::CreateMask<T1>();
        dataMask5 = AscendC::MicroAPI::CreateMask<T2>();
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopy<
                T1, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            if constexpr (IsSameType<T1, half>::value) {
                AscendC::MicroAPI::Cast<float, T1, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T1, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(
                    floatScaleForMul, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                AscendC::MicroAPI::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);

                AscendC::MicroAPI::Cast<float, T1, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                AscendC::MicroAPI::Cast<float, T1, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                AscendC::MicroAPI::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);

            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T1>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T1>&)halfScaleForMul, dataMask1);

                AscendC::MicroAPI::Cast<float, T1, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T1, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<float, T1, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                AscendC::MicroAPI::Cast<float, T1, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
            }
            AscendC::MicroAPI::Cast<T2, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::MicroAPI::Cast<T2, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            AscendC::MicroAPI::Cast<T2, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::MicroAPI::Cast<T2, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMask4);

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

            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ALL, dataMask5);
        }
    }
    return;
}

template <typename T>

__aicore__ inline void ComputeMaxExpOCP(
    __ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    uint16_t elementAfterReduce = VL_BLOCK_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
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
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::MaskReg invalidDataMask0;
        AscendC::MicroAPI::MaskReg invalidDataMask1;
        AscendC::MicroAPI::UnalignReg u1;
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            if constexpr (IsSameType<T, half>::value) {
                AscendC::MicroAPI::And(
                    vdExpSelect0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpSelect1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16, scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidMaskFP16, scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidMaskFP16, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
            }

            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

template <typename T1, typename T2>
__aicore__ inline void ComputeScaleOCP(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    uint16_t dtypeEmax = 0;
    if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
        dtypeEmax = FP8_E4M3_MAX_EXP;
    } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
        dtypeEmax = FP8_E5M2_MAX_EXP;
    } else if constexpr (IsSameType<T2, fp4x2_e2m1_t>::value) {
        dtypeEmax = FP4_E2M1_BF16_MAX_EXP;
    } else {
        dtypeEmax = FP4_E1M2_MAX_EXP;
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;

        AscendC::MicroAPI::RegTensor<T1> vdExp0;
        AscendC::MicroAPI::RegTensor<T1> vdExp1;

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg cmpResultSub;
        AscendC::MicroAPI::MaskReg preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
        AscendC::MicroAPI::Duplicate(maxExpValue, dtypeEmax);
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
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, VL_B16);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, VL_B16 / DIGIT_TWO, preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, VL_B16, preMaskScale);
        }
    }
    return;
}

template <typename T>
__aicore__ inline void ComputeMaxExpcuBLAS(
    __ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    uint16_t elementAfterReduce = VL_BLOCK_NUM;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> absMask16Bit;
        AscendC::MicroAPI::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                absMask16Bit, scaleMask1);
            AscendC::MicroAPI::And(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1,
                absMask16Bit, scaleMask1);
            AscendC::MicroAPI::Max(
                vdMaxExp, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}
template <typename T1, typename T2>
__aicore__ inline void ComputeScalecuBLAS(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale4NV)
{
    uint32_t zeroForAll = 0x00000000;
    uint32_t Exp254 = 0x000000fe;
    uint32_t halfForMan = 0x00400000;

    uint32_t dtypeMax = 0; //算法暂不支持fp4量化
    if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
        dtypeMax = FP8_E4M3_MAX;
    } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
        dtypeMax = FP8_E5M2_MAX;
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
        preMaskScale = AscendC::MicroAPI::CreateMask<uint32_t>();
        maskHalf = AscendC::MicroAPI::CreateMask<uint16_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Float = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < loopNumScale4NV; i++) {
            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, VL_FP32);

            AscendC::MicroAPI::Cast<float, T1, castTraitHalf2Float>(
                (AscendC::MicroAPI::RegTensor<float>&)max32, (AscendC::MicroAPI::RegTensor<T1>&)max16, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);

            AscendC::MicroAPI::Mul(
                (AscendC::MicroAPI::RegTensor<float>&)max32, (AscendC::MicroAPI::RegTensor<float>&)max32,
                (AscendC::MicroAPI::RegTensor<float>&)invMax, preMaskScale);
            AscendC::MicroAPI::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            AscendC::MicroAPI::And(man32, max32, manMaskFP32, preMaskScale);

            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, Exp254, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p1, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p2, preMaskScale);

            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, halfForMan, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p1, p1, p2, preMaskScale);
            AscendC::MicroAPI::MaskOr(p0, p0, p1, preMaskScale);

            AscendC::MicroAPI::Adds(expAddOne32, exp32, 1, preMaskScale);
            AscendC::MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expOut, extractExp);

            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, expOut, VL_FP32 / DIGIT_TWO, maskHalf);

            AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(recExpOut, halfScale);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, recExpOut, VL_FP32, maskHalf);
        }
    }
    return;
}

template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, typename T2>
__aicore__ inline void ComputeFP4FromHalf(MicroAPI::RegTensor<float>& Reg)
{
    MicroAPI::MaskReg pregAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg zeroMask;
    MicroAPI::MaskReg specialMask;
    MicroAPI::MaskReg negInfMask;

    MicroAPI::RegTensor<int32_t> negZero;
    MicroAPI::RegTensor<int32_t> maxExpFP32;
    MicroAPI::RegTensor<int32_t> exp0FP32;
    MicroAPI::RegTensor<int32_t> exp1FP32;

    MicroAPI::Duplicate(negZero, NEG_ZERO);
    MicroAPI::Compare<int32_t, CMPMODE::EQ>(negInfMask, (MicroAPI::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<T2, fp4x2_e1m2_t>::value) {
        MicroAPI::Muls(Reg, Reg, FOUR, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        MicroAPI::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        MicroAPI::And(exp0FP32, (MicroAPI::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        MicroAPI::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        MicroAPI::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float>&)exp1FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float>&)exp0FP32, pregAll32);
    }
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    MicroAPI::Select<int32_t>(
        (MicroAPI::RegTensor<int32_t>&)Reg, negZero, (MicroAPI::RegTensor<int32_t>&)Reg, zeroMask);
}

template <typename T2>
__aicore__ inline void FP16Convert(
    AscendC::MicroAPI::RegTensor<half>& output, AscendC::MicroAPI::RegTensor<half>& input,
    AscendC::MicroAPI::MaskReg& mask)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> specialValueTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> newMantissa;
        AscendC::MicroAPI::RegTensor<uint16_t> andResult;
        AscendC::MicroAPI::RegTensor<uint16_t> newValue;
        AscendC::MicroAPI::MaskReg specialMask;
        AscendC::MicroAPI::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (IsSameType<T2, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        AscendC::MicroAPI::Duplicate(specialValueTensor, specialValue);
        AscendC::MicroAPI::Duplicate(newMantissa, NEW_MANTISSA);
        AscendC::MicroAPI::And(andResult, (AscendC::MicroAPI::RegTensor<uint16_t>&)input, specialValueTensor, mask);
        AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        AscendC::MicroAPI::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        AscendC::MicroAPI::Or(newValue, (AscendC::MicroAPI::RegTensor<uint16_t>&)input, newMantissa, mask);
        AscendC::MicroAPI::Select<uint16_t>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)output, newValue, (AscendC::MicroAPI::RegTensor<uint16_t>&)input,
            specialMask);
    }
    return;
}

template <typename T>
__aicore__ inline void DeletePadData(
    __ubuf__ int8_t* outLocalAddr, __ubuf__ int8_t* inLocalAddr, uint16_t rowNum, uint32_t colNum, uint32_t colStride)
{
    uint16_t mloops = static_cast<uint16_t>(rowNum);
    uint32_t vl_b8 = static_cast<uint32_t>(VECTOR_LENGTH);
    uint32_t nNum = static_cast<uint32_t>(colNum);
    uint32_t inStride = static_cast<uint32_t>(colStride);
    if constexpr (IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, fp4x2_e1m2_t>::value) {
        nNum = nNum / 2;
        inStride = inStride / 2;
    }
    uint16_t nloops = static_cast<uint16_t>(ops::CeilDiv(nNum, vl_b8));
    uint32_t remain = nNum - (nloops - 1) * vl_b8;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;

        for (uint16_t i = 0; i < mloops; i++) {
            __ubuf__ int8_t* currInAddr = inLocalAddr + i * inStride;
            __ubuf__ int8_t* currOutAddr = outLocalAddr + i * nNum;
            for (uint16_t j = 0; j < static_cast<uint16_t>(nloops - 1); j++) {
                AscendC::MicroAPI::DataCopyUnAlignPre(uIn, currInAddr);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    inputRegTensor, uIn, currInAddr, vl_b8);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    currOutAddr, inputRegTensor, uOut, vl_b8);
                AscendC::MicroAPI::DataCopyUnAlignPost(currOutAddr, uOut, 0);
            }

            AscendC::MicroAPI::DataCopyUnAlignPre(uIn, currInAddr);
            AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                inputRegTensor, uIn, currInAddr, remain);
            AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                currOutAddr, inputRegTensor, uOut, remain);
            AscendC::MicroAPI::DataCopyUnAlignPost(currOutAddr, uOut, 0);
        }
    }
    return;
}

__aicore__ inline void PadScaleData(
    __ubuf__ int8_t* outLocalAddr, __ubuf__ int8_t* inLocalAddr, uint16_t rowNum, uint32_t colNum)
{
    uint16_t mloops = static_cast<uint16_t>(rowNum);
    uint32_t vl_b8 = static_cast<uint32_t>(VECTOR_LENGTH);
    uint32_t nNum = static_cast<uint32_t>(colNum);
    uint16_t nloops = static_cast<uint16_t>(ops::CeilDiv(nNum, vl_b8));
    uint32_t outStride = static_cast<uint32_t>(nNum + 1);
    uint32_t remain = nNum - (nloops - 1) * vl_b8;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;
        AscendC::MicroAPI::RegTensor<int8_t> zeroRegTensor;
        AscendC::MicroAPI::Duplicate<int8_t>(zeroRegTensor, 0);
        for (uint16_t i = 0; i < mloops; i++) {
            __ubuf__ int8_t* currInAddr = inLocalAddr + i * nNum;
            __ubuf__ int8_t* currOutAddr = outLocalAddr + i * outStride;
            for (uint16_t j = 0; j < static_cast<uint16_t>(nloops - 1); j++) {
                AscendC::MicroAPI::DataCopyUnAlignPre(uIn, currInAddr);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    inputRegTensor, uIn, currInAddr, vl_b8);
                AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    currOutAddr, inputRegTensor, uOut, vl_b8);
                AscendC::MicroAPI::DataCopyUnAlignPost(currOutAddr, uOut, 0);
            }

            AscendC::MicroAPI::DataCopyUnAlignPre(uIn, currInAddr);
            AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                inputRegTensor, uIn, currInAddr, remain);
            AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                currOutAddr, inputRegTensor, uOut, remain);
            AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                currOutAddr, zeroRegTensor, uOut, 1);
            AscendC::MicroAPI::DataCopyUnAlignPost(currOutAddr, uOut, 0);
        }
    }
    return;
}

} // namespace RmsNormDynamicMxQuantNs

#endif // RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H
