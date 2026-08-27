/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file situ_mx_quant_common.h
 * \brief Common definitions and shared regbase impl for Situ + MX quantization
 */

#ifndef SITU_MX_QUANT_COMMON_H
#define SITU_MX_QUANT_COMMON_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace SituMxQuant {
// ==================== Type Traits ====================
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

// ==================== Constants ====================
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint32_t NAN_CUSTOMIZATION_FP32 = 0x7f810000;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t EXP_MASK_FP16 = 0x7c00;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t THRESHOLD_E2M1 = 0x0100;
constexpr uint16_t THRESHOLD_E1M2 = 0x0080;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint32_t SPECIAL_EXP_THRESHOLD_FP32 = 0x00400000;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint32_t FP4_E2M1_FP32_MAX_EXP = 0x01000000;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
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
constexpr uint32_t FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925;
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint32_t ZERO_FOR_ALL = 0x00000000;
constexpr uint32_t EXP_254 = 0x000000fe;
constexpr uint32_t HALF_FOR_MAN = 0x00400000;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t X_ONCE_NUM = 512;
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128;
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr int64_t CONST_64 = 64;
constexpr int64_t CONST_32 = 32;
constexpr int64_t CONST_2 = 2;
constexpr int64_t CONST_4 = 4;
constexpr uint32_t VF_LEN_T = Ops::Base::GetVRegSize() / sizeof(half);     // 128
constexpr uint32_t VF_LEN_FP32 = Ops::Base::GetVRegSize() / sizeof(float); // 64
constexpr uint32_t ONE_BLOCK_UB = Ops::Base::GetUbBlockSize();
constexpr uint32_t ONE_BLOCK_NUM = ONE_BLOCK_UB / sizeof(half); // 16

// ==================== Cast Traits ====================
static constexpr AscendC::Reg::CastTrait CAST_ZERO = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait CAST_ONE = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_HALF_TO_BF16 = {
    AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_80 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_81 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_82 = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_83 = {AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

using namespace AscendC;

// ===================================================================
// MxQuant helper: Extract BF16 exponent and compute per-32-block max
// Adapted from swiglu_mx_quant_common.h (BF16-only path)
// ===================================================================
template <typename T>
__aicore__ inline void ComputeVfMaxExpVfLast(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, int64_t dim0OnceSize,
                                             int64_t alignDim1Size)
{
    uint32_t totalCountInUB = dim0OnceSize * alignDim1Size;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    uint16_t maxExpbf16 = MAX_EXP_FOR_BF16;
    uint16_t invalidFp16 = INVALID_FLOAT16;
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vdExp0, vdExp1;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
        AscendC::Reg::RegTensor<uint16_t> vdExpSelect0, vdExpSelect1;
        AscendC::Reg::RegTensor<uint16_t> expMaskBF16, vdMaxExp;
        AscendC::Reg::Duplicate(expMaskBF16, maxExpbf16);
        AscendC::Reg::RegTensor<uint16_t> invalidmaskfp16;
        AscendC::Reg::Duplicate(invalidmaskfp16, invalidFp16);
        AscendC::Reg::MaskReg scaleMask1, invalidDataMask0, invalidDataMask1;
        AscendC::Reg::UnalignRegForStore u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceNum);
            if constexpr (IsSame<T, half>::value) {
                // FP16 path: check for Inf/NaN, then cast to BF16 for exponent extraction
                AscendC::Reg::And(vdExpSelect0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, invalidmaskfp16,
                                  scaleMask1);
                AscendC::Reg::And(vdExpSelect1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, invalidmaskfp16,
                                  scaleMask1);
                AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidmaskfp16,
                                                             scaleMask1);
                AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidmaskfp16,
                                                             scaleMask1);
                AscendC::Reg::Cast<bfloat16_t, T, CAST_HALF_TO_BF16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::Reg::Cast<bfloat16_t, T, CAST_HALF_TO_BF16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16,
                                  scaleMask1);
                AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16,
                                  scaleMask1);
                AscendC::Reg::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::Reg::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                // BF16 path: direct exponent extraction
                AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
            }
            AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::Reg::ReduceDataBlock<ReduceType::MAX>(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, u1,
                                                                                              scaleNum);
        }
        AscendC::Reg::StoreUnAlignPost(maxExpAddr, u1, 0);
    }
}

// ===================================================================
// MxQuant helper: Compute E8M0 scale and reciprocal scale (OCP algorithm)
// Adapted from swiglu_mx_quant_common.h
// ===================================================================
template <typename T>
__aicore__ inline void ComputeScaleLast(uint16_t fEmax, __ubuf__ uint16_t* maxExpAddr,
                                        __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                        int64_t dim0OnceSize, int64_t alignDim1Size)
{
    uint32_t totalScaleInUB = dim0OnceSize * (alignDim1Size / CONST_32);
    uint16_t loopNumScale = CeilDivision(totalScaleInUB, QUANT_ONCE_NUM_FP4);
    uint16_t maxExpBf16 = MAX_EXP_FOR_BF16;
    int64_t onceNum = QUANT_ONCE_NUM_FP4;
    int64_t onceNumMxScale = CONST_64;
    uint16_t bf16ExpBias = BF16_EXP_BIAS;
    uint16_t maxExpFp8 = MAX_EXP_FOR_FP8;
    uint16_t nanCustomZation = NAN_CUSTOMIZATION;
    uint16_t specailExpThreshold = SPECIAL_EXP_THRESHOLD;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> expMask, vdMaxExp;
        AscendC::Reg::Duplicate(expMask, maxExpBf16);
        AscendC::Reg::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue, sharedExp, scaleValue, scaleBias, halfScale;
        AscendC::Reg::Duplicate(maxExpValue, fEmax);
        AscendC::Reg::Duplicate(scaleBias, bf16ExpBias);
        AscendC::Reg::RegTensor<uint16_t> fp8NanRegTensor, zeroRegTensor, nanRegTensor;
        AscendC::Reg::Duplicate(fp8NanRegTensor, maxExpFp8);
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::Duplicate(nanRegTensor, nanCustomZation);
        AscendC::Reg::MaskReg invalidDataMask, specialDataMask;
        AscendC::Reg::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::Reg::Duplicate(specialExpRegTensor, specailExpThreshold);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                           onceNum);
            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                             onceNumMxScale, preMaskScale);
            AscendC::Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, onceNum, preMaskScale);
        }
    }
}

// ===================================================================
// MxQuant helper: Quantize FP16/BF16 data to FP8 (multiply by reciprocal scale, then cast)
// Adapted from swiglu_mx_quant_common.h (FP16 and BF16 paths)
// ===================================================================
template <typename T, typename U>
__aicore__ inline void ComputeDataF8Last(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                         __ubuf__ int8_t* outLocalAddr, int64_t dim0OnceSize, int64_t dim1AlignSize)
{
    uint32_t totalCountInUB = dim0OnceSize * dim1AlignSize;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<float> floatScaleForMul;
        AscendC::Reg::RegTensor<T> vdExp0, vdExp1;
        AscendC::Reg::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One;
        AscendC::Reg::RegTensor<float> vdExp1FP32Zero, vdExp1FP32One;
        AscendC::Reg::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One;
        AscendC::Reg::RegTensor<U> vdExp1FP8Zero, vdExp1FP8One;
        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB8 = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceXNum);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                          elementAfterReduce);
            if constexpr (IsSame<T, half>::value) {
                // FP16 path: cast data to FP32 and scale (BF16 bit pattern) to FP32, then multiply.
                // halfScale is assembled as a BF16 bit pattern by ComputeScaleLast; reinterpreting it
                // as FP16 would decode a wrong value (e.g. BF16 0x4400=512.0 vs FP16 0x4400=4.0).
                AscendC::Reg::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
                AscendC::Reg::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_ZERO>(
                    floatScaleForMul, (AscendC::Reg::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                AscendC::Reg::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, maskAll);
                AscendC::Reg::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, maskAll);
                AscendC::Reg::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
                AscendC::Reg::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
                AscendC::Reg::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, maskAll);
                AscendC::Reg::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, maskAll);
            } else {
                // BF16 path: multiply in BF16 domain, then cast to FP32, then to FP8
                AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, maskAll);
                AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, maskAll);
                AscendC::Reg::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
                AscendC::Reg::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
                AscendC::Reg::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
                AscendC::Reg::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
            }
            AscendC::Reg::Cast<U, float, CAST_32_TO_80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::Reg::Cast<U, float, CAST_32_TO_82>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::Reg::Cast<U, float, CAST_32_TO_81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::Reg::Cast<U, float, CAST_32_TO_83>(vdExp1FP8One, vdExp1FP32One, maskAll);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8One, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp1FP8Zero, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp1FP8One, maskAllB8);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8Zero, onceXNum, maskAllB8);
        }
    }
}

// ===================================================================
// MxQuant helper: Compute FP4 from half/bf16 via FP32 intermediate
// Adapted from swiglu_mx_quant_common.h ComputeFP4FromHalf
// ===================================================================
template <typename U, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg, Reg::MaskReg& pregAll32)
{
    Reg::MaskReg zeroMask, specialMask, negInfMask;
    Reg::RegTensor<int32_t> negZero, maxExpFP32, exp0FP32, exp1FP32;
    Reg::Duplicate(negZero, NEG_ZERO);
    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        Reg::Muls(Reg, Reg, FOUR, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        Reg::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::Compares<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::And(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::Or(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}

// ===================================================================
// MxQuant helper: Quantize data to FP4 (multiply by reciprocal scale, then cast)
// Adapted from swiglu_mx_quant_common.h ComputeDataF4Last
// ===================================================================
template <typename T, typename U, AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeDataF4Last(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                         __ubuf__ int8_t* outLocalAddr, int64_t dim0OnceSize, int64_t dim1AlignSize)
{
    uint32_t totalCountInUB = dim0OnceSize * dim1AlignSize;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    int64_t onceYNum = OUT_ELE_NUM_ONE_BLK;
    static constexpr AscendC::Reg::CastTrait castTrait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr AscendC::Reg::CastTrait castTraitHalf2Bf16 = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        toBf16RoundMode};
    static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, roundMode};
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg dataMask1;
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<T> vdExp0, vdExp1;
        AscendC::Reg::RegTensor<U> vdExp0FP4, vdExp1FP4;
        Reg::RegTensor<float> halfScaleForMulFP32;
        Reg::RegTensor<float> vdExp0ZeroFP32, vdExp0OneFP32, vdExp1ZeroFP32, vdExp1OneFP32;
        Reg::RegTensor<bfloat16_t> vdExp0ZeroBF16, vdExp0OneBF16, vdExp1ZeroBF16, vdExp1OneBF16;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceXNum);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                          elementAfterReduce);
            if constexpr (IsSame<T, half>::value) {
                // FP16 path: cast to FP32, multiply by reciprocal, ComputeFP4FromHalf, cast back to BF16
                Reg::Cast<float, T, CAST_ZERO>(vdExp0ZeroFP32, vdExp0, dataMaskB16);
                Reg::Cast<float, T, CAST_ONE>(vdExp0OneFP32, vdExp0, dataMaskB16);
                Reg::Cast<float, T, CAST_ZERO>(vdExp1ZeroFP32, vdExp1, dataMaskB16);
                Reg::Cast<float, T, CAST_ONE>(vdExp1OneFP32, vdExp1, dataMaskB16);
                Reg::Cast<float, bfloat16_t, CAST_ZERO>(halfScaleForMulFP32,
                                                        (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);
                Reg::Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<U, roundMode>(vdExp0ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp0ZeroBF16, vdExp0ZeroFP32, dataMaskB32);
                Reg::Cast<float, bfloat16_t, CAST_ONE>(halfScaleForMulFP32,
                                                       (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);
                Reg::Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<U, roundMode>(vdExp0OneFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp0OneBF16, vdExp0OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)vdExp0ZeroBF16,
                                                                        (Reg::RegTensor<uint32_t>&)vdExp0ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)vdExp0OneBF16,
                                                                        (Reg::RegTensor<uint32_t>&)vdExp0OneBF16);
                Reg::Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);

                Reg::Cast<float, bfloat16_t, CAST_ZERO>(halfScaleForMulFP32,
                                                        (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);
                Reg::Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<U, roundMode>(vdExp1ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp1ZeroBF16, vdExp1ZeroFP32, dataMaskB32);
                Reg::Cast<float, bfloat16_t, CAST_ONE>(halfScaleForMulFP32,
                                                       (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);
                Reg::Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<U, roundMode>(vdExp1OneFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp1OneBF16, vdExp1OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)vdExp1ZeroBF16,
                                                                        (Reg::RegTensor<uint32_t>&)vdExp1ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)vdExp1OneBF16,
                                                                        (Reg::RegTensor<uint32_t>&)vdExp1OneBF16);
                Reg::Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);
                Reg::Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0ZeroBF16, dataMaskB16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1ZeroBF16, dataMaskB16);
            } else {
                // BF16 path: multiply in BF16 domain, then cast to FP4
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T>&)halfScaleForMul, dataMaskB16);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T>&)halfScaleForMul, dataMaskB16);
                Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMaskB16);
                Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMaskB16);
            }
            // FP4 output: 0.5 byte per element, two halves stored with POST_MODE_UPDATE
            // Matches swiglu_mx_quant_common.h ComputeDataF4Last store pattern
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP4, onceYNum, dataMaskB16);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP4, onceYNum, dataMaskB16);
        }
    }
}

// ===================================================================
// Situ activation: beta * tanh(gate / beta) * sigmoid(gate) * up
//                  (+ optional linear_beta * tanh(up / linear_beta) on up)
// Replaces ComputeVfSwigluV1 from swiglu_mx_quant
// ===================================================================
template <typename T, bool hasLinearBeta>
__aicore__ inline void ComputeVfSitu(__ubuf__ T* gateUbAddr, __ubuf__ T* upUbAddr, __ubuf__ T* situUbAddr,
                                     int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize, float beta,
                                     float invBeta, float linearBeta, float invLinearBeta)
{
    uint16_t dim0VfTimes = dim0OnceSize;
    uint16_t dim1VfTimes = dim1OnceSize / VF_LEN_FP32;
    uint32_t dim1Tail = dim1OnceSize % VF_LEN_FP32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    uint32_t alignDim1In = ((dim1OnceSize + ONE_BLOCK_NUM - 1) / ONE_BLOCK_NUM) * ONE_BLOCK_NUM;
    uint32_t alignDim1Out = dim1AlignSize;
    auto gateUbAddr1 = gateUbAddr;
    auto upUbAddr1 = upUbAddr;
    auto situUbAddr1 = situUbAddr;
    auto situUbAddr2 = situUbAddr;
    T numZero = 0;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = alignDim1Out - dim1VfTimes * VF_LEN_FP32;
        if (padNum <= VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = VF_LEN_FP32;
            mask3Num = padNum - VF_LEN_FP32;
        }
        int32_t offsetAlgin = dim1VfTimes * VF_LEN_FP32;
        gateUbAddr1 = gateUbAddr + offsetAlgin;
        upUbAddr1 = upUbAddr + offsetAlgin;
        situUbAddr1 = situUbAddr + offsetAlgin;
        situUbAddr2 = situUbAddr + offsetAlgin + dim1TailTimes * VF_LEN_FP32;
    }
    float scalarOne = 1.0f;
    float negScalarOne = -1.0f;
    float scalarTwo = 2.0f;
    float negTwo = -2.0f;
    // Two-path tanh (adapted from tanh.h reference):
    //   |x| < 0.6:  degree-9 polynomial, FMA Horner (matches tanh.h exactly)
    //   |x| >= 0.6: sigmoid decomposition, sign naturally preserved
    float tanhC1 = -0.333327681f;
    float tanhC2 = 0.133152977f;
    float tanhC3 = -0.0523039624f;
    float tanhC4 = 0.0157396831f;
    float tanhThreshold = 0.6f;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vregGate;
        AscendC::Reg::RegTensor<T> vregUp;
        AscendC::Reg::RegTensor<float> gateF;
        AscendC::Reg::RegTensor<float> upF;
        AscendC::Reg::RegTensor<float> gateDivBeta;
        AscendC::Reg::RegTensor<float> polyReg; // sigmoid path result
        AscendC::Reg::RegTensor<float> x2;      // x² for Horner / temp
        AscendC::Reg::MaskReg cmpMask;          // comparison result for Select
        AscendC::Reg::RegTensor<float> negGate;
        AscendC::Reg::RegTensor<float> expReg; // sigmoid Exp
        AscendC::Reg::RegTensor<float> oneReg;
        AscendC::Reg::RegTensor<float> sigmoidReg; // sigmoid result / linear_beta work reg
        AscendC::Reg::RegTensor<float> c1Reg;      // tanh polynomial coeff c1 (preloaded)
        AscendC::Reg::RegTensor<float> c2Reg;      // tanh polynomial coeff c2 (preloaded)
        AscendC::Reg::RegTensor<float> outFReg;
        AscendC::Reg::RegTensor<T> outTReg;
        AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg mask1 = AscendC::Reg::UpdateMask<float>(mask1Num);
        AscendC::Reg::MaskReg mask2 = AscendC::Reg::UpdateMask<float>(mask2Num);
        AscendC::Reg::MaskReg mask3 = AscendC::Reg::UpdateMask<T>(mask3Num);
        AscendC::Reg::Duplicate(oneReg, scalarOne);
        AscendC::Reg::Duplicate(c1Reg, tanhC1);
        AscendC::Reg::Duplicate(c2Reg, tanhC2);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::Reg::AddrReg srcIdxOffset = AscendC::Reg::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In,
                                                                                    dim1vfLoopIdx, 64);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregGate, gateUbAddr, srcIdxOffset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregUp, upUbAddr, srcIdxOffset);
                AscendC::Reg::Cast<float, T, CAST_ZERO>(gateF, vregGate, mask);
                AscendC::Reg::Cast<float, T, CAST_ZERO>(upF, vregUp, mask);

                // Two-path tanh(gate/beta) — adapted from tanh.h reference:
                //   small |x|: degree-7 polynomial (Horner)
                //   large |x|: sigmoid decomposition on |x|, sign restore
                AscendC::Reg::Muls(gateDivBeta, gateF, invBeta, mask); // x = gate/beta

                // --- Polynomial path (all x, used for |x| < 0.6) ---
                // tanh(x) ≈ x * (1 + c1*x² + c2*x⁴ + c3*x⁶ + c4*x⁸)
                // FMA Horner (matches tanh.h reference: 7 ops, 7 roundings)
                AscendC::Reg::Mul(x2, gateDivBeta, gateDivBeta, mask);                    // x²
                AscendC::Reg::Muls(sigmoidReg, x2, tanhC4, mask);                         // c4*x²
                AscendC::Reg::Adds(sigmoidReg, sigmoidReg, tanhC3, mask);                 // +c3
                AscendC::Reg::FusedMulDstAdd(sigmoidReg, x2, c2Reg, mask);                // *x²+c2
                AscendC::Reg::FusedMulDstAdd(sigmoidReg, x2, c1Reg, mask);                // *x²+c1
                AscendC::Reg::Mul(sigmoidReg, sigmoidReg, x2, mask);                      // *x²
                AscendC::Reg::FusedMulDstAdd(sigmoidReg, gateDivBeta, gateDivBeta, mask); // *x+x = x*(p+1)

                // --- Sigmoid path (used for |x| >= 0.6) ---
                // tanh(x) = 2*sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1, sign naturally preserved
                AscendC::Reg::Muls(negGate, gateDivBeta, negTwo, mask); // -2x
                AscendC::Reg::Exp(expReg, negGate, mask);
                AscendC::Reg::Adds(expReg, expReg, scalarOne, mask);      // 1+exp(-2x)
                AscendC::Reg::Div(polyReg, oneReg, expReg, mask);         // sigmoid = 1/(1+exp(-2x))
                AscendC::Reg::Muls(polyReg, polyReg, scalarTwo, mask);    // 2*sigmoid
                AscendC::Reg::Adds(polyReg, polyReg, negScalarOne, mask); // 2*sigmoid - 1

                // --- Path selection: sigmoid if |x| >= 0.6, else polynomial ---
                AscendC::Reg::Muls(x2, gateDivBeta, negScalarOne, mask);
                AscendC::Reg::Max(x2, gateDivBeta, x2, mask);   // |x|
                AscendC::Reg::Duplicate(expReg, tanhThreshold); // 0.6
                AscendC::Reg::Compare<float, CMPMODE::GE>(cmpMask, x2, expReg, mask);
                AscendC::Reg::Select(sigmoidReg, polyReg, sigmoidReg, cmpMask);
                // sigmoidReg = tanh(gate/beta) — save to negGate (free after |x|)
                AscendC::Reg::Mul(negGate, sigmoidReg, oneReg, mask); // negGate = tanh

                // sigmoid(gate) = 1 / (1 + exp(-gate))  → result in polyReg
                AscendC::Reg::Muls(polyReg, gateF, negScalarOne, mask); // -gate
                AscendC::Reg::Exp(expReg, polyReg, mask);
                AscendC::Reg::Adds(expReg, expReg, scalarOne, mask);
                AscendC::Reg::Div(polyReg, oneReg, expReg, mask); // sigmoid(gate)

                // situ_a = beta * tanh * sigmoid
                AscendC::Reg::Mul(polyReg, negGate, polyReg, mask); // tanh * sigmoid
                AscendC::Reg::Muls(polyReg, polyReg, beta, mask);   // * beta

                // Optional: up = linear_beta * tanh(up / linear_beta)
                // Uses sigmoidReg/negGate as work registers to preserve polyReg (situ_a)
                if constexpr (hasLinearBeta) {
                    AscendC::Reg::Muls(upF, upF, invLinearBeta, mask); // x = up/lb

                    // Poly path → sigmoidReg (FMA Horner)
                    AscendC::Reg::Mul(x2, upF, upF, mask);
                    AscendC::Reg::Muls(sigmoidReg, x2, tanhC4, mask);
                    AscendC::Reg::Adds(sigmoidReg, sigmoidReg, tanhC3, mask);
                    AscendC::Reg::FusedMulDstAdd(sigmoidReg, x2, c2Reg, mask);
                    AscendC::Reg::FusedMulDstAdd(sigmoidReg, x2, c1Reg, mask);
                    AscendC::Reg::Mul(sigmoidReg, sigmoidReg, x2, mask);
                    AscendC::Reg::FusedMulDstAdd(sigmoidReg, upF, upF, mask);

                    // Sigmoid path on x (sign naturally preserved) → negGate
                    AscendC::Reg::Muls(expReg, upF, negTwo, mask); // -2x
                    AscendC::Reg::Exp(expReg, expReg, mask);
                    AscendC::Reg::Adds(expReg, expReg, scalarOne, mask); // 1+exp(-2x)
                    AscendC::Reg::Div(negGate, oneReg, expReg, mask);
                    AscendC::Reg::Muls(negGate, negGate, scalarTwo, mask);
                    AscendC::Reg::Adds(negGate, negGate, negScalarOne, mask); // 2*sig-1

                    // Path selection → sigmoidReg = tanh(up/lb)
                    AscendC::Reg::Muls(x2, upF, negScalarOne, mask);
                    AscendC::Reg::Max(x2, upF, x2, mask); // |x|
                    AscendC::Reg::Duplicate(expReg, tanhThreshold);
                    AscendC::Reg::Compare<float, CMPMODE::GE>(cmpMask, x2, expReg, mask);
                    AscendC::Reg::Select(sigmoidReg, negGate, sigmoidReg, cmpMask);

                    AscendC::Reg::Muls(upF, sigmoidReg, linearBeta, mask);
                }

                // situOut = situ_a * up
                AscendC::Reg::Mul(outFReg, polyReg, upF, mask);

                AscendC::Reg::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::Reg::AddrReg outOffset = AscendC::Reg::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out,
                                                                                 dim1vfLoopIdx, 64);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(situUbAddr, outTReg, outOffset,
                                                                                    mask);
            }
            // Handle tail elements
            AscendC::Reg::AddrReg srcIdxOffset1 = AscendC::Reg::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In);
            AscendC::Reg::AddrReg outOffset1 = AscendC::Reg::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out);
            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregGate, gateUbAddr1,
                                                                                    srcIdxOffset1);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregUp, upUbAddr1, srcIdxOffset1);
                AscendC::Reg::Cast<float, T, CAST_ZERO>(gateF, vregGate, mask1);
                AscendC::Reg::Cast<float, T, CAST_ZERO>(upF, vregUp, mask1);

                // Two-path tanh(gate/beta) — tail path
                AscendC::Reg::Muls(gateDivBeta, gateF, invBeta, mask1);

                // Poly path → sigmoidReg (FMA Horner)
                AscendC::Reg::Mul(x2, gateDivBeta, gateDivBeta, mask1);
                AscendC::Reg::Muls(sigmoidReg, x2, tanhC4, mask1);
                AscendC::Reg::Adds(sigmoidReg, sigmoidReg, tanhC3, mask1);
                AscendC::Reg::FusedMulDstAdd(sigmoidReg, x2, c2Reg, mask1);
                AscendC::Reg::FusedMulDstAdd(sigmoidReg, x2, c1Reg, mask1);
                AscendC::Reg::Mul(sigmoidReg, sigmoidReg, x2, mask1);
                AscendC::Reg::FusedMulDstAdd(sigmoidReg, gateDivBeta, gateDivBeta, mask1);

                // Sigmoid path on x → polyReg: tanh(x) = 2/(1+exp(-2x)) - 1
                AscendC::Reg::Muls(negGate, gateDivBeta, negTwo, mask1);
                AscendC::Reg::Exp(expReg, negGate, mask1);
                AscendC::Reg::Adds(expReg, expReg, scalarOne, mask1);
                AscendC::Reg::Div(polyReg, oneReg, expReg, mask1);
                AscendC::Reg::Muls(polyReg, polyReg, scalarTwo, mask1);
                AscendC::Reg::Adds(polyReg, polyReg, negScalarOne, mask1);

                // Path selection → sigmoidReg = tanh
                AscendC::Reg::Muls(x2, gateDivBeta, negScalarOne, mask1);
                AscendC::Reg::Max(x2, gateDivBeta, x2, mask1); // |x|
                AscendC::Reg::Duplicate(expReg, tanhThreshold);
                AscendC::Reg::Compare<float, CMPMODE::GE>(cmpMask, x2, expReg, mask1);
                AscendC::Reg::Select(sigmoidReg, polyReg, sigmoidReg, cmpMask);
                AscendC::Reg::Mul(negGate, sigmoidReg, oneReg, mask1); // save tanh

                // sigmoid(gate) → polyReg
                AscendC::Reg::Muls(polyReg, gateF, negScalarOne, mask1);
                AscendC::Reg::Exp(expReg, polyReg, mask1);
                AscendC::Reg::Adds(expReg, expReg, scalarOne, mask1);
                AscendC::Reg::Div(polyReg, oneReg, expReg, mask1);

                // situ_a = beta * tanh * sigmoid
                AscendC::Reg::Mul(polyReg, negGate, polyReg, mask1);
                AscendC::Reg::Muls(polyReg, polyReg, beta, mask1);

                // Optional: up = linear_beta * tanh(up / linear_beta)
                if constexpr (hasLinearBeta) {
                    AscendC::Reg::Muls(upF, upF, invLinearBeta, mask1);

                    AscendC::Reg::Mul(x2, upF, upF, mask1);
                    AscendC::Reg::Muls(sigmoidReg, x2, tanhC4, mask1);
                    AscendC::Reg::Adds(sigmoidReg, sigmoidReg, tanhC3, mask1);
                    AscendC::Reg::FusedMulDstAdd(sigmoidReg, x2, c2Reg, mask1);
                    AscendC::Reg::FusedMulDstAdd(sigmoidReg, x2, c1Reg, mask1);
                    AscendC::Reg::Mul(sigmoidReg, sigmoidReg, x2, mask1);
                    AscendC::Reg::FusedMulDstAdd(sigmoidReg, upF, upF, mask1);

                    // Sigmoid path on x → negGate: tanh(x) = 2/(1+exp(-2x)) - 1
                    AscendC::Reg::Muls(expReg, upF, negTwo, mask1);
                    AscendC::Reg::Exp(expReg, expReg, mask1);
                    AscendC::Reg::Adds(expReg, expReg, scalarOne, mask1);
                    AscendC::Reg::Div(negGate, oneReg, expReg, mask1);
                    AscendC::Reg::Muls(negGate, negGate, scalarTwo, mask1);
                    AscendC::Reg::Adds(negGate, negGate, negScalarOne, mask1);

                    // Path selection → sigmoidReg = tanh(up/lb)
                    AscendC::Reg::Muls(x2, upF, negScalarOne, mask1);
                    AscendC::Reg::Max(x2, upF, x2, mask1); // |x|
                    AscendC::Reg::Duplicate(expReg, tanhThreshold);
                    AscendC::Reg::Compare<float, CMPMODE::GE>(cmpMask, x2, expReg, mask1);
                    AscendC::Reg::Select(sigmoidReg, negGate, sigmoidReg, cmpMask);

                    AscendC::Reg::Muls(upF, sigmoidReg, linearBeta, mask1);
                }

                // situOut = situ_a * up
                AscendC::Reg::Mul(outFReg, polyReg, upF, mask1);

                AscendC::Reg::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(situUbAddr1, outTReg, outOffset1,
                                                                                    mask2);
            }
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                AscendC::Reg::Duplicate<T>(vregGate, numZero);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(situUbAddr2, vregGate, outOffset1,
                                                                                    mask3);
            }
        }
    }
}

} // namespace SituMxQuant

#endif // SITU_MX_QUANT_COMMON_H
