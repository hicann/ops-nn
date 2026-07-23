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
 * \file mx_quant_common.h
 * \brief
 */

#ifndef QUANT_COMMON_H
#define QUANT_COMMON_H

namespace MxQuantCommon {
using namespace AscendC;

constexpr uint64_t TPL_SCALE_ALG_0 = 0;
constexpr uint64_t TPL_SCALE_ALG_1 = 1;
constexpr uint64_t TPL_SCALE_ALG_2 = 2;
constexpr uint64_t TPL_DST_TYPE_MAX_0 = 0; // normal
constexpr uint64_t TPL_DST_TYPE_MAX_1 = 1; // 0,6
constexpr uint64_t TPL_DST_TYPE_MAX_2 = 2; // 7
constexpr uint64_t TPL_DST_TYPE_MAX_3 = 3; // 1.875
constexpr uint64_t TPL_DST_TYPE_0 = 0;     // fp8
constexpr uint64_t TPL_DST_TYPE_1 = 1;     // fp4_e2m1
constexpr uint64_t TPL_DST_TYPE_2 = 2;     // fp4_e1m2
constexpr uint64_t TPL_ROUND_MODE_ROUND = 0;
constexpr uint64_t TPL_ROUND_MODE_FLOOR = 1;
constexpr uint64_t TPL_ROUND_MODE_RINT = 4;

constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t ABS_FOR_UINT16 = 0x7fff;
constexpr uint32_t MAN_FOR_FP32 = 0x007fffff;
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint32_t NUMBER_HALF = 0x00400000;
constexpr float DIGIT_ZERO_FLOAT = 0.0;
constexpr float DIGIT_SIX_FLOAT = 6.0;
constexpr float DIGIT_SEVEN_FLOAT = 7.0;
constexpr float DIGIT_ONE_POINT_EIGHT_SEVEN_FIVE_FLOAT = 1.875;
constexpr uint16_t EXP_MASK_FP16 = 0x7c00;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint16_t EXP_MASK_BF16 = 0x7f80;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_BF16_MAX_EXP = 0x0000;
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN3 = 0x000f;
constexpr uint32_t FP32_MX_MAX_EXP = 0x7f800000;
constexpr int32_t FP32_BIAS_VALUE = 127;
constexpr int32_t FP32_BIAS_NEG_VALUE = -127;
constexpr int32_t FP32_NEG_ONE = -1;
constexpr int32_t FP32_NEG_ZERO_BITS = 0x80000000;
constexpr int16_t FP32_SHR_NUM_VAL = 23;
constexpr float FP4_SCALE_FACTOR = 4.0f;
constexpr float FP4_INV_SCALE_FACTOR = 0.25f;

static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                                RoundMode::UNKNOWN};
static constexpr Reg::CastTrait castTrait32to8 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_RINT};

template <typename T, typename U>
__aicore__ inline void ComputeMxScaleOCP(const int64_t dataLen, const uint16_t loop, __ubuf__ T* xAddr,
                                         Reg::RegTensor<uint8_t>& scaleReg, Reg::RegTensor<uint16_t>& reversedScaleReg)
{
    static constexpr Reg::CastTrait castTraitHalf2Bf16OCP = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    // ===== OCP 算法 =====
    Reg::RegTensor<T> xReg;
    Reg::RegTensor<bfloat16_t> xBf16Reg;
    Reg::RegTensor<uint16_t> expReg;
    Reg::RegTensor<uint16_t> expMaxReg;
    Reg::RegTensor<uint16_t> mxScaleReg;

    Reg::RegTensor<uint16_t> infReg;
    Reg::RegTensor<uint16_t> dstTypeExpMaxReg;
    Reg::RegTensor<uint16_t> fp8NanReg;
    Reg::RegTensor<uint16_t> zeroReg;
    Reg::RegTensor<uint16_t> biasReg;
    Reg::RegTensor<uint16_t> nanReg;
    Reg::RegTensor<uint16_t> specialExpReg;

    Reg::MaskReg invalidDataMask;
    Reg::MaskReg infMask;
    Reg::MaskReg specialDataMask;
    Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        Reg::Duplicate(dstTypeExpMaxReg, FP8_E4M3_MAX_EXP);
    } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        Reg::Duplicate(dstTypeExpMaxReg, FP8_E5M2_MAX_EXP);
    } else if constexpr (IsSameType<U, fp4x2_e2m1_t>::value) {
        Reg::Duplicate(dstTypeExpMaxReg, FP4_E2M1_MAX_EXP);
    } else {
        Reg::Duplicate(dstTypeExpMaxReg, FP4_E1M2_MAX_EXP);
    }
    Reg::Duplicate(infReg, MAX_EXP_FOR_BF16);
    Reg::Duplicate(fp8NanReg, MAX_EXP_FOR_FP8);
    Reg::Duplicate(zeroReg, 0);
    Reg::Duplicate(biasReg, BF16_EXP_BIAS);
    Reg::Duplicate(nanReg, NAN_CUSTOMIZATION);
    Reg::Duplicate(specialExpReg, SPECIAL_EXP_THRESHOLD);

    Reg::Duplicate(expMaxReg, 0);
    for (uint16_t loopIdx1 = 0; loopIdx1 < loop; loopIdx1++) {
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(xReg, xAddr, dataLen);
        if constexpr (IsSameType<T, half>::value) {
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16OCP>(xBf16Reg, xReg, maskAll);
            Reg::And(expReg, (Reg::RegTensor<uint16_t>&)xBf16Reg, infReg, maskAll);
        } else {
            Reg::And(expReg, (Reg::RegTensor<uint16_t>&)xReg, infReg, maskAll);
        }
        Reg::Max(expMaxReg, expMaxReg, expReg, maskAll);
    }
    Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxReg, infReg, maskAll);
    Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxReg, dstTypeExpMaxReg, maskAll);
    Reg::Sub(expMaxReg, expMaxReg, dstTypeExpMaxReg, maskAll);
    Reg::ShiftRights(mxScaleReg, expMaxReg, SHR_NUM_FOR_BF16, maskAll);
    Reg::Select<uint16_t>(mxScaleReg, mxScaleReg, fp8NanReg, infMask);
    Reg::Select<uint16_t>(mxScaleReg, zeroReg, mxScaleReg, invalidDataMask);
    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scaleReg, mxScaleReg);
    Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMaxReg, biasReg, maskAll);
    Reg::Sub(reversedScaleReg, biasReg, expMaxReg, maskAll);
    Reg::Select<uint16_t>(reversedScaleReg, reversedScaleReg, nanReg, infMask);
    Reg::Select<uint16_t>(reversedScaleReg, zeroReg, reversedScaleReg, invalidDataMask);
    Reg::Select<uint16_t>(reversedScaleReg, specialExpReg, reversedScaleReg, specialDataMask);
}

template <typename T, typename U, const uint64_t scaleAlg>
__aicore__ inline void ComputeMxScaleCuBLAS(const int64_t dataLen, const uint16_t loop, __ubuf__ T* xAddr,
                                            Reg::RegTensor<uint8_t>& scaleReg,
                                            Reg::RegTensor<uint16_t>& reversedScaleReg, const float invDstTypeMax_)
{
    // ===== cuBLAS 算法 =====
    Reg::RegTensor<T> xReg;
    Reg::RegTensor<T> xAbsReg;
    Reg::RegTensor<T> xMaxReg;
    Reg::RegTensor<float> xMax0Reg;
    Reg::RegTensor<float> xMax1Reg;
    Reg::RegTensor<uint32_t> expMax0Reg;
    Reg::RegTensor<uint32_t> expMax1Reg;
    Reg::RegTensor<uint32_t> manMax0Reg;
    Reg::RegTensor<uint32_t> manMax1Reg;
    Reg::RegTensor<uint16_t> expMax0Reg16;
    Reg::RegTensor<uint16_t> expMax1Reg16;
    Reg::RegTensor<uint16_t> expMaxReg16;

    Reg::RegTensor<uint16_t> absMaskReg;
    Reg::RegTensor<float> dstTypeMaxReg;
    Reg::RegTensor<uint32_t> manMaskReg;
    Reg::RegTensor<uint16_t> infReg;
    Reg::RegTensor<uint16_t> biasReg;
    Reg::RegTensor<uint16_t> nanReg;
    Reg::RegTensor<uint16_t> specialExpReg;

    Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg p0;
    Reg::MaskReg p1;
    Reg::MaskReg p2;
    Reg::MaskReg p3;

    Reg::Duplicate(dstTypeMaxReg, invDstTypeMax_);
    Reg::Duplicate(absMaskReg, ABS_FOR_UINT16);
    Reg::Duplicate(manMaskReg, MAN_FOR_FP32);
    Reg::Duplicate(infReg, MAX_EXP_FOR_BF16);
    Reg::Duplicate(biasReg, BF16_EXP_BIAS);
    Reg::Duplicate(nanReg, NAN_CUSTOMIZATION);
    Reg::Duplicate(specialExpReg, SPECIAL_EXP_THRESHOLD);

    Reg::Duplicate(xMaxReg, 0);
    for (int i = 0; i < loop; i++) {
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(xReg, xAddr, dataLen);
        Reg::And((Reg::RegTensor<uint16_t>&)xAbsReg, (Reg::RegTensor<uint16_t>&)xReg, absMaskReg, maskAll);
        Reg::Max(xMaxReg, xMaxReg, xAbsReg, maskAll);
    }

    Reg::Cast<float, T, castTraitZero>(xMax0Reg, xMaxReg, maskAll);
    Reg::Cast<float, T, castTraitOne>(xMax1Reg, xMaxReg, maskAll);
    Reg::Mul(xMax0Reg, xMax0Reg, dstTypeMaxReg, maskAll);
    Reg::Mul(xMax1Reg, xMax1Reg, dstTypeMaxReg, maskAll);

    Reg::ShiftRights(expMax0Reg, (Reg::RegTensor<uint32_t>&)xMax0Reg, SHR_NUM_FOR_FP32, maskAll);
    Reg::And(manMax0Reg, (Reg::RegTensor<uint32_t>&)xMax0Reg, manMaskReg, maskAll);
    Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, expMax0Reg, NUMBER_ZERO, maskAll);
    Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, expMax0Reg, NUMBER_TWO_FIVE_FOUR, p0);
    Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, manMax0Reg, NUMBER_ZERO, p0);
    if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
        Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expMax0Reg, NUMBER_ZERO, maskAll);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, manMax0Reg, NUMBER_HALF, p1);
        Reg::MaskXor(p0, p0, p1, maskAll);
    }
    Reg::Adds(manMax0Reg, expMax0Reg, 1, maskAll);
    Reg::Select(expMax0Reg, manMax0Reg, expMax0Reg, p0);
    Reg::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expMax0Reg16, expMax0Reg);

    Reg::ShiftRights(expMax1Reg, (Reg::RegTensor<uint32_t>&)xMax1Reg, SHR_NUM_FOR_FP32, maskAll);
    Reg::And(manMax1Reg, (Reg::RegTensor<uint32_t>&)xMax1Reg, manMaskReg, maskAll);
    Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, expMax1Reg, NUMBER_ZERO, maskAll);
    Reg::CompareScalar<uint32_t, CMPMODE::LT>(p2, expMax1Reg, NUMBER_TWO_FIVE_FOUR, p2);
    Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, manMax1Reg, NUMBER_ZERO, p2);
    if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
        Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p3, expMax1Reg, NUMBER_ZERO, maskAll);
        Reg::CompareScalar<uint32_t, CMPMODE::GT>(p3, manMax1Reg, NUMBER_HALF, p3);
        Reg::MaskXor(p2, p2, p3, maskAll);
    }
    Reg::Adds(manMax1Reg, expMax1Reg, 1, maskAll);
    Reg::Select(expMax1Reg, manMax1Reg, expMax1Reg, p2);
    Reg::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expMax1Reg16, expMax1Reg);

    Reg::Interleave(expMax0Reg16, expMax1Reg16, expMax0Reg16, expMax1Reg16);
    Reg::ShiftLefts(expMaxReg16, expMax0Reg16, SHR_NUM_FOR_BF16, maskAll);
    Reg::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(scaleReg, expMax0Reg16);

    AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(p0, expMaxReg16, infReg, maskAll);
    AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(p1, expMaxReg16, biasReg, maskAll);
    AscendC::MicroAPI::Sub(reversedScaleReg, biasReg, expMaxReg16, maskAll);
    AscendC::MicroAPI::Select<uint16_t>(reversedScaleReg, reversedScaleReg, nanReg, p0);
    AscendC::MicroAPI::Select<uint16_t>(reversedScaleReg, specialExpReg, reversedScaleReg, p1);
}

template <typename T, typename U, const uint64_t dstTypeMax>
__aicore__ inline void ComputeMxScaleDynamicDtypeRange(const int64_t dataLen, const uint16_t loop, __ubuf__ T* xAddr,
                                                       Reg::RegTensor<uint8_t>& scaleReg,
                                                       Reg::RegTensor<uint16_t>& reversedScaleReg)
{
    static constexpr Reg::CastTrait castTraitHalf2Bf16CuBLAS = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                                Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    // ===== DynamicDtypeRange 算法 =====
    Reg::RegTensor<T> xReg;
    Reg::RegTensor<uint16_t> expRegFP16;
    Reg::RegTensor<bfloat16_t> xRegBF16;
    Reg::RegTensor<uint16_t> xAbsReg;
    Reg::RegTensor<uint16_t> xMaxReg;
    Reg::RegTensor<uint16_t> expMaxReg;
    Reg::RegTensor<uint16_t> expMaxRegAdd;
    Reg::RegTensor<uint16_t> sharedExp;
    Reg::RegTensor<uint16_t> scaleValue;
    Reg::RegTensor<uint16_t> nanReg;
    Reg::RegTensor<uint16_t> specialExpReg;

    Reg::RegTensor<uint16_t> expMaskFP16;
    Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
    Reg::RegTensor<uint16_t> absMask16Bit;
    Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);
    Reg::RegTensor<uint16_t> expMaskBF16;
    Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
    Reg::Duplicate(xMaxReg, 0);
    Reg::RegTensor<uint16_t> zeroReg;
    Reg::Duplicate(zeroReg, 0);
    Reg::RegTensor<uint16_t> maxExpValue;
    if constexpr (IsSameType<U, fp4x2_e2m1_t>::value) {
        Reg::Duplicate(maxExpValue, FP4_E2M1_BF16_MAX_EXP);
    } else if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        Reg::Duplicate(maxExpValue, FP4_E1M2_BF16_MAX_EXP);
    }
    Reg::RegTensor<uint16_t> addValue;
    if constexpr (dstTypeMax == TPL_DST_TYPE_MAX_1) {
        Reg::Duplicate(addValue, ADD_VALUE_FOR_BF16_MAN1);
    } else if constexpr (dstTypeMax == TPL_DST_TYPE_MAX_2) {
        Reg::Duplicate(addValue, ADD_VALUE_FOR_BF16_MAN2);
    } else if constexpr (dstTypeMax == TPL_DST_TYPE_MAX_3) {
        Reg::Duplicate(addValue, ADD_VALUE_FOR_BF16_MAN3);
    }
    Reg::RegTensor<uint16_t> scaleBias;
    Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
    Reg::Duplicate(nanReg, NAN_CUSTOMIZATION);
    Reg::Duplicate(specialExpReg, SPECIAL_EXP_THRESHOLD);

    Reg::MaskReg mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg invalidDataMask;
    Reg::MaskReg infMask;
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialDataMask;

    for (int i = 0; i < loop; i++) {
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(xReg, xAddr, dataLen);
        if constexpr (IsSameType<T, half>::value) {
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16CuBLAS>(xRegBF16, xReg, mask);
            Reg::And(xAbsReg, (Reg::RegTensor<uint16_t>&)xRegBF16, absMask16Bit, mask);
        } else {
            Reg::And(xAbsReg, (Reg::RegTensor<uint16_t>&)xReg, absMask16Bit, mask);
        }
        Reg::Max(xMaxReg, xMaxReg, xAbsReg, mask);
    }

    Reg::And(expMaxReg, xMaxReg, expMaskBF16, mask);
    Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxReg, expMaskBF16, mask);
    Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxReg, zeroReg, mask);
    Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expMaxReg, maxExpValue, mask);

    Reg::Add(expMaxRegAdd, xMaxReg, addValue, mask);         // 进位后的结果
    Reg::And(expMaxRegAdd, expMaxRegAdd, expMaskBF16, mask); // 进位后的指数位
    Reg::Select<uint16_t>(expMaxRegAdd, maxExpValue, expMaxRegAdd, invalidDataMask);
    Reg::Sub(sharedExp, expMaxRegAdd, maxExpValue, mask);
    Reg::Select<uint16_t>(scaleValue, sharedExp, expMaskBF16, infMask);
    Reg::Select<uint16_t>(scaleValue, scaleValue, zeroReg, zeroMask);
    Reg::ShiftRights(scaleValue, scaleValue, SHR_NUM_FOR_BF16, mask);
    Reg::Pack<uint8_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(scaleReg, scaleValue);

    Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, mask);
    Reg::Sub(reversedScaleReg, scaleBias, sharedExp, mask);
    Reg::Select<uint16_t>(reversedScaleReg, reversedScaleReg, nanReg, infMask);
    Reg::Select<uint16_t>(reversedScaleReg, reversedScaleReg, zeroReg, zeroMask);
    Reg::Select<uint16_t>(reversedScaleReg, specialExpReg, reversedScaleReg, specialDataMask);
}

template <typename U, RoundMode roundMode>
__aicore__ inline void ComputeFP4FromFp32(Reg::RegTensor<float>& Reg)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZero;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZero, FP32_NEG_ZERO_BITS);
    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        Reg::Muls(Reg, Reg, FP4_SCALE_FACTOR, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, FP4_INV_SCALE_FACTOR, pregAll32);
    } else {
        Reg::Duplicate(maxExpFP32, FP32_MX_MAX_EXP);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, FP32_SHR_NUM_VAL, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG_VALUE, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, FP32_NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS_VALUE, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, FP32_SHR_NUM_VAL, pregAll32);

        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_VALUE, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, FP32_SHR_NUM_VAL, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
__aicore__ inline void ComputeData(int64_t inDataLen, int64_t outDataLen, uint16_t loop, __ubuf__ T* xAddr,
                                   __ubuf__ uint8_t* yAddr, Reg::RegTensor<uint16_t> reversedScaleReg)
{
    Reg::RegTensor<T> xReg;
    Reg::RegTensor<float> reversedScale0Reg;
    Reg::RegTensor<float> reversedScale1Reg;
    Reg::RegTensor<float> x0Reg;
    Reg::RegTensor<float> x1Reg;
    Reg::RegTensor<uint8_t> y0Reg;
    Reg::RegTensor<uint8_t> y1Reg;

    Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

    Reg::Cast<float, bfloat16_t, castTraitZero>(reversedScale0Reg, (Reg::RegTensor<bfloat16_t>&)reversedScaleReg,
                                                maskAll);
    Reg::Cast<float, bfloat16_t, castTraitOne>(reversedScale1Reg, (Reg::RegTensor<bfloat16_t>&)reversedScaleReg,
                                               maskAll);
    if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
        static constexpr Reg::CastTrait castTrait0Fp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTrait1Fp32toBF16 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                                Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitBf16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, roundMode};
        Reg::RegTensor<bfloat16_t> yBf16_0Reg;
        Reg::RegTensor<bfloat16_t> yBf16_1Reg;
        for (uint16_t loopIdx2 = 0; loopIdx2 < loop; loopIdx2++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(xReg, xAddr, inDataLen);

            Reg::Cast<float, T, castTraitZero>(x0Reg, xReg, maskAll);
            Reg::Cast<float, T, castTraitOne>(x1Reg, xReg, maskAll);
            Reg::Mul(x0Reg, x0Reg, reversedScale0Reg, maskAll);
            Reg::Mul(x1Reg, x1Reg, reversedScale1Reg, maskAll);

            ComputeFP4FromFp32<U, roundMode>(x0Reg);
            ComputeFP4FromFp32<U, roundMode>(x1Reg);

            Reg::Cast<bfloat16_t, float, castTrait0Fp32toBF16>(yBf16_0Reg, x0Reg, maskAll);
            Reg::Cast<bfloat16_t, float, castTrait1Fp32toBF16>(yBf16_1Reg, x1Reg, maskAll);
            Reg::Add((Reg::RegTensor<uint16_t>&)yBf16_0Reg, (Reg::RegTensor<uint16_t>&)yBf16_0Reg,
                     (Reg::RegTensor<uint16_t>&)yBf16_1Reg, maskAll);

            Reg::Cast<U, bfloat16_t, castTraitBf16toFp4>((Reg::RegTensor<U>&)y0Reg, yBf16_0Reg, maskB16);

            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(yAddr + outDataLen * loopIdx2, y0Reg, maskAll);
        }
    } else {
        for (uint16_t loopIdx2 = 0; loopIdx2 < loop; loopIdx2++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(xReg, xAddr, inDataLen);

            Reg::Cast<float, T, castTraitZero>(x0Reg, xReg, maskAll);
            Reg::Cast<float, T, castTraitOne>(x1Reg, xReg, maskAll);
            Reg::Mul(x0Reg, x0Reg, reversedScale0Reg, maskAll);
            Reg::Mul(x1Reg, x1Reg, reversedScale1Reg, maskAll);

            Reg::Cast<U, float, castTrait32to8>((Reg::RegTensor<U>&)y0Reg, x0Reg, maskAll);
            Reg::Cast<U, float, castTrait32to8>((Reg::RegTensor<U>&)y1Reg, x1Reg, maskAll);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)y0Reg,
                                                                    (Reg::RegTensor<uint32_t>&)y0Reg);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)y1Reg,
                                                                    (Reg::RegTensor<uint32_t>&)y1Reg);
            Reg::Interleave((Reg::RegTensor<uint16_t>&)y0Reg, (Reg::RegTensor<uint16_t>&)y1Reg,
                            (Reg::RegTensor<uint16_t>&)y0Reg, (Reg::RegTensor<uint16_t>&)y1Reg);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(y0Reg, (Reg::RegTensor<uint16_t>&)y0Reg);

            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, y0Reg, outDataLen, maskAll);
        }
    }
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax>
__aicore__ inline void ComputeMxScale(const int64_t dataLen, const uint16_t loop, __ubuf__ T* xAddr,
                                      Reg::RegTensor<uint8_t>& scaleReg, Reg::RegTensor<uint16_t>& reversedScaleReg,
                                      const float invDstTypeMax_)
{
    if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
        ComputeMxScaleOCP<T, U>(dataLen, loop, xAddr, scaleReg, reversedScaleReg);
    } else if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
        ComputeMxScaleCuBLAS<T, U, TPL_SCALE_ALG_1>(dataLen, loop, xAddr, scaleReg, reversedScaleReg, invDstTypeMax_);
    } else if constexpr (scaleAlg == TPL_SCALE_ALG_2) {
        if constexpr (dstTypeMax == TPL_DST_TYPE_MAX_0) {
            ComputeMxScaleCuBLAS<T, U, TPL_SCALE_ALG_2>(dataLen, loop, xAddr, scaleReg, reversedScaleReg,
                                                        invDstTypeMax_);
        } else {
            ComputeMxScaleDynamicDtypeRange<T, U, dstTypeMax>(dataLen, loop, xAddr, scaleReg, reversedScaleReg);
        }
    }
}

} // namespace MxQuantCommon

#endif // MX_QUANT_COMMON_H
