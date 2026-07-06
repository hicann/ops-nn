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
 * \file swiglu_group_quant_base.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_BASE_H
#define SWIGLU_GROUP_QUANT_BASE_H

#include "kernel_operator.h"

namespace SwigluGroupQuant {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t PER_BLOCK_FP16 = 128;
constexpr int32_t PER_MX_FP16 = 32;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float TOPK_WEIGHT_DEFAULT = 1.0f;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64LL;
constexpr uint16_t FP16_EMASK_AND_INF_VAL = 0x7c00;
constexpr uint16_t BF16_EMASK_AND_INF_VAL = 0x7f80;
constexpr uint16_t BF16_NAN_VAL = 0x7f81;
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E5M2 = 0x0780;
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 = 0x0400;
constexpr uint16_t FP8_E8M0_NAN_VAL = 0x00ff;
constexpr uint16_t FP8_E8M0_SPECIAL_MIN = 0x0040;
constexpr int16_t BF16_EXP_SHR_BITS = 7;
constexpr uint16_t BF16_EXP_INVSUB = 0x7f00;
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint16_t FOUR_UNFOLD = 4;
constexpr int64_t MX_BLOCK_SIZE_MXFP4 = 32LL;
constexpr int64_t OUT_ELE_NUM_ONE_BLK_MAXFP4 = 64LL;
constexpr uint16_t FP16_EMASK_AND_INF_VAL_MAXFP4 = 0x7c00;
constexpr uint16_t BF16_EMASK_AND_INF_VAL_MAXFP4 = 0x7f80;
constexpr uint16_t BF16_NAN_VAL_MAXFP4 = 0x7f81;
constexpr int16_t BF16_EXP_SHR_BITS_MAXFP4 = 7;
constexpr uint16_t BF16_EXP_INVSUB_MAXFP4 = 0x7f00;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t FP4_PACK_NUM = 2;
constexpr int32_t TBUF_POOL_NUM = 12;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t NEW_MANTISSA = 0x0008;

#define FLOAT_OVERFLOW_MODE_CTRL 60
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float POS_INFINITY = INFINITY;
constexpr float NEG_INFINITY = -INFINITY;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int b) { return CeilDiv(a, b) * b; }

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

constexpr AscendC::Reg::CastTrait castTraitB162B32Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitB322B16Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait castTraitF32toFp8Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait castTraitU32toU8Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr AscendC::Reg::CastTrait traitB16ToB32Layout0 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait traitB16ToB32Layout1 = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T>
__simd_callee__ inline void LoadInputData(RegTensor<float>& dst, __ubuf__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__simd_callee__ inline void StoreOutputData(__ubuf__ T* dst, RegTensor<float>& src, MaskReg pregLoop,
                                            uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toFp8Even>(tmp, src, pregLoop);
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__simd_callee__ inline void StoreOuputDataUnalign(RegTensor<float>& src, __ubuf__ T*& dst, UnalignRegForStore& uDst,
                                                  MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreUnAlign(dst, src, uDst, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpPack;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        Pack((RegTensor<uint16_t>&)tmpPack, (RegTensor<uint32_t>&)tmp);
        StoreUnAlign(dst, tmpPack, uDst, postUpdateStride);
    }
}

template <typename T>
__simd_callee__ inline void StoreMxFp8Scale(__ubuf__ T* dst, RegTensor<int32_t>& src, MaskReg pregLoop,
                                            uint32_t dstOffset)
{
    RegTensor<uint8_t> tmp1;
    Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp1, src, pregLoop);
    StoreAlign<T, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B8>(dst + dstOffset, (RegTensor<T>&)tmp1, pregLoop);
}

__simd_callee__ inline void VFSwiGlu(RegTensor<float>& y, RegTensor<float>& x0, RegTensor<float>& x1,
                                     RegTensor<float>& one, RegTensor<float>& vreg, MaskReg pregLoop)
{
    Muls(vreg, x0, static_cast<float>(-1.0f), pregLoop);
    Exp(vreg, vreg, pregLoop);
    Adds(vreg, vreg, static_cast<float>(1.0f), pregLoop);
    Div(vreg, x0, vreg, pregLoop);
    Mul(y, vreg, x1, pregLoop);
}

template <typename T, typename U>
__simd_callee__ inline void FP16Convert(AscendC::Reg::RegTensor<half>& output, AscendC::Reg::RegTensor<half>& input,
                                        AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<uint16_t> specialValueTensor;
    AscendC::Reg::RegTensor<uint16_t> newMantissa;
    AscendC::Reg::RegTensor<uint16_t> andResult;
    AscendC::Reg::RegTensor<uint16_t> newValue;
    AscendC::Reg::MaskReg specialMask;
    AscendC::Reg::MaskReg nonzeroMask;
    uint16_t specialValue = SPECIAL_VALUE_E1M2;
    if constexpr (IsSameType<U, fp4x2_e2m1_t>::value) {
        specialValue = SPECIAL_VALUE_E2M1;
    }
    AscendC::Reg::Duplicate(specialValueTensor, specialValue);
    AscendC::Reg::Duplicate(newMantissa, NEW_MANTISSA);
    AscendC::Reg::And(andResult, (AscendC::Reg::RegTensor<uint16_t>&)input, specialValueTensor, mask);
    AscendC::Reg::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
    AscendC::Reg::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
    AscendC::Reg::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
    AscendC::Reg::Or(newValue, (AscendC::Reg::RegTensor<uint16_t>&)input, newMantissa, mask);
    AscendC::Reg::Select<uint16_t>((AscendC::Reg::RegTensor<uint16_t>&)output, newValue,
                                   (AscendC::Reg::RegTensor<uint16_t>&)input, specialMask);
}

template <typename T, bool hasTopkWeight = false, bool hasClampValue = false>
__simd_vf__ inline void VFProcessSwigluVf(__ubuf__ T* yLocalAddr, __ubuf__ T* x0LocalAddr, __ubuf__ T* x1LocalAddr,
                                          __ubuf__ float* topkWeightLocalAddr, uint16_t loopCount, uint32_t sregNum,
                                          uint32_t curColNumAlign, const uint16_t curRowNum, float clampValue)
{
    RegTensor<float> weight;
    RegTensor<float> x0;
    RegTensor<float> x1;
    RegTensor<float> y;
    RegTensor<float> one;
    RegTensor<float> tmp;
    MaskReg pregLoop = CreateMask<float>();
    Duplicate(one, static_cast<float>(1.0f), pregLoop);
    for (uint16_t i = 0; i < curRowNum; i++) {
        if constexpr (hasTopkWeight) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, topkWeightLocalAddr + i);
        }
        uint32_t sreg = sregNum;
        for (uint16_t j = 0; j < loopCount; j++) {
            pregLoop = UpdateMask<float>(sreg);
            LoadInputData<T>(x0, x0LocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
            LoadInputData<T>(x1, x1LocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
            if constexpr (hasClampValue) {
                Mins(x0, x0, clampValue, pregLoop);
                Maxs(x1, x1, -clampValue, pregLoop);
                Mins(x1, x1, clampValue, pregLoop);
            }
            VFSwiGlu(y, x0, x1, one, tmp, pregLoop);
            if constexpr (hasTopkWeight) {
                Mul(y, y, weight, pregLoop);
            }
            StoreOutputData<T>(yLocalAddr, y, pregLoop, j * VL_FP32 + i * curColNumAlign);
        }
    }
}

template <typename T, bool hasTopkWeight = false, bool hasClampValue = false>
__aicore__ inline void VFProcessSwiglu(const LocalTensor<T>& yLocal, const LocalTensor<T>& x0Local,
                                       const LocalTensor<T>& x1Local, const LocalTensor<float>& topkWeightLocal,
                                       const uint16_t curRowNum, const uint32_t curColNum, float clampValue)
{
    __ubuf__ T* yLocalAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* x0LocalAddr = (__ubuf__ T*)x0Local.GetPhyAddr();
    __ubuf__ T* x1LocalAddr = (__ubuf__ T*)x1Local.GetPhyAddr();
    __ubuf__ float* topkWeightLocalAddr = hasTopkWeight ? (__ubuf__ float*)topkWeightLocal.GetPhyAddr() : nullptr;
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t sregNum = curColNum;
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    AscendC::VF_CALL<VFProcessSwigluVf<T, hasTopkWeight, hasClampValue>>(yLocalAddr, x0LocalAddr, x1LocalAddr,
                                                                         topkWeightLocalAddr, loopCount, sregNum,
                                                                         curColNumAlign, curRowNum, clampValue);
}

template <typename T>
__simd_vf__ inline void VFComputeMaxExpMXFP4Vf(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t scaleNum,
                                               uint32_t onceXNum, uint32_t totalCountInUB, uint16_t loopNum)
{
    AscendC::Reg::RegTensor<T> vdExp0;
    AscendC::Reg::RegTensor<T> vdExp1;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp1BF16;
    AscendC::Reg::RegTensor<uint16_t> vdExpExtract0;
    AscendC::Reg::RegTensor<uint16_t> vdExpExtract1;
    AscendC::Reg::RegTensor<uint16_t> vdExpSelect0;
    AscendC::Reg::RegTensor<uint16_t> vdExpSelect1;
    AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
    AscendC::Reg::Duplicate(expMaskBF16, BF16_EMASK_AND_INF_VAL_MAXFP4);
    AscendC::Reg::RegTensor<uint16_t> invalidmaskfp16;
    AscendC::Reg::Duplicate(invalidmaskfp16, FP16_EMASK_AND_INF_VAL_MAXFP4);
    AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
    AscendC::Reg::MaskReg scaleMask1;
    AscendC::Reg::MaskReg invalidDataMask0;
    AscendC::Reg::MaskReg invalidDataMask1;
    AscendC::Reg::UnalignRegForStore u1;
    static constexpr AscendC::Reg::CastTrait castTraitHalf2Bf16 = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_TRUNC};

    for (uint16_t i = 0; i < loopNum; i++) {
        scaleMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceXNum);

        if constexpr (IsSameType<T, half>::value) {
            AscendC::Reg::And(vdExpSelect0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, invalidmaskfp16, scaleMask1);
            AscendC::Reg::And(vdExpSelect1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, invalidmaskfp16, scaleMask1);
            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidmaskfp16, scaleMask1);
            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidmaskfp16, scaleMask1);
            AscendC::Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
            AscendC::Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
            AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, scaleMask1);
            AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, scaleMask1);
            AscendC::Reg::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
            AscendC::Reg::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
        } else {
            AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
            AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
        }

        AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
        AscendC::Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

        AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, u1,
                                                                                          scaleNum);
    }
    AscendC::Reg::StoreUnAlignPost(maxExpAddr, u1, 0);
}

template <typename T>
__aicore__ inline void VFComputeMaxExpMXFP4(const LocalTensor<T>& srcLocal, const LocalTensor<uint16_t>& maxExpLocal,
                                            uint16_t curRowNum, uint32_t curColNum)
{
    __ubuf__ T* srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();
    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)maxExpLocal.GetPhyAddr();
    uint32_t vlForB16 = REPEAT_SIZE / sizeof(T);
    uint32_t scaleNum = REPEAT_SIZE / BLOCK_SIZE;
    uint32_t onceXNum = vlForB16 * DIGIT_TWO;
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    uint32_t totalCountInUB = static_cast<uint32_t>(curRowNum) * curColNumAlign;
    uint16_t loopNum = CeilDiv(totalCountInUB, onceXNum);
    AscendC::VF_CALL<VFComputeMaxExpMXFP4Vf<T>>(srcAddr, maxExpAddr, scaleNum, onceXNum, totalCountInUB, loopNum);
}

__simd_vf__ inline void VFComputeScaleMXFP4Vf(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                              __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                              uint16_t loopNumScale, uint16_t f4Emax)
{
    constexpr uint16_t onceNum = REPEAT_SIZE / sizeof(uint16_t); // 128 scales per loop
    constexpr uint16_t onceNumMxScale = 64;                      // e8m0 store stride (uint16 words)
    AscendC::Reg::RegTensor<uint16_t> expMask;
    AscendC::Reg::Duplicate(expMask, BF16_EMASK_AND_INF_VAL_MAXFP4);
    AscendC::Reg::RegTensor<uint16_t> vdMaxExp;

    AscendC::Reg::MaskReg cmpResult;
    AscendC::Reg::MaskReg zeroMask;
    AscendC::Reg::MaskReg preMaskScale;
    AscendC::Reg::RegTensor<uint16_t> maxExpValue;
    AscendC::Reg::Duplicate(maxExpValue, f4Emax);
    AscendC::Reg::RegTensor<uint16_t> sharedExp;
    AscendC::Reg::RegTensor<uint16_t> scaleValue;
    AscendC::Reg::RegTensor<uint16_t> scaleBias;
    AscendC::Reg::Duplicate(scaleBias, BF16_EXP_INVSUB_MAXFP4);
    AscendC::Reg::RegTensor<uint16_t> halfScale;
    AscendC::Reg::RegTensor<uint16_t> fp8NanRegTensor;
    AscendC::Reg::Duplicate(fp8NanRegTensor, FP8_E8M0_NAN_VAL);
    AscendC::Reg::RegTensor<uint16_t> zeroRegTensor;
    AscendC::Reg::Duplicate(zeroRegTensor, 0);
    AscendC::Reg::RegTensor<uint16_t> nanRegTensor;
    AscendC::Reg::Duplicate(nanRegTensor, BF16_NAN_VAL_MAXFP4);
    AscendC::Reg::MaskReg invalidDataMask;
    AscendC::Reg::MaskReg specialDataMask;
    AscendC::Reg::RegTensor<uint16_t> specialExpRegTensor;
    AscendC::Reg::Duplicate(specialExpRegTensor, FP8_E8M0_SPECIAL_MIN);
    uint32_t sreg = totalScaleInUB;
    for (uint16_t i = 0; i < loopNumScale; i++) {
        preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(sreg);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr, onceNum);
        AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
        AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
        AscendC::Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

        AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

        AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::Reg::ShiftRights(scaleValue, sharedExp, BF16_EXP_SHR_BITS_MAXFP4, preMaskScale);

        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue, onceNumMxScale,
                                                                         preMaskScale);

        AscendC::Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
        AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr, halfScale,
                                                                                        onceNum, preMaskScale);
    }
}

__aicore__ inline void VFComputeScaleMXFP4(const LocalTensor<uint16_t>& maxExpLocal,
                                           const LocalTensor<uint16_t>& mxScaleLocal,
                                           const LocalTensor<uint16_t>& halfScaleLocal, uint16_t curRowNum,
                                           uint32_t curColNum, uint16_t f4Emax)
{
    __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)mxScaleLocal.GetPhyAddr();
    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)maxExpLocal.GetPhyAddr();
    __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)halfScaleLocal.GetPhyAddr();
    uint32_t scaleColAlign = RoundUp<uint16_t>(CeilDiv(curColNum, MX_BLOCK_SIZE_MXFP4));
    uint32_t totalScaleInUB = curRowNum * scaleColAlign;
    constexpr uint16_t onceNum = REPEAT_SIZE / sizeof(uint16_t); // 128 scales per loop
    uint16_t loopNumScale = CeilDiv(totalScaleInUB, static_cast<uint32_t>(onceNum));
    AscendC::VF_CALL<VFComputeScaleMXFP4Vf>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB,
                                            loopNumScale, f4Emax);
}

template <typename T, typename U>
__simd_vf__ inline void VFComputeDataMXFP4Vf(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                             __ubuf__ int8_t* outLocalAddr, uint32_t numUbBlocksPerVReg,
                                             uint32_t onceXNum, uint32_t totalCountInUB, uint16_t loopNum)
{
    AscendC::Reg::MaskReg dataMask1;
    AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
    AscendC::Reg::RegTensor<T> vdExp0;
    AscendC::Reg::RegTensor<T> vdExp1;

    AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp1BF16;

    AscendC::Reg::RegTensor<U> vdExp0FP4;
    AscendC::Reg::RegTensor<U> vdExp1FP4;

    static constexpr AscendC::Reg::CastTrait castTrait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
    static constexpr AscendC::Reg::CastTrait castTraitHalf2Bf16 = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_TRUNC};
    for (uint16_t i = 0; i < loopNum; i++) {
        dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceXNum);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                      numUbBlocksPerVReg);

        if constexpr (IsSameType<T, half>::value) {
            FP16Convert<T, U>(vdExp0, vdExp0, dataMask1);
            FP16Convert<T, U>(vdExp1, vdExp1, dataMask1);
            AscendC::Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
            AscendC::Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
            AscendC::Reg::Mul(vdExp0BF16, vdExp0BF16, (AscendC::Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Mul(vdExp1BF16, vdExp1BF16, (AscendC::Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
            AscendC::Reg::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
            AscendC::Reg::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
        } else {
            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
        }

        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK_MAXFP4, dataMask1);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK_MAXFP4, dataMask1);
    }
}

template <typename T, typename U>
__aicore__ inline void VFComputeDataMXFP4(const LocalTensor<T>& srcLocal, const LocalTensor<uint16_t>& halfScaleLocal,
                                          const LocalTensor<int8_t>& outLocal, uint16_t curRowNum, uint32_t curColNum)
{
    __ubuf__ T* srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();
    __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)halfScaleLocal.GetPhyAddr();
    __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
    uint32_t vlForB16 = REPEAT_SIZE / sizeof(uint16_t);
    uint32_t numUbBlocksPerVReg = REPEAT_SIZE / BLOCK_SIZE;
    uint32_t onceXNum = vlForB16 * DIGIT_TWO;
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    uint32_t totalCountInUB = static_cast<uint32_t>(curRowNum) * curColNumAlign;
    uint16_t loopNum = CeilDiv(totalCountInUB, onceXNum);
    AscendC::VF_CALL<VFComputeDataMXFP4Vf<T, U>>(srcAddr, halfScaleLocalAddr, outLocalAddr, numUbBlocksPerVReg,
                                                 onceXNum, totalCountInUB, loopNum);
}

// curColNum must be divisible by 128.
template <typename T0, typename T1, typename T2, bool hasRoundScale = false, bool hasClampValue = false,
          bool hasOutput = false, bool hasTopkWeight = false>
__simd_vf__ inline void VFProcessSwigluGroupQuantVf(__ubuf__ T0* yLocalAddr, __ubuf__ T1* yOriginLocalAddr,
                                                    __ubuf__ T2* scaleLocalAddr, __ubuf__ T1* x0LocalAddr,
                                                    __ubuf__ T1* x1LocalAddr, __ubuf__ float* topkWeightLocalAddr,
                                                    float clampValue, const uint16_t curRowNum, uint32_t curColNumAlign,
                                                    uint32_t dstCurColNumAlign, uint32_t scaleColNum,
                                                    uint16_t loopCountFoldTwo, uint32_t maxValueInt, float fp8Max)
{
    RegTensor<float> weight;
    RegTensor<float> one;
    RegTensor<uint32_t> oneUint32;
    RegTensor<float> zero;
    RegTensor<uint32_t> zeroUint32;
    RegTensor<float> xLeft;
    RegTensor<float> xRight;
    RegTensor<float> x0Left;
    RegTensor<float> x0Right;
    RegTensor<float> x1Left;
    RegTensor<float> x1Right;
    RegTensor<float> xAbsLeft;
    RegTensor<float> xAbsRight;
    RegTensor<float> tmp;
    RegTensor<uint32_t> tmp0;
    RegTensor<uint32_t> tmp1;
    RegTensor<uint32_t> tmp2;
    RegTensor<uint32_t> tmp3;
    RegTensor<int32_t> tmp4;
    RegTensor<float> scale;
    RegTensor<float> clampScale;
    RegTensor<float> invScale;
    RegTensor<float> dupInvScale;
    RegTensor<float> scale0;
    RegTensor<float> scale1;
    RegTensor<uint32_t> scale2;
    RegTensor<uint32_t> scale3;
    RegTensor<int32_t> scale4;
    RegTensor<int32_t> scale5;
    RegTensor<uint32_t> coeff;
    RegTensor<float> invCoeff;
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    MaskReg compareLeft;
    MaskReg compareRight;
    MaskReg compareMask0;
    Duplicate(zero, 0.0f, pregMain);
    Duplicate(zeroUint32, static_cast<uint32_t>(0), pregMain);
    Duplicate(one, 1.0f, pregMain);
    Duplicate(oneUint32, static_cast<uint32_t>(1), pregMain);
    Duplicate(coeff, maxValueInt, pregMerge);
    Duplicate(invCoeff, fp8Max, pregMerge);
    Duplicate(tmp0, FAST_LOG_AND_VALUE1, pregMerge);
    Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
    Duplicate(tmp3, static_cast<uint32_t>(127), pregMerge);
    Duplicate(tmp4, static_cast<int32_t>(127), pregMerge);
    for (uint16_t i = 0; i < curRowNum; i++) {
        if constexpr (hasTopkWeight) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, topkWeightLocalAddr + i);
        }
        for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
            LoadInputData<T1>(x0Left, x0LocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
            LoadInputData<T1>(x0Right, x0LocalAddr, pregMain, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
            LoadInputData<T1>(x1Left, x1LocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
            LoadInputData<T1>(x1Right, x1LocalAddr, pregMain, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
            if constexpr (hasClampValue) {
                Mins(x0Left, x0Left, clampValue, pregMain);
                Mins(x0Right, x0Right, clampValue, pregMain);
                Maxs(x1Left, x1Left, -clampValue, pregMain);
                Mins(x1Left, x1Left, clampValue, pregMain);
                Maxs(x1Right, x1Right, -clampValue, pregMain);
                Mins(x1Right, x1Right, clampValue, pregMain);
            }
            VFSwiGlu(xLeft, x0Left, x1Left, one, tmp, pregMain);
            VFSwiGlu(xRight, x0Right, x1Right, one, tmp, pregMain);
            if constexpr (hasTopkWeight) {
                Mul(xLeft, xLeft, weight, pregMain);
            }
            Add(xLeft, xLeft, zero, pregMain);
            if constexpr (hasTopkWeight) {
                Mul(xRight, xRight, weight, pregMain);
            }
            Add(xRight, xRight, zero, pregMain);
            if constexpr (hasOutput) {
                StoreOutputData<T1>(yOriginLocalAddr, xLeft, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                StoreOutputData<T1>(yOriginLocalAddr, xRight, pregMain, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
            }
            Muls(xAbsLeft, xLeft, 0.0f, pregMain);
            Compare<float, CMPMODE::NE>(compareLeft, xAbsLeft, xAbsLeft, pregMain);
            MaskNot(compareLeft, compareLeft, pregMain);
            Abs(xAbsLeft, xLeft, compareLeft);
            ReduceMax(scale0, xAbsLeft, pregMain);

            Muls(xAbsRight, xRight, 0.0f, pregMain);
            Compare<float, CMPMODE::NE>(compareRight, xAbsRight, xAbsRight, pregMain);
            MaskNot(compareRight, compareRight, pregMain);
            Abs(xAbsRight, xRight, compareRight);
            ReduceMax(scale1, xAbsRight, pregMain);
            Max(scale, scale0, scale1, pregMerge);
            Maxs(clampScale, scale, 0.0001f, pregMerge); // amax

            Mul(scale, clampScale, (RegTensor<float>&)coeff, pregMerge);
            if constexpr (!hasRoundScale) {
                Div(invScale, invCoeff, clampScale, pregMerge);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)scaleLocalAddr + j + i * scaleColNum, scale, pregMerge); // copy out scale
            } else {
                // bits == (RegTensor<uint32_t>&)scale
                ShiftRights(scale2, (RegTensor<uint32_t>&)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                And(scale2, scale2, tmp0, pregMerge);                      // exp
                And(scale3, (RegTensor<uint32_t>&)scale, tmp1, pregMerge); // man_bits
                Compare<uint32_t, AscendC::CMPMODE::NE>(compareMask0, scale3, zeroUint32, pregMerge);
                Select(tmp2, oneUint32, zeroUint32, compareMask0); // man_bits != 0
                Sub(scale3, scale2, tmp3, pregMerge);              // exp - 127
                // exp_scale-uint32 = exp - 127 + (man_bits != 0)
                Add((RegTensor<uint32_t>&)scale4, scale3, tmp2, pregMerge);

                Adds(scale5, scale4, 127, pregMerge); // sf_uint32
                ShiftLefts((RegTensor<int32_t>&)scale, scale5, static_cast<int16_t>(23), pregMerge);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)scaleLocalAddr + j + i * scaleColNum, scale, pregMerge); // copy out scale

                Sub(scale5, tmp4, scale4, pregMerge); // 127 - exp_scale
                // ((127 - exp_scale) << 23).view(float32)
                ShiftLefts((RegTensor<int32_t>&)invScale, scale5, static_cast<int16_t>(23), pregMerge);
            }
            Duplicate(dupInvScale, invScale, pregMain);

            Mul(x0Left, xLeft, dupInvScale, pregMain);
            Muls(x1Left, x0Left, 0.0f, pregMain);
            Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregMain);
            Select(xLeft, xLeft, x0Left, compareLeft);
            Mul(x0Right, xRight, dupInvScale, pregMain);
            Muls(x1Right, x0Right, 0.0f, pregMain);
            Compare<float, CMPMODE::NE>(compareRight, x1Right, x1Right, pregMain);
            Select(xRight, xRight, x0Right, compareRight);
            StoreOutputData<T0>(yLocalAddr, xLeft, pregMain, 2 * j * VL_FP32 + i * dstCurColNumAlign);
            StoreOutputData<T0>(yLocalAddr, xRight, pregMain, (2 * j + 1) * VL_FP32 + i * dstCurColNumAlign);
        }
    }
}

template <typename T0, typename T1, typename T2, bool hasRoundScale = false, bool hasClampValue = false,
          bool hasOutput = false, bool hasTopkWeight = false>
__aicore__ inline void VFProcessSwigluGroupQuant(const LocalTensor<T0>& yLocal, const LocalTensor<T1>& yOriginLocal,
                                                 const LocalTensor<T2>& scaleLocal, const LocalTensor<T1>& x0Local,
                                                 const LocalTensor<T1>& x1Local,
                                                 const LocalTensor<float>& topkWeightLocal, float clampValue,
                                                 const uint16_t curRowNum, const uint32_t curColNum)
{
    __ubuf__ T0* yLocalAddr = (__ubuf__ T0*)yLocal.GetPhyAddr();
    __ubuf__ T1* yOriginLocalAddr = hasOutput ? (__ubuf__ T1*)yOriginLocal.GetPhyAddr() : nullptr;
    __ubuf__ T2* scaleLocalAddr = (__ubuf__ T2*)scaleLocal.GetPhyAddr();
    __ubuf__ T1* x0LocalAddr = (__ubuf__ T1*)x0Local.GetPhyAddr();
    __ubuf__ T1* x1LocalAddr = (__ubuf__ T1*)x1Local.GetPhyAddr();
    __ubuf__ float* topkWeightLocalAddr = hasTopkWeight ? (__ubuf__ float*)topkWeightLocal.GetPhyAddr() : nullptr;

    uint32_t maxValueInt = 0;
    float fp8Max = 0.0f;
    if constexpr (IsSameType<T0, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
        fp8Max = FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T0, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
        fp8Max = FP8_E4M3FN_MAX_VALUE;
    }

    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint32_t scaleColNum = CeilDiv(curColNum, PER_BLOCK_FP16);
    uint16_t loopCountFoldTwo = loopCount / 2;
    AscendC::VF_CALL<VFProcessSwigluGroupQuantVf<T0, T1, T2, hasRoundScale, hasClampValue, hasOutput, hasTopkWeight>>(
        yLocalAddr, yOriginLocalAddr, scaleLocalAddr, x0LocalAddr, x1LocalAddr, topkWeightLocalAddr, clampValue,
        curRowNum, curColNumAlign, dstCurColNumAlign, scaleColNum, loopCountFoldTwo, maxValueInt, fp8Max);
}

template <typename T0, typename T1, typename T2>
__aicore__ inline void SwigluGroupQuantDispatcher(const LocalTensor<T0>& yLocal, const LocalTensor<T1>& yOriginLocal,
                                                  const LocalTensor<T2>& scaleLocal, const LocalTensor<T1>& x0Local,
                                                  const LocalTensor<T1>& x1Local,
                                                  const LocalTensor<float>& topkWeightLocal, float clampValue,
                                                  const uint16_t curRowNum, const uint32_t curColNum, int32_t maskBit)
{
    if (maskBit == 0b000) {
        VFProcessSwigluGroupQuant<T0, T1, T2, false, false, false, false>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b001) {
        VFProcessSwigluGroupQuant<T0, T1, T2, false, false, false, true>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b010) {
        VFProcessSwigluGroupQuant<T0, T1, T2, false, true, false, false>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b011) {
        VFProcessSwigluGroupQuant<T0, T1, T2, false, true, false, true>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b100) {
        VFProcessSwigluGroupQuant<T0, T1, T2, true, false, false, false>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b101) {
        VFProcessSwigluGroupQuant<T0, T1, T2, true, false, false, true>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b110) {
        VFProcessSwigluGroupQuant<T0, T1, T2, true, true, false, false>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b111) {
        VFProcessSwigluGroupQuant<T0, T1, T2, true, true, false, true>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    }
}

template <typename T0, typename T1, typename T2>
__aicore__ inline void SwigluGroupQuantDispatcherYOrigin(
    const LocalTensor<T0>& yLocal, const LocalTensor<T1>& yOriginLocal, const LocalTensor<T2>& scaleLocal,
    const LocalTensor<T1>& x0Local, const LocalTensor<T1>& x1Local, const LocalTensor<float>& topkWeightLocal,
    float clampValue, const uint16_t curRowNum, const uint32_t curColNum, int32_t maskBit)
{
    if (maskBit == 0b000) {
        VFProcessSwigluGroupQuant<T0, T1, T2, false, false, true, false>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b001) {
        VFProcessSwigluGroupQuant<T0, T1, T2, false, false, true, true>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b010) {
        VFProcessSwigluGroupQuant<T0, T1, T2, false, true, true, false>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b011) {
        VFProcessSwigluGroupQuant<T0, T1, T2, false, true, true, true>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b100) {
        VFProcessSwigluGroupQuant<T0, T1, T2, true, false, true, false>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b101) {
        VFProcessSwigluGroupQuant<T0, T1, T2, true, false, true, true>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b110) {
        VFProcessSwigluGroupQuant<T0, T1, T2, true, true, true, false>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b111) {
        VFProcessSwigluGroupQuant<T0, T1, T2, true, true, true, true>(
            yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    }
}

template <typename T, bool withUbReduce = false>
__simd_vf__ inline void VFProcessGroupIndexSmallVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t curColNum,
                                                   uint16_t vlLen, uint16_t loopCount)
{
    RegTensor<T> x;
    RegTensor<T> sum;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum, static_cast<T>(0), pregMain);
    uint32_t sreg = curColNum;
    MaskReg pregLoop;
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x, xLocalAddr + i * vlLen);
        Adds(x, x, static_cast<T>(0), pregLoop);
        Add(sum, sum, x, pregMain);
    }
    ReduceSum(sum, sum, pregMain);
    if (withUbReduce) {
        RegTensor<T> origin;
        LoadAlign(origin, yLocalAddr);
        Add(sum, sum, origin, pregMerge);
    }
    StoreAlign(yLocalAddr, sum, pregMerge);
}

template <typename T, bool withUbReduce = false>
__simd_vf__ inline void VFProcessGroupIndexLargeVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t vlLen,
                                                   uint16_t fourLoopCount, uint16_t tailLoopNum, uint32_t tailReminder)
{
    RegTensor<T> x0;
    RegTensor<T> x1;
    RegTensor<T> x2;
    RegTensor<T> x3;
    RegTensor<T> sum0;
    RegTensor<T> sum1;
    RegTensor<T> sum2;
    RegTensor<T> sum3;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum0, static_cast<T>(0), pregMain);
    Duplicate(sum1, static_cast<T>(0), pregMain);
    Duplicate(sum2, static_cast<T>(0), pregMain);
    Duplicate(sum3, static_cast<T>(0), pregMain);
    MaskReg pregLoop;
    for (uint16_t i = 0; i < fourLoopCount; i++) {
        LoadAlign(x0, xLocalAddr + i * FOUR_UNFOLD * vlLen);
        Add(sum0, sum0, x0, pregMain);
        LoadAlign(x1, xLocalAddr + (i * FOUR_UNFOLD + 1) * vlLen);
        Add(sum1, sum1, x1, pregMain);
        LoadAlign(x2, xLocalAddr + (i * FOUR_UNFOLD + 2) * vlLen);
        Add(sum2, sum2, x2, pregMain);
        LoadAlign(x3, xLocalAddr + (i * FOUR_UNFOLD + 3) * vlLen);
        Add(sum3, sum3, x3, pregMain);
    }
    uint32_t sreg = tailReminder;
    for (uint16_t i = 0; i < tailLoopNum; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x0, xLocalAddr + (fourLoopCount * FOUR_UNFOLD + i) * vlLen);
        Adds(x0, x0, static_cast<T>(0), pregLoop);
        Add(sum0, sum0, x0, pregMain);
    }
    Add(sum0, sum0, sum1, pregMain);
    Add(sum2, sum2, sum3, pregMain);
    Add(sum0, sum0, sum2, pregMain);
    ReduceSum(sum0, sum0, pregMain);
    if (withUbReduce) {
        RegTensor<T> origin;
        LoadAlign(origin, yLocalAddr);
        Add(sum0, sum0, origin, pregMerge);
    }
    StoreAlign(yLocalAddr, sum0, pregMerge);
}

template <typename T, bool withUbReduce = false>
__aicore__ inline void VFProcessGroupIndex(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                           uint16_t curColNum)
{
    __ubuf__ T* yLocalAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    uint16_t vlLen = REPEAT_SIZE / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint16_t fourLoopCount = loopCount / FOUR_UNFOLD;
    uint16_t tailLoopNum = loopCount % FOUR_UNFOLD;
    uint32_t tailReminder = curColNum - fourLoopCount * vlLen * FOUR_UNFOLD;
    if (loopCount < FOUR_UNFOLD) {
        AscendC::VF_CALL<VFProcessGroupIndexSmallVf<T, withUbReduce>>(yLocalAddr, xLocalAddr, curColNum, vlLen,
                                                                      loopCount);
    } else {
        AscendC::VF_CALL<VFProcessGroupIndexLargeVf<T, withUbReduce>>(yLocalAddr, xLocalAddr, vlLen, fourLoopCount,
                                                                      tailLoopNum, tailReminder);
    }
}

template <typename T0, typename T1, bool hasClampLimit = false, bool hasOutput = false, bool hasWeight = false>
__simd_vf__ inline void VFProcessSwigluMxFp8InvScaleVf(__ubuf__ T0* yOriginLocalAddr, __ubuf__ float* yLocalAddr,
                                                       __ubuf__ uint8_t* scaleLocalAddr,
                                                       __ubuf__ float* invScaleLocalAddr, __ubuf__ T0* x0LocalAddr,
                                                       __ubuf__ T0* x1LocalAddr, __ubuf__ float* weightLocalAddr,
                                                       float clampLimit, const uint16_t curRowNum, uint16_t loopCount,
                                                       uint32_t curColNumAlignT, uint32_t curColNumAlignFloat,
                                                       uint32_t invScaleColNumAlign, uint16_t vlLen,
                                                       uint32_t maxValueInt)
{
    RegTensor<float> weight;
    RegTensor<float> one;
    RegTensor<uint32_t> oneUint32;
    RegTensor<float> zero;
    RegTensor<uint32_t> zeroUint32;
    RegTensor<T0> x0;
    RegTensor<T0> x1;
    RegTensor<float> x0Layout0;
    RegTensor<float> x0Layout1;
    RegTensor<float> x1Layout0;
    RegTensor<float> x1Layout1;
    RegTensor<float> yLayout0; // 奇偶交错 偶数部分
    RegTensor<float> yLayout1; // 奇偶交错 奇数部分
    RegTensor<float> y0;       // 还原交错逻辑 高64位
    RegTensor<float> y1;       // 还原交错逻辑 低64位
    RegTensor<float> yMax0;
    RegTensor<float> yMax1;
    RegTensor<float> yMax1Layout0;
    RegTensor<float> yMax1Layout1;
    RegTensor<float> scale;
    RegTensor<float> clampScale;
    RegTensor<float> invScale;
    RegTensor<float> invScale0;
    RegTensor<float> invScale1;
    RegTensor<float> dupInvScale;
    RegTensor<float> scale0;
    RegTensor<float> scale1;
    RegTensor<uint32_t> scale2;
    RegTensor<uint32_t> scale3;
    RegTensor<int32_t> scale4;
    RegTensor<int32_t> scale5;
    RegTensor<uint8_t> scale6;
    RegTensor<uint16_t> scale7;
    RegTensor<uint32_t> coeff;
    RegTensor<float> tmp;
    RegTensor<uint32_t> tmp0;
    RegTensor<uint32_t> tmp1;
    RegTensor<uint32_t> tmp2;
    RegTensor<uint32_t> tmp3;
    RegTensor<int32_t> tmp4;
    RegTensor<uint8_t> tmp5;
    UnalignRegForStore uReg;
    uint32_t sreg = (REPEAT_SIZE / sizeof(T0)) / 32;
    uint32_t sreg1 = 4 * (REPEAT_SIZE / sizeof(T0)) / 32;
    MaskReg pregMain0 = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregMain1 = CreateMask<T0, MaskPattern::ALL>();
    MaskReg pregMerge = UpdateMask<float>(sreg);
    MaskReg pregMerge1 = UpdateMask<float>(sreg1);
    MaskReg compareMask0;
    Duplicate(one, 1.0f, pregMain0);
    Duplicate(zeroUint32, static_cast<uint32_t>(0), pregMain0);
    Duplicate(zero, 0.0f, pregMain0);
    Duplicate(oneUint32, static_cast<uint32_t>(1), pregMain0);
    Duplicate(coeff, maxValueInt, pregMain0);
    Duplicate(tmp0, FAST_LOG_AND_VALUE1, pregMerge);
    Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
    Duplicate(tmp3, static_cast<uint32_t>(127), pregMerge);
    Duplicate(tmp4, static_cast<int32_t>(127), pregMerge);
    for (uint16_t i = 0; i < curRowNum; i++) {
        if constexpr (hasWeight) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, weightLocalAddr + i);
        }
        for (uint16_t j = 0; j < loopCount; j++) {
            LoadAlign(x0, x0LocalAddr + i * curColNumAlignT + j * vlLen);
            LoadAlign(x1, x1LocalAddr + i * curColNumAlignT + j * vlLen);
            Cast<float, T0, traitB16ToB32Layout0>(x0Layout0, x0, pregMain1);
            Cast<float, T0, traitB16ToB32Layout1>(x0Layout1, x0, pregMain1);
            Cast<float, T0, traitB16ToB32Layout0>(x1Layout0, x1, pregMain1);
            Cast<float, T0, traitB16ToB32Layout1>(x1Layout1, x1, pregMain1);

            if constexpr (hasClampLimit) {
                Mins(x0Layout0, x0Layout0, clampLimit, pregMain0);
                Mins(x0Layout1, x0Layout1, clampLimit, pregMain0);
                Maxs(x1Layout0, x1Layout0, -clampLimit, pregMain0);
                Mins(x1Layout0, x1Layout0, clampLimit, pregMain0);
                Maxs(x1Layout1, x1Layout1, -clampLimit, pregMain0);
                Mins(x1Layout1, x1Layout1, clampLimit, pregMain0);
            }
            VFSwiGlu(yLayout0, x0Layout0, x1Layout0, one, tmp, pregMain0);
            VFSwiGlu(yLayout1, x0Layout1, x1Layout1, one, tmp, pregMain0);
            if constexpr (hasWeight) {
                Mul(yLayout0, yLayout0, weight, pregMain0);
                Mul(yLayout1, yLayout1, weight, pregMain0);
            }
            Add(yLayout0, yLayout0, zero, pregMain0);
            Add(yLayout1, yLayout1, zero, pregMain0);
            // 合并奇偶位置
            Interleave(y0, y1, yLayout0, yLayout1);
            if constexpr (hasOutput) {
                StoreOutputData<T0>(yOriginLocalAddr, y0, pregMain0, 2 * j * VL_FP32 + i * curColNumAlignT);
                StoreOutputData<T0>(yOriginLocalAddr, y1, pregMain0, (2 * j + 1) * VL_FP32 + i * curColNumAlignT);
            }
            StoreOutputData<float>(yLocalAddr, y0, pregMain0, 2 * j * VL_FP32 + i * curColNumAlignFloat);
            StoreOutputData<float>(yLocalAddr, y1, pregMain0, (2 * j + 1) * VL_FP32 + i * curColNumAlignFloat);

            Abs(yLayout0, yLayout0, pregMain0);
            Abs(yLayout1, yLayout1, pregMain0);

            // fp32场景，32个数对应4个Block；先做一次Max，接着ReduceMaxWithBlock
            // 然后DeInterLeave并且Max，得到每4个Block的最大值
            Max(yMax0, yLayout0, yLayout1, pregMain0); // 4 --> 2
            ReduceMaxWithDataBlock(yMax1, yMax0, pregMain0);
            DeInterleave(yMax1Layout0, yMax1Layout1, yMax1, yMax1);

            Max(scale, yMax1Layout0, yMax1Layout1, pregMerge);

            Maxs(clampScale, scale, 0.0001f, pregMerge);                 // amax
            Mul(scale, clampScale, (RegTensor<float>&)coeff, pregMerge); // sf = amax / 448.0

            // 量化逻辑
            ShiftRights(scale2, (RegTensor<uint32_t>&)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
            And(scale2, scale2, tmp0, pregMerge);                      // exp
            And(scale3, (RegTensor<uint32_t>&)scale, tmp1, pregMerge); // man_bits
            Compare<uint32_t, AscendC::CMPMODE::NE>(compareMask0, scale3, zeroUint32, pregMerge);
            Select(tmp2, oneUint32, zeroUint32, compareMask0); // man_bits != 0
            Sub(scale3, scale2, tmp3, pregMerge);              // exp - 127
            // exp_scale-uint32 = exp - 127 + (man_bits != 0)
            Add((RegTensor<uint32_t>&)scale4, scale3, tmp2, pregMerge);
            Adds(scale5, scale4, 127, pregMerge); // sf_uint32

            Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp5, scale5, pregMerge);
            Pack(scale7, (RegTensor<uint32_t>&)tmp5);
            Pack(scale6, scale7);
            StoreUnAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(scaleLocalAddr, scale6, uReg, 4);

            Sub(scale5, tmp4, scale4, pregMerge); // 127 - exp_scale
            // ((127 - exp_scale) << 23).view(float32)
            ShiftLefts((RegTensor<int32_t>&)invScale, scale5, static_cast<int16_t>(23), pregMerge);
            Interleave(invScale0, invScale1, invScale, invScale);
            Interleave(invScale1, invScale, invScale0, invScale0);
            StoreOutputData<float>(invScaleLocalAddr, invScale1, pregMerge1, j * 16 + i * invScaleColNumAlign);
        }
        StoreUnAlignPost(scaleLocalAddr, uReg, 0);
    }
}

template <typename T0, typename T1, bool hasClampLimit = false, bool hasOutput = false, bool hasWeight = false>
__aicore__ inline void VFProcessSwigluMxFp8InvScale(
    const LocalTensor<T0>& yOriginLocal, const LocalTensor<float>& yLocal, const LocalTensor<uint8_t>& scaleLocal,
    const LocalTensor<float>& invScaleLocal, const LocalTensor<T0>& x0Local, const LocalTensor<T0>& x1Local,
    const LocalTensor<float>& weightLocal, float clampLimit, const uint16_t curRowNum, const uint32_t curColNum)
{
    __ubuf__ float* yLocalAddr = (__ubuf__ float*)yLocal.GetPhyAddr();
    __ubuf__ T0* yOriginLocalAddr = hasOutput ? (__ubuf__ T0*)yOriginLocal.GetPhyAddr() : nullptr;
    __ubuf__ uint8_t* scaleLocalAddr = (__ubuf__ uint8_t*)scaleLocal.GetPhyAddr();
    __ubuf__ float* invScaleLocalAddr = (__ubuf__ float*)invScaleLocal.GetPhyAddr();
    __ubuf__ T0* x0LocalAddr = (__ubuf__ T0*)x0Local.GetPhyAddr();
    __ubuf__ T0* x1LocalAddr = (__ubuf__ T0*)x1Local.GetPhyAddr();
    __ubuf__ float* weightLocalAddr = hasWeight ? (__ubuf__ float*)weightLocal.GetPhyAddr() : nullptr;
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<T1, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T1, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    }
    uint16_t vlLen = REPEAT_SIZE / sizeof(T0);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint32_t curColNumAlignT = RoundUp<T0>(curColNum);
    uint32_t curColNumAlignFloat = RoundUp<float>(curColNum);
    uint32_t invScaleColNumAlign = RoundUp<float>(curColNumAlignT / 8);
    AscendC::VF_CALL<VFProcessSwigluMxFp8InvScaleVf<T0, T1, hasClampLimit, hasOutput, hasWeight>>(
        yOriginLocalAddr, yLocalAddr, scaleLocalAddr, invScaleLocalAddr, x0LocalAddr, x1LocalAddr, weightLocalAddr,
        clampLimit, curRowNum, loopCount, curColNumAlignT, curColNumAlignFloat, invScaleColNumAlign, vlLen,
        maxValueInt);
}

template <typename T>
__simd_vf__ inline void VFProcessSwigluMxFp8QuantVf(__ubuf__ T* yQuantLocalAddr, __ubuf__ float* yLocalAddr,
                                                    __ubuf__ float* scaleLocalAddr, const uint16_t curRowNum,
                                                    uint16_t loopCount, uint32_t curColNumAlign,
                                                    uint32_t dstCurColNumAlign, uint32_t invScaleColNumAlign)
{
    RegTensor<float> y;
    RegTensor<float> invScale;
    RegTensor<float> dupInvScale;
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t i = 0; i < curRowNum; i++) {
        for (uint16_t j = 0; j < loopCount; j++) {
            LoadInputData<float>(y, yLocalAddr, pregMain, i * curColNumAlign + j * VL_FP32);
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_E2B_B32>(invScale,
                                                                   scaleLocalAddr + j * 8 + i * invScaleColNumAlign);
            Mul(y, y, invScale, pregMain);
            StoreOutputData<T>(yQuantLocalAddr, y, pregMain, j * VL_FP32 + i * dstCurColNumAlign);
        }
    }
}

template <typename T>
__aicore__ inline void VFProcessSwigluMxFp8Quant(const LocalTensor<T>& yQuantLocal, const LocalTensor<float>& yLocal,
                                                 const LocalTensor<float>& invScaleLocal, const uint16_t curRowNum,
                                                 const uint32_t curColNum)
{
    __ubuf__ T* yQuantLocalAddr = (__ubuf__ T*)yQuantLocal.GetPhyAddr();
    __ubuf__ float* yLocalAddr = (__ubuf__ float*)yLocal.GetPhyAddr();
    __ubuf__ float* scaleLocalAddr = (__ubuf__ float*)invScaleLocal.GetPhyAddr();

    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T>(curColNum);
    uint32_t invScaleColNumAlign = RoundUp<float>(CeilDiv(curColNum, 8));
    AscendC::VF_CALL<VFProcessSwigluMxFp8QuantVf<T>>(yQuantLocalAddr, yLocalAddr, scaleLocalAddr, curRowNum, loopCount,
                                                     curColNumAlign, dstCurColNumAlign, invScaleColNumAlign);
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen, uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

__aicore__ inline void SetDefaultBlockTiling(const SwigluGroupQuantTilingData* tilingData, int64_t& usedCoreNums,
                                             int64_t& rowOfFormerBlock, int64_t& rowOfTailBlock,
                                             int64_t& rowLoopOfFormerBlock, int64_t& rowLoopOfTailBlock,
                                             int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    rowOfFormerBlock = tilingData->rowOfFormerBlock;
    rowOfTailBlock = tilingData->rowOfTailBlock;
    rowLoopOfFormerBlock = tilingData->rowLoopOfFormerBlock;
    rowLoopOfTailBlock = tilingData->rowLoopOfTailBlock;
    tailRowFactorOfFormerBlock = tilingData->tailRowFactorOfFormerBlock;
    tailRowFactorOfTailBlock = tilingData->tailRowFactorOfTailBlock;
    usedCoreNums = GetBlockNum();
}

__aicore__ inline void SetGroupIndexBlockTiling(const SwigluGroupQuantTilingData* tilingData, int64_t realBs,
                                                int64_t& usedCoreNums, int64_t& rowOfFormerBlock,
                                                int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock,
                                                int64_t& rowLoopOfTailBlock, int64_t& tailRowFactorOfFormerBlock,
                                                int64_t& tailRowFactorOfTailBlock)
{
    rowOfFormerBlock = CeilDiv(realBs, static_cast<int64_t>(tilingData->coreNum));
    usedCoreNums = CeilDiv(realBs, rowOfFormerBlock) < tilingData->coreNum ? CeilDiv(realBs, rowOfFormerBlock) :
                                                                             tilingData->coreNum;
    rowOfTailBlock = realBs - (usedCoreNums - 1) * rowOfFormerBlock;

    rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, tilingData->rowFactor);
    rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, tilingData->rowFactor);
    tailRowFactorOfFormerBlock = rowOfFormerBlock % tilingData->rowFactor == 0 ?
                                     tilingData->rowFactor :
                                     rowOfFormerBlock % tilingData->rowFactor;
    tailRowFactorOfTailBlock = rowOfTailBlock % tilingData->rowFactor == 0 ? tilingData->rowFactor :
                                                                             rowOfTailBlock % tilingData->rowFactor;
}

template <typename TBufPoolType>
__aicore__ inline void ProcessGroupIndexTiling(GM_ADDR groupIndex, const SwigluGroupQuantTilingData* tilingData,
                                               TBufPoolType& tBufPool, TQue<QuePosition::VECIN, 1>& groupIndexQue,
                                               TBuf<QuePosition::VECCALC>& groupIndexSumBuf,
                                               GlobalTensor<int64_t>& groupIndexGm, LocalTensor<int64_t>& groupSumLocal,
                                               bool& hasGroupIndex, int64_t& usedCoreNums, int64_t& rowOfFormerBlock,
                                               int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock,
                                               int64_t& rowLoopOfTailBlock, int64_t& tailRowFactorOfFormerBlock,
                                               int64_t& tailRowFactorOfTailBlock)
{
    if (groupIndex == nullptr) {
        SetDefaultBlockTiling(tilingData, usedCoreNums, rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock,
                              rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
        return;
    }

    hasGroupIndex = true;
    groupIndexGm.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    tBufPool.InitBuffer(groupIndexQue, DOUBLE_BUFFER_NUM, RoundUp<int64_t>(tilingData->gFactor) * sizeof(int64_t));
    tBufPool.InitBuffer(groupIndexSumBuf, BLOCK_SIZE);
    groupSumLocal = groupIndexSumBuf.Get<int64_t>();
    for (int64_t idx = 0; idx < tilingData->gLoop; idx++) {
        int64_t curGFactor = (idx == tilingData->gLoop - 1) ? tilingData->tailGFactor : tilingData->gFactor;
        LocalTensor<int64_t> groupIndexLocal = groupIndexQue.template AllocTensor<int64_t>();
        CopyIn(groupIndexGm[idx * tilingData->gFactor], groupIndexLocal, 1, curGFactor);
        groupIndexQue.template EnQue(groupIndexLocal);
        groupIndexLocal = groupIndexQue.template DeQue<int64_t>();
        if (idx == 0) {
            VFProcessGroupIndex<int64_t, false>(groupSumLocal, groupIndexLocal, curGFactor);
        } else {
            VFProcessGroupIndex<int64_t, true>(groupSumLocal, groupIndexLocal, curGFactor);
        }
        groupIndexQue.template FreeTensor(groupIndexLocal);
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
    int64_t realBs = groupSumLocal.GetValue(0) > tilingData->bs ? tilingData->bs : groupSumLocal.GetValue(0);
    SetGroupIndexBlockTiling(tilingData, realBs, usedCoreNums, rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock,
                             rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
    tBufPool.Reset();
}

template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}
} // namespace SwigluGroupQuant

#endif
