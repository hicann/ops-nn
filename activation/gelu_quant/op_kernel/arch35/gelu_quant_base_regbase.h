/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_quant_base_regbase.h
 * \brief
 */

#ifndef GELU_QUANT_BASE_REGBASE_H
#define GELU_QUANT_BASE_REGBASE_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "../inc/platform.h"
#include "atvoss/util/vec.h"

namespace GeluQuantALL {
using namespace AscendC;
using namespace Ops::Base;

constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121f * 0.044715f;
constexpr float TANH_APPROX_FACTOR = 1.0f / 0.044715f;

constexpr float ERF_PARAM1 = -0.3512339572e-8f;
constexpr float ERF_PARAM2 = 0.2645266170e-6f;
constexpr float ERF_PARAM3 = -0.7929488134e-5f;
constexpr float ERF_PARAM4 = 0.1106123840e-3f;
constexpr float ERF_PARAM5 = 0.6518995814e-4f;
constexpr float ERF_PARAM6 = -0.7266616915e-1f;
constexpr float ERF_PARAM7 = -0.1595769883e1f;
constexpr float ERF_MIN = 5.75f;
constexpr float ERF_MAX = -13.15f;
constexpr float MAX_INT8 = 127.0f;
constexpr float ONE_OVER_SQRT_TWO = 0.707106781f;

constexpr uint32_t APPROXIMATE_NONE = 0;
constexpr uint32_t APPROXIMATE_TANH = 1;

constexpr uint32_t BUFFER_NUM = 2;

constexpr uint32_t EMPTY_TENSOR = 0;
constexpr uint32_t SCALAR_TENSOR = 1;
constexpr uint32_t NORMAL_TENSOR = 2;

constexpr uint32_t FP32_BLOCK_NUM = 8;
constexpr uint32_t FP16_BLOCK_NUM = 16;
constexpr uint32_t INT8_BLOCK_NUM = 32;
constexpr uint32_t FP32_VECTOR_CAPACITY_ONE_CYCLE = 64;
constexpr uint32_t MAX_ROWS_NUM = 1024;
constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL = 60;
// 控制饱和模式的使能位，配合Cast的SatMode使用
// SatMode::SAT下为单指令饱和模式
// SatMode::NO_SAT下为单指令非饱和模式
constexpr int64_t FLOAT_OVERFLOW_MODE_SATURATE = 0;

#ifdef __CCE_AICORE__
constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
#endif
class GeluQuantBase {
public:
    __aicore__ inline GeluQuantBase(){};

    __aicore__ inline void ParseTilingData(const GeluQuantTilingData& tilingData);
    __aicore__ inline void GeluV2ErfPost(
        LocalTensor<float>& dst, LocalTensor<float>& src1, LocalTensor<float>& src2, uint32_t count);
    template <typename T>
    __aicore__ inline void ComputeGeluTanh(const LocalTensor<T>& src, const LocalTensor<float>& dst, uint32_t calCount);
    template <typename T>
    __aicore__ float GetFloatScalar(GlobalTensor<T> inputScaleGm_);
    template <typename dstType, AscendC::RoundMode roundMode>
    __aicore__ inline void CastOutLocal(LocalTensor<float>& src, LocalTensor<int8_t>& dst, int32_t calCount);
    template <typename dstType>
    __aicore__ inline void SetFloatOverflowModeForRegbase()
    {
#if (__NPU_ARCH__ == 3510)
        if constexpr (
            IsSameType<dstType, hifloat8_t>::value || IsSameType<dstType, fp8_e5m2_t>::value ||
            IsSameType<dstType, fp8_e4m3fn_t>::value) {
            AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(FLOAT_OVERFLOW_MODE_SATURATE);
        }
#endif
    }
    // 变量区
    uint32_t usedCoreNum_;
    int64_t curCoreProcessNum_;
    int64_t normalCoreProcessNum_;
    int64_t tailCoreProcessNum_;
    int64_t endAxisLen_;

    uint32_t coexistentNodeNum_;
    uint32_t coexistentNodeElementNum_;

    uint32_t approximate_;
    uint32_t inputScaleType_;
    uint32_t inputOffsetType_;

    // optional
    float inputScaleScalar_;
    float inputOffsetScalar_;

    uint32_t blockIdx_;
    uint32_t dstType_;
    AscendC::RoundMode roundMode_;

    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToF16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ODD};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF16ToI8Rint = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF16ToI8Round = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToF8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToH8Round = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToH8Hybrid = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_HYBRID};
};

__aicore__ inline void GeluQuantBase::ParseTilingData(const GeluQuantTilingData& tilingData)
{
    endAxisLen_ = tilingData.endAxisLen;
    usedCoreNum_ = tilingData.usedCoreNum;
    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    coexistentNodeNum_ = tilingData.coexistentNodeNum;
    coexistentNodeElementNum_ = tilingData.coexistentNodeElementNum;
    approximate_ = tilingData.approximate;
    inputScaleType_ = tilingData.inputScaleType;
    inputOffsetType_ = tilingData.inputOffsetType;
    dstType_ = tilingData.dstType;
    roundMode_ = static_cast<AscendC::RoundMode>(tilingData.roundMode);
}

__aicore__ inline void GeluQuantBase::GeluV2ErfPost(
    LocalTensor<float>& dst, LocalTensor<float>& src1, LocalTensor<float>& src2, uint32_t count)
{
#ifdef __CCE_AICORE__
    uint32_t dtypeSize = sizeof(float);
    uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
    uint16_t loopNum = CeilDivision(count, vl);
    uint32_t vlSize = vl;
    __ubuf__ float* src1Addr = (__ubuf__ float*)src1.GetPhyAddr();
    __ubuf__ float* src2Addr = (__ubuf__ float*)src2.GetPhyAddr();
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();

    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInput1;
    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInput2;
    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputAdds;
    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputMuls;
    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOutput;
    MicroAPI::MaskReg mask;
    __VEC_SCOPE__
    {
        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
            // OpCopyIn
            MicroAPI::DataCopy(vregInput1, (__ubuf__ float*)(src1Addr + loopIdx * vlSize));
            MicroAPI::DataCopy(vregInput2, (__ubuf__ float*)(src2Addr + loopIdx * vlSize));
            MicroAPI::Adds(vregInputAdds, vregInput2, (float)1.0, mask);
            MicroAPI::Muls(vregInputMuls, vregInput1, (float)0.5, mask);
            MicroAPI::Mul(vregOutput, vregInputAdds, vregInputMuls, mask);
            // OpCopyOut
            MicroAPI::DataCopy((__ubuf__ float*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
        }
    }
#endif
}

template <typename T>
__aicore__ inline void GeluQuantBase::ComputeGeluTanh(
    const LocalTensor<T>& src, const LocalTensor<float>& dst, uint32_t count)
{
#ifdef __CCE_AICORE__
    uint32_t dtypeSize = sizeof(float);
    uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
    uint16_t loopNum = (count + vl - 1) / vl;
    uint32_t vlSize = vl;
    __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();

    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInput;
    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputSqr;
    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputCub;
    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOutput;
    MicroAPI::MaskReg mask;
    if constexpr (std::is_same_v<T, float>) {
        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                // OpCopyIn
                MicroAPI::DataCopy(vregInput, (__ubuf__ float*)(srcAddr + loopIdx * vlSize));
                MicroAPI::Mul(vregInputSqr, vregInput, vregInput, mask);
                MicroAPI::Mul(vregInputCub, vregInputSqr, vregInput, mask);
                MicroAPI::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
                MicroAPI::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                MicroAPI::Exp(vregInputCub, vregInputCub, mask);
                MicroAPI::Adds(vregInputCub, vregInputCub, 1.0f, mask);
                MicroAPI::Div(vregOutput, vregInput, vregInputCub, mask);

                // OpCopyOut
                MicroAPI::DataCopy((__ubuf__ float*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
    } else {
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput16;
        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                // OpCopyIn
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregInput16, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                MicroAPI::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);
                MicroAPI::Mul(vregInputSqr, vregInput, vregInput, mask);
                MicroAPI::Mul(vregInputCub, vregInputSqr, vregInput, mask);
                MicroAPI::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
                MicroAPI::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                MicroAPI::Exp(vregInputCub, vregInputCub, mask);
                MicroAPI::Adds(vregInputCub, vregInputCub, 1.0f, mask);
                MicroAPI::Div(vregOutput, vregInput, vregInputCub, mask);

                // OpCopyOut
                MicroAPI::DataCopy((__ubuf__ float*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
    }
#endif
}

template <typename T>
__aicore__ float GeluQuantBase::GetFloatScalar(GlobalTensor<T> inputScaleGm_)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        T scaleValue = inputScaleGm_.GetValue(0);
        uint16_t test1 = *reinterpret_cast<uint16_t*>(&scaleValue);
        uint32_t f32_bits = static_cast<uint32_t>(test1) << 16;
        return *reinterpret_cast<float*>(&f32_bits);
    }
    return (float)inputScaleGm_.GetValue(0);
}

template <typename dstType, AscendC::RoundMode roundMode>
__aicore__ inline void GeluQuantBase::CastOutLocal(LocalTensor<float>& src, LocalTensor<int8_t>& dst, int32_t calCount)
{
    this->SetFloatOverflowModeForRegbase<dstType>();
    uint32_t dtypeSize = sizeof(float);
    uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
    uint16_t loopNum = Ops::Base::CeilDiv(static_cast<uint32_t>(calCount), vl);
    __ubuf__ float* xAddr = (__ubuf__ float*)src.GetPhyAddr();
    __ubuf__ dstType* yAddr = (__ubuf__ dstType*)dst.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregInput;
        AscendC::MicroAPI::RegTensor<half> vregHalf;
        AscendC::MicroAPI::RegTensor<dstType> vregY;
        AscendC::MicroAPI::MaskReg preg0;

        uint32_t sreg1 = calCount;
        for (uint16_t i = 0; i < loopNum; i++) {
            auto yOutAddr = yAddr + i * vl;
            preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
            AscendC::MicroAPI::DataCopy(vregInput, xAddr + i * vl);

            if constexpr (IsSameType<dstType, int8_t>::value) {
                AscendC::MicroAPI::Cast<half, float, castTraitF32ToF16>(vregHalf, vregInput, preg0);
                if constexpr (roundMode == AscendC::RoundMode::CAST_ROUND) {
                    AscendC::MicroAPI::Cast<dstType, half, castTraitF16ToI8Round>(vregY, vregHalf, preg0);
                } else if constexpr (roundMode == AscendC::RoundMode::CAST_RINT) {
                    AscendC::MicroAPI::Cast<dstType, half, castTraitF16ToI8Rint>(vregY, vregHalf, preg0);
                }
            } else if constexpr (IsSameType<dstType, fp8_e4m3fn_t>::value || IsSameType<dstType, fp8_e5m2_t>::value) {
                AscendC::MicroAPI::Cast<dstType, float, castTraitF32ToF8>(vregY, vregInput, preg0);
            } else if constexpr (
                IsSameType<dstType, hifloat8_t>::value && roundMode == AscendC::RoundMode::CAST_HYBRID) {
                AscendC::MicroAPI::Cast<dstType, float, castTraitF32ToH8Hybrid>(vregY, vregInput, preg0);
            } else if constexpr (
                IsSameType<dstType, hifloat8_t>::value && roundMode == AscendC::RoundMode::CAST_ROUND) {
                AscendC::MicroAPI::Cast<dstType, float, castTraitF32ToH8Round>(vregY, vregInput, preg0);
            }
            AscendC::MicroAPI::DataCopy<dstType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(yOutAddr, vregY, preg0);
        }
    }
}
} // namespace GeluQuantALL

#endif // GELU_QUANT_BASE_REGBASE_H
