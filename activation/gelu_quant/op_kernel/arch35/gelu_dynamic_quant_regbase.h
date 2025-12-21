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
 * \file gelu_dynamic_quant_regbase.h
 * \brief
 */

#ifndef GELU_DYNAMIC_QUANT_REGBASE_H
#define GELU_DYNAMIC_QUANT_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gelu_quant_base_regbase.h"

namespace GeluQuantALL {
using namespace AscendC;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;

// dynamic quant 基础模板  ub内单个尾轴或者多个尾轴
template <typename T1, typename T2>
class GeluDynamicQuant : public GeluQuantBase
{
public:
    __aicore__ inline GeluDynamicQuant(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale, GM_ADDR workspace,
        const GeluQuantTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMulRows(LocalTensor<float>& scaleLocalFp32, int64_t offset, int32_t rowCount);
    __aicore__ inline void ProcessOptionalScale(LocalTensor<float>& scaleLocalFp32, int32_t calCount);
    __aicore__ inline void CopyIn(int64_t offset, int32_t rowCount);
    __aicore__ inline void Compute(LocalTensor<float>& scaleLocalFp32, int32_t rowCount);
    __aicore__ inline void CopyOut(int64_t offset, int32_t rowCount);
    __aicore__ inline void ComputeGelu(LocalTensor<float>& geluRes, int32_t rowCount);

    template <typename dstType, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeDynamicQuantRegbase(
        LocalTensor<float>& geluRes, LocalTensor<float>& scaleLocalFp32, int32_t rowCount);
    __aicore__ inline void SetMaxValue()
    {
        if (dstType_ == DT_INT8) {
            maxValue_ = static_cast<float>(1.0) / INT8_MAX_VALUE;
        } else if (dstType_ == DT_FLOAT8_E5M2) {
            maxValue_ = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
        } else if (dstType_ == DT_FLOAT8_E4M3FN) {
            maxValue_ = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
        } else if (dstType_ == DT_HIFLOAT8) {
            maxValue_ = static_cast<float>(1.0) / HIFLOAT8_MAX_VALUE;
        }
    };

private:
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> scaleOutQueue_;

    TBuf<TPosition::VECCALC> scaleQueue_; // smooth scale buffer
    TBuf<TPosition::VECCALC> geluQueue_;  // gelu result
    /* global memory address */
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> inputScaleGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> outScaleGm_;

    /* variable */
    uint32_t mulRows_;
    uint32_t endAxisLenAlignTo32_; // endAxis len (byte) aligned to
    uint32_t endAxisLenAlignTo16_;
    uint32_t endAxisLenAlignTo8_;
    uint32_t endAxisActualAlignLen_;

    float maxValue_ = 0.0;
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToF16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ODD};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF16ToI8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToF8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToH8Round = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToH8Hybrid = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_HYBRID};
};

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::Init(
    GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale, GM_ADDR workspace,
    const GeluQuantTilingData& tilingData)
{
    // Init tiling data
    GeluQuantBase::ParseTilingData(tilingData);
    endAxisLenAlignTo32_ = (endAxisLen_ + FP32_BLOCK_NUM - 1) / FP32_BLOCK_NUM * FP32_BLOCK_NUM;
    endAxisLenAlignTo16_ = (endAxisLen_ + FP16_BLOCK_NUM - 1) / FP16_BLOCK_NUM * FP16_BLOCK_NUM;
    endAxisLenAlignTo8_ = (endAxisLen_ + INT8_BLOCK_NUM - 1) / INT8_BLOCK_NUM * INT8_BLOCK_NUM;
    endAxisActualAlignLen_ = IsSameType<T1, float>::value ? endAxisLenAlignTo32_ : endAxisLenAlignTo16_;

    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // shield global memory address between different core
    uint64_t intraCoreOffset1 = blockIdx_ * normalCoreProcessNum_ * endAxisLen_;
    uint64_t intraCoreOffset2 = blockIdx_ * normalCoreProcessNum_;

    // shield normal core and tail core
    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreProcessNum_ = tailCoreProcessNum_;
    } else {
        curCoreProcessNum_ = normalCoreProcessNum_;
    }

    xGm_.SetGlobalBuffer((__gm__ T1*)x + intraCoreOffset1, curCoreProcessNum_ * endAxisLen_);
    inputScaleGm_.SetGlobalBuffer((__gm__ T2*)inputScale, endAxisLen_);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y + intraCoreOffset1, curCoreProcessNum_ * endAxisLen_);
    outScaleGm_.SetGlobalBuffer((__gm__ float*)outScale + intraCoreOffset2, curCoreProcessNum_ * FP32_BLOCK_NUM);

    pipe.InitBuffer(inQueue_, DOUBLE_BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(geluQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(outQueue_, DOUBLE_BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleOutQueue_, DOUBLE_BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));

    SetMaxValue();
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::ProcessMulRows(
    LocalTensor<float>& scaleLocalFp32, int64_t offset, int32_t rowCount)
{
    CopyIn(offset, rowCount);
    Compute(scaleLocalFp32, rowCount);
    CopyOut(offset, rowCount);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    LocalTensor<float> scaleLocalFp32 = scaleQueue_.Get<float>();
    ProcessOptionalScale(scaleLocalFp32, endAxisActualAlignLen_);

    // coexistenNodeElementNum_: 每次最多进行coexistenNodeElementNum_个元素的计算
    mulRows_ = coexistentNodeElementNum_ / endAxisActualAlignLen_;
    int64_t ubLoopNum = curCoreProcessNum_ / mulRows_;
    int64_t ubLoopTail = curCoreProcessNum_ % mulRows_;
    for (int64_t loopIndex = 0; loopIndex < ubLoopNum; loopIndex++) {
        ProcessMulRows(scaleLocalFp32, mulRows_ * endAxisActualAlignLen_ * loopIndex, mulRows_);
    }
    if (ubLoopTail != 0) {
        ProcessMulRows(scaleLocalFp32, mulRows_ * endAxisActualAlignLen_ * ubLoopNum, ubLoopTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::ProcessOptionalScale(
    LocalTensor<float>& scaleLocalFp32, int32_t calCount)
{
    if (inputScaleType_ == NORMAL_TENSOR) {
        LocalTensor<T2> optionalScale = inQueue_.AllocTensor<T2>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(endAxisLen_ * sizeof(T2)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

        DataCopyPadExtParams<T2> padParams{
            true, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};

        DataCopyPad(optionalScale, inputScaleGm_[0], copyParams, padParams);

        inQueue_.EnQue(optionalScale);
        LocalTensor<T2> scaleInput = inQueue_.DeQue<T2>();

        if constexpr (IsSameType<T2, float>::value) {
            Muls(scaleLocalFp32, scaleInput, 1.0f, calCount);
        } else {
            Cast(scaleLocalFp32, scaleInput, RoundMode::CAST_NONE, calCount);
        }
        inQueue_.FreeTensor(optionalScale);
    } else if (inputScaleType_ == SCALAR_TENSOR) {
        inputScaleScalar_ = GetFloatScalar(inputScaleGm_);
        AscendC::Duplicate<float>(scaleLocalFp32, inputScaleScalar_, calCount);
    } else {
        AscendC::Duplicate<float>(scaleLocalFp32, (float)1.0, calCount);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::CopyIn(int64_t offset, int32_t rowCount)
{
    LocalTensor<T1> xLocal = inQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(rowCount), static_cast<uint32_t>(endAxisLen_ * sizeof(T1)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T1> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T1>(0.0)};
    DataCopyPad(xLocal, xGm_[offset / endAxisActualAlignLen_ * endAxisLen_], copyParams, padParams);
    inQueue_.EnQue(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::Compute(LocalTensor<float>& scaleLocalFp32, int32_t rowCount)
{
    LocalTensor<float> geluRes = geluQueue_.Get<float>();
    ComputeGelu(geluRes, rowCount);
    if (dstType_ == DT_INT8) {
        ComputeDynamicQuantRegbase<int8_t, AscendC::RoundMode::CAST_RINT>(geluRes, scaleLocalFp32, rowCount);
    } else if (dstType_ == DT_FLOAT8_E4M3FN) {
        ComputeDynamicQuantRegbase<fp8_e4m3fn_t, AscendC::RoundMode::CAST_RINT>(geluRes, scaleLocalFp32, rowCount);
    } else if (dstType_ == DT_FLOAT8_E5M2) {
        ComputeDynamicQuantRegbase<fp8_e5m2_t, AscendC::RoundMode::CAST_RINT>(geluRes, scaleLocalFp32, rowCount);
    } else if (roundMode_ == AscendC::RoundMode::CAST_ROUND) { // hifloat8 roundMode hi8_round
        ComputeDynamicQuantRegbase<hifloat8_t, AscendC::RoundMode::CAST_ROUND>(geluRes, scaleLocalFp32, rowCount);
    } else { // hifloat8 roundMode hi8_hybrid
        ComputeDynamicQuantRegbase<hifloat8_t, AscendC::RoundMode::CAST_HYBRID>(geluRes, scaleLocalFp32, rowCount);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::CopyOut(int64_t offset, int32_t rowCount)
{
    LocalTensor<float> scaleOutLocal = scaleOutQueue_.DeQue<float>();
    DataCopyExtParams copyParamsFloat{
        static_cast<uint16_t>(rowCount), static_cast<uint32_t>(1 * sizeof(float)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outScaleGm_[offset / endAxisActualAlignLen_], scaleOutLocal, copyParamsFloat);
    scaleOutQueue_.FreeTensor(scaleOutLocal);

    LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParamsInt8{
        static_cast<uint16_t>(rowCount), static_cast<uint32_t>(endAxisLen_ * sizeof(int8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    auto curOffset = offset / endAxisActualAlignLen_ * endAxisLen_;
    DataCopyPad(yGm_[curOffset], outLocal, copyParamsInt8);
    outQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuant<T1, T2>::ComputeGelu(LocalTensor<float>& geluRes, int32_t rowCount)
{
    LocalTensor<T1> xLocal = inQueue_.DeQue<T1>();
    if (approximate_ == APPROXIMATE_NONE) {
        ComputeGeluErf<T1>(xLocal, geluRes, endAxisActualAlignLen_ * rowCount);
    } else {
        ComputeGeluTanh<T1>(xLocal, geluRes, endAxisActualAlignLen_ * rowCount);
    }
    inQueue_.FreeTensor(xLocal);
}

template <typename T1, typename T2>
template <typename dstType, AscendC::RoundMode roundMode>
__aicore__ inline void GeluDynamicQuant<T1, T2>::ComputeDynamicQuantRegbase(
    LocalTensor<float>& geluRes, LocalTensor<float>& scaleLocalFp32, int32_t rowCount)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
    uint16_t rowCountLocal = static_cast<uint16_t>(rowCount);
    uint16_t loopNum = (endAxisActualAlignLen_ + vl - 1) / vl;
    __ubuf__ float* xAddr = (__ubuf__ float*)geluRes.GetPhyAddr();
    __ubuf__ float* smoothScaleAddr = (__ubuf__ float*)scaleLocalFp32.GetPhyAddr();

    LocalTensor<float> scaleOutLocalFp32 = scaleOutQueue_.AllocTensor<float>();
    LocalTensor<dstType> yLocal = outQueue_.AllocTensor<dstType>();
    __ubuf__ float* scaleOutAddr = (__ubuf__ float*)scaleOutLocalFp32.GetPhyAddr();
    __ubuf__ dstType* yAddr = (__ubuf__ dstType*)yLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregInput;
        AscendC::MicroAPI::RegTensor<float> vregSmoothScale;
        AscendC::MicroAPI::RegTensor<float> vregAbs;
        AscendC::MicroAPI::RegTensor<float> vregReduceMax;
        AscendC::MicroAPI::RegTensor<float> vregOutScale;
        AscendC::MicroAPI::RegTensor<float> vregQuantRes;
        AscendC::MicroAPI::RegTensor<float> vregMax;
        AscendC::MicroAPI::RegTensor<half> vregHalf;
        AscendC::MicroAPI::RegTensor<dstType> vregY;
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg2;

        for (uint16_t i = 0; i < rowCountLocal; i++) {
            AscendC::MicroAPI::Duplicate(vregMax, 0.0);
            uint32_t sreg0 = endAxisActualAlignLen_;
            for (uint16_t j = 0; j < loopNum; j++) {
                preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                AscendC::MicroAPI::DataCopy(vregInput, xAddr + i * endAxisActualAlignLen_ + j * vl);
                // compute smoothscale
                AscendC::MicroAPI::DataCopy(vregSmoothScale, smoothScaleAddr + j * vl);
                AscendC::MicroAPI::Mul(vregInput, vregInput, vregSmoothScale, preg0);
                AscendC::MicroAPI::Abs(vregAbs, vregInput, preg0);
                AscendC::MicroAPI::Max(vregMax, vregAbs, vregMax, preg1);
            }
            {
                AscendC::MicroAPI::ReduceMax(vregReduceMax, vregMax, preg1);
                AscendC::MicroAPI::Muls(vregReduceMax, vregReduceMax, maxValue_, preg1);
                AscendC::MicroAPI::Duplicate(vregOutScale, vregReduceMax, preg1);
                AscendC::MicroAPI::DataCopy(scaleOutAddr + i * FP32_BLOCK_NUM, vregOutScale, preg1);
            }
            uint32_t sreg1 = endAxisLen_;
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < loopNum; j++) {
                auto yOutAddr = yAddr + i * endAxisLenAlignTo8_ + j * vl;
                preg2 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                AscendC::MicroAPI::DataCopy(vregInput, xAddr + i * endAxisActualAlignLen_ + j * vl);
                AscendC::MicroAPI::DataCopy(vregSmoothScale, smoothScaleAddr + j * vl);
                AscendC::MicroAPI::Mul(vregInput, vregInput, vregSmoothScale, preg2);
                AscendC::MicroAPI::Div(vregQuantRes, vregInput, vregOutScale, preg2);

                if constexpr (IsSameType<dstType, int8_t>::value) {
                    AscendC::MicroAPI::Cast<half, float, castTraitF32ToF16>(vregHalf, vregQuantRes, preg2);
                    AscendC::MicroAPI::Cast<dstType, half, castTraitF16ToI8>(vregY, vregHalf, preg2);
                } else if constexpr (
                    IsSameType<dstType, fp8_e4m3fn_t>::value || IsSameType<dstType, fp8_e5m2_t>::value) {
                    AscendC::MicroAPI::Cast<dstType, float, castTraitF32ToF8>(vregY, vregQuantRes, preg2);
                } else if constexpr (
                    IsSameType<dstType, hifloat8_t>::value && roundMode == AscendC::RoundMode::CAST_HYBRID) {
                    AscendC::MicroAPI::Cast<dstType, float, castTraitF32ToH8Hybrid>(vregY, vregQuantRes, preg2);
                } else if constexpr (
                    IsSameType<dstType, hifloat8_t>::value && roundMode == AscendC::RoundMode::CAST_ROUND) {
                    AscendC::MicroAPI::Cast<dstType, float, castTraitF32ToH8Round>(vregY, vregQuantRes, preg2);
                }
                AscendC::MicroAPI::DataCopy<dstType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    yOutAddr, vregY, preg2);
            }
        }
    }
    outQueue_.EnQue<dstType>(yLocal);
    scaleOutQueue_.EnQue<float>(scaleOutLocalFp32);
}
} // namespace GeluQuantALL
#endif