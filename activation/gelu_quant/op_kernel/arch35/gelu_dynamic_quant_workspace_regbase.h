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
 * \file gelu_dynamic_quant_workspace_regbase.h
 * \brief
 */

#ifndef GELU_DYNAMIC_QUANT_WORKSPACE_REGBASE_H
#define GELU_DYNAMIC_QUANT_WORKSPACE_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gelu_quant_base_regbase.h"

namespace GeluQuantALL {
using namespace AscendC;
// dynamic quant 基础泛化模板  ub内单个大尾轴

constexpr float FP8_E5M2_MAX = 57344.0f;
constexpr float FP8_E4M3FN_MAX = 448.0f;
constexpr float HIFLOAT8_MAX = 32768.0f;
constexpr float MAX_VALUE_INT8 = 127.0f;

template <typename T1, typename T2>
class GeluDynamicQuantWorkspace : public GeluQuantBase
{
public:
    __aicore__ inline GeluDynamicQuantWorkspace(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale, GM_ADDR workspace,
        const GeluQuantTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPerBlock(int64_t endAxisOffset, int32_t calCount, LocalTensor<float>& maxValueRes);
    __aicore__ inline void ProcessPerEndAxis();
    __aicore__ inline void ProcessOptionalScale(
        LocalTensor<float>& scaleLocalFp32, int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void WorkspaceToScaleRes(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void CopyIn(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void Compute(
        LocalTensor<float>& scaleLocalFp32, int64_t endAxisOffset, int32_t calCount, LocalTensor<float>& maxValueRes);
    __aicore__ inline void CopyOut(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void CopyOutWorkspace(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void CopyOutScaleOut(LocalTensor<float>& scaleOut);
    __aicore__ inline void ComputeGelu(LocalTensor<float>& geluRes, int32_t calCount);
    __aicore__ inline void ComputeDynamicQuant(LocalTensor<float>& scaleOut, int32_t calCount);
    __aicore__ inline void ComputeDynamicQuantRegbase(
        LocalTensor<float>& geluRes, LocalTensor<float>& workspaceLocal, LocalTensor<float>& scaleLocalFp32,
        LocalTensor<float>& maxValueRes, int32_t calCount);
    __aicore__ inline void SetMaxValue()
    {
        if (dstType_ == DT_INT8) {
            maxValue_ = MAX_VALUE_INT8;
        } else if (dstType_ == DT_FLOAT8_E5M2) {
            maxValue_ = FP8_E5M2_MAX;
        } else if (dstType_ == DT_FLOAT8_E4M3FN) {
            maxValue_ = FP8_E4M3FN_MAX;
        } else if (dstType_ == DT_HIFLOAT8) {
            maxValue_ = HIFLOAT8_MAX;
        }
    };

private:
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> workspaceQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scaleOutQueue_;

    TBuf<TPosition::VECCALC> castBuf_;
    TBuf<TPosition::VECCALC> geluBuf_;
    TBuf<TPosition::VECCALC> maxValueBuf_;

    /* global memory address */
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> inputScaleGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> outScaleGm_;
    GlobalTensor<float> workspaceGm_;

    /* variable */
    int64_t loopNum_;
    float maxValue_ = 0.0;
};

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::Init(
    GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale, GM_ADDR workspace,
    const GeluQuantTilingData& tilingData)
{
    // Init tiling data
    GeluQuantBase::ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // shield global memory address between different core
    uint64_t intraCoreOffset1 = blockIdx_ * normalCoreProcessNum_ * endAxisLen_;
    uint64_t intraCoreOffset2 = blockIdx_ * normalCoreProcessNum_;
    uint64_t intraCoreOffset3 = blockIdx_ * endAxisLen_;

    // shield normal core and tail core
    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreProcessNum_ = tailCoreProcessNum_;
    } else {
        curCoreProcessNum_ = normalCoreProcessNum_;
    }

    xGm_.SetGlobalBuffer((__gm__ T1*)x + intraCoreOffset1);
    inputScaleGm_.SetGlobalBuffer((__gm__ T2*)inputScale);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y + intraCoreOffset1);
    outScaleGm_.SetGlobalBuffer((__gm__ float*)outScale + intraCoreOffset2);
    workspaceGm_.SetGlobalBuffer((__gm__ float*)workspace + intraCoreOffset3);

    pipe.InitBuffer(inQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(geluBuf_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(castBuf_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(maxValueBuf_, FP32_VECTOR_CAPACITY_ONE_CYCLE * sizeof(float));
    pipe.InitBuffer(workspaceQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleOutQueue_, BUFFER_NUM, FP32_BLOCK_NUM * sizeof(float));

    SetMaxValue();
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ProcessPerBlock(
    int64_t endAxisOffset, int32_t calCount, LocalTensor<float>& maxValueRes)
{
    LocalTensor<float> scaleLocalFp32 = castBuf_.Get<float>();
    CopyIn(endAxisOffset, calCount);
    Compute(scaleLocalFp32, endAxisOffset, calCount, maxValueRes);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ProcessPerEndAxis()
{
    int64_t endAxisLoopNum = endAxisLen_ / coexistentNodeElementNum_;
    int64_t endAxisLoopTail = endAxisLen_ % coexistentNodeElementNum_;

    LocalTensor<float> maxValueRes = maxValueBuf_.Get<float>();
    Duplicate<float>(maxValueRes, 0.0f, INT8_BLOCK_NUM);
    for (int64_t endAxisLoopIndex = 0; endAxisLoopIndex < endAxisLoopNum; endAxisLoopIndex++) {
        ProcessPerBlock(endAxisLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_, maxValueRes);
    }
    if (endAxisLoopTail != 0) {
        ProcessPerBlock(endAxisLoopNum * coexistentNodeElementNum_, endAxisLoopTail, maxValueRes);
    }
    LocalTensor<float> scaleOutLocal = scaleOutQueue_.AllocTensor<float>();
    Divs(scaleOutLocal, maxValueRes, maxValue_, FP32_BLOCK_NUM);
    scaleOutQueue_.EnQue<float>(scaleOutLocal);
    LocalTensor<float> scaleOut = scaleOutQueue_.DeQue<float>();
    int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

    for (int64_t endAxisLoopIndex = 0; endAxisLoopIndex < endAxisLoopNum; endAxisLoopIndex++) {
        WorkspaceToScaleRes(endAxisLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_);
        ComputeDynamicQuant(scaleOut, coexistentNodeElementNum_);
        CopyOut(endAxisLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    if (endAxisLoopTail != 0) {
        WorkspaceToScaleRes(endAxisLoopNum * coexistentNodeElementNum_, endAxisLoopTail);
        ComputeDynamicQuant(scaleOut, endAxisLoopTail);
        CopyOut(endAxisLoopNum * coexistentNodeElementNum_, endAxisLoopTail);
    }
    CopyOutScaleOut(scaleOut);
    scaleOutQueue_.FreeTensor(scaleOut);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    if (inputScaleType_ == SCALAR_TENSOR) {
        inputScaleScalar_ = GetFloatScalar(inputScaleGm_);
    }
    for (loopNum_ = 0; loopNum_ < curCoreProcessNum_; loopNum_++) {
        ProcessPerEndAxis();
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ProcessOptionalScale(
    LocalTensor<float>& scaleLocalFp32, int64_t endAxisOffset, int32_t calCount)
{
    if (inputScaleType_ == NORMAL_TENSOR) {
        LocalTensor<T2> optionalScale = scaleQueue_.AllocTensor<T2>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T2)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

        DataCopyPadExtParams<T2> padParams{
            false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};

        DataCopyPad(optionalScale, inputScaleGm_[endAxisOffset], copyParams, padParams);

        scaleQueue_.EnQue(optionalScale);
        LocalTensor<T2> scaleInput = scaleQueue_.DeQue<T2>();

        if constexpr (IsSameType<T2, float>::value) {
            Muls(scaleLocalFp32, scaleInput, 1.0f, calCount);
        } else {
            Cast(scaleLocalFp32, scaleInput, RoundMode::CAST_NONE, calCount);
        }
        scaleQueue_.FreeTensor(optionalScale);
    } else if (inputScaleType_ == SCALAR_TENSOR) {
        Duplicate<float>(scaleLocalFp32, inputScaleScalar_, calCount);
    } else {
        Duplicate<float>(scaleLocalFp32, 1.0f, calCount);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::CopyIn(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<T1> xLocal = inQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T1)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T1> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};
    DataCopyPad(xLocal, xGm_[endAxisLen_ * loopNum_ + endAxisOffset], copyParams, padParams);
    inQueue_.EnQue(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::Compute(
    LocalTensor<float>& scaleLocalFp32, int64_t endAxisOffset, int32_t calCount, LocalTensor<float>& maxValueRes)
{
    LocalTensor<float> geluRes = geluBuf_.Get<float>();
    LocalTensor<float> workspaceLocal = workspaceQueue_.AllocTensor<float>();
    ComputeGelu(geluRes, calCount);
    ProcessOptionalScale(scaleLocalFp32, endAxisOffset, calCount);
    ComputeDynamicQuantRegbase(geluRes, workspaceLocal, scaleLocalFp32, maxValueRes, calCount);
    workspaceQueue_.EnQue<float>(workspaceLocal);
    CopyOutWorkspace(endAxisOffset, calCount);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::WorkspaceToScaleRes(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<float> scaleResLocal = inQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(float)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<float> padParams{
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};
    DataCopyPad(scaleResLocal, workspaceGm_[endAxisOffset], copyParams, padParams);
    inQueue_.EnQue(scaleResLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::CopyOut(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<int8_t> outLocal = workspaceQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParamsInt8{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(int8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[endAxisLen_ * loopNum_ + endAxisOffset], outLocal, copyParamsInt8);
    workspaceQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::CopyOutWorkspace(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<float> workspaceLocalOut = workspaceQueue_.DeQue<float>();
    DataCopyExtParams copyParamsFloat{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(float)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(workspaceGm_[endAxisOffset], workspaceLocalOut, copyParamsFloat);
    workspaceQueue_.FreeTensor(workspaceLocalOut);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::CopyOutScaleOut(LocalTensor<float>& scaleOut)
{
    DataCopyExtParams copyParamsFloat{
        static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(float)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outScaleGm_[loopNum_], scaleOut, copyParamsFloat);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ComputeGelu(LocalTensor<float>& geluRes, int32_t calCount)
{
    LocalTensor<T1> xLocal = inQueue_.DeQue<T1>();

    if (approximate_ == APPROXIMATE_NONE) {
        ComputeGeluErf<T1>(xLocal, geluRes, calCount);
    } else {
        ComputeGeluTanh<T1>(xLocal, geluRes, calCount);
    }
    inQueue_.FreeTensor(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ComputeDynamicQuant(
    LocalTensor<float>& scaleOut, int32_t calCount)
{
    LocalTensor<float> geluRes = inQueue_.DeQue<float>();
    Divs(geluRes, geluRes, scaleOut[0], calCount);
    LocalTensor<int8_t> outLocal = workspaceQueue_.AllocTensor<int8_t>();

    if (dstType_ == DT_INT8) {
        LocalTensor<half> castFp16 = castBuf_.Get<half>();
        Cast<half, float>(castFp16, geluRes, AscendC::RoundMode::CAST_ODD, calCount);
        Cast<int8_t, half>(outLocal, castFp16, AscendC::RoundMode::CAST_RINT, calCount);
    } else if (dstType_ == DT_FLOAT8_E5M2) {
        Cast<fp8_e5m2_t, float>(
            outLocal.ReinterpretCast<fp8_e5m2_t>(), geluRes, AscendC::RoundMode::CAST_RINT, calCount);
    } else if (dstType_ == DT_FLOAT8_E4M3FN) {
        Cast<fp8_e4m3fn_t, float>(
            outLocal.ReinterpretCast<fp8_e4m3fn_t>(), geluRes, AscendC::RoundMode::CAST_RINT, calCount);
    } else if (dstType_ == DT_HIFLOAT8) {
        Cast<hifloat8_t, float>(outLocal.ReinterpretCast<hifloat8_t>(), geluRes, roundMode_, calCount);
    }

    inQueue_.FreeTensor(geluRes);
    workspaceQueue_.EnQue<int8_t>(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluDynamicQuantWorkspace<T1, T2>::ComputeDynamicQuantRegbase(
    LocalTensor<float>& geluRes, LocalTensor<float>& workspaceLocal, LocalTensor<float>& scaleLocalFp32,
    LocalTensor<float>& maxValueRes, int32_t calCount)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
    uint16_t loopNum = (calCount + vl - 1) / vl;
    uint32_t vlSize = vl;
    __ubuf__ float* xAddr = (__ubuf__ float*)geluRes.GetPhyAddr();
    __ubuf__ float* smoothScaleAddr = (__ubuf__ float*)scaleLocalFp32.GetPhyAddr();
    __ubuf__ float* maxValueAddr = (__ubuf__ float*)maxValueRes.GetPhyAddr();

    __ubuf__ float* workspaceLocalAddr = (__ubuf__ float*)workspaceLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> geluResReg;
        AscendC::MicroAPI::RegTensor<float> scaleReg;
        AscendC::MicroAPI::RegTensor<float> absReg;
        AscendC::MicroAPI::RegTensor<float> reduceMaxReg;
        AscendC::MicroAPI::RegTensor<float> vregMax;
        AscendC::MicroAPI::RegTensor<float> maxTemReg;
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg ureg0;

        AscendC::MicroAPI::Duplicate(vregMax, 0.0);
        uint32_t sreg0 = calCount;
        for (uint16_t i = 0; i < loopNum; i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
            AscendC::MicroAPI::DataCopy(geluResReg, xAddr + i * vl);
            // compute smoothscale
            AscendC::MicroAPI::DataCopy(scaleReg, smoothScaleAddr + i * vl);
            AscendC::MicroAPI::Mul(geluResReg, geluResReg, scaleReg, preg0);
            AscendC::MicroAPI::DataCopy(workspaceLocalAddr + i * vl, geluResReg, preg0);
            AscendC::MicroAPI::Abs(absReg, geluResReg, preg0);
            AscendC::MicroAPI::Max(vregMax, absReg, vregMax, preg1);
        }
        AscendC::MicroAPI::ReduceMax(reduceMaxReg, vregMax, preg1);
        AscendC::MicroAPI::DataCopy(maxTemReg, maxValueAddr);
        AscendC::MicroAPI::Max(maxTemReg, reduceMaxReg, maxTemReg, preg1);
        AscendC::MicroAPI::DataCopy(maxValueAddr, maxTemReg, preg1);
    }
}

} // namespace GeluQuantALL
#endif