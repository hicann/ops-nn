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
 * \file gelu_static_quant_regbase.h
 * \brief
 */

#ifndef GELU_STATIC_QUANT_REGBASE_H
#define GELU_STATIC_QUANT_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gelu_quant_base_regbase.h"

namespace GeluQuantALL {
using namespace AscendC;
// static quant function template   单行处理或者大尾轴UB内for循环

template <typename T1, typename T2>
class GeluQuant : public GeluQuantBase
{
public:
    __aicore__ inline GeluQuant(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale, GM_ADDR workspace,
        const GeluQuantTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPerLoop(int64_t endAxisOffset, int32_t calCount);
    __aicore__ inline void ProcessOptionalInput(
        LocalTensor<float>& optionalLocalFp32, GlobalTensor<T2>& optionalGlobal, int64_t endAxisOffset,
        int32_t calCount);
    __aicore__ inline void CopyIn(int64_t endAxisOffset, int32_t calCount, int64_t loopNum);
    __aicore__ inline void Compute(
        LocalTensor<float>& scaleLocalFp32, LocalTensor<float>& offsetLocalFp32, int32_t calCount);
    __aicore__ inline void CopyOut(int64_t endAxisOffset, int32_t calCount, int64_t loopNum);
    __aicore__ inline void ComputeGelu(LocalTensor<float>& geluRes, int32_t calCount);
    __aicore__ inline void ComputeQuant(
        LocalTensor<float>& geluRes, LocalTensor<float>& scaleLocalFp32, LocalTensor<float>& offsetLocalFp32,
        int32_t calCount);
    __aicore__ inline void ComputeScaleOffset(
        LocalTensor<float>& geluRes, LocalTensor<float>& scaleLocalFp32, LocalTensor<float>& offsetLocalFp32,
        uint32_t calCount);

private:
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> offsetQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> tempQueue_;
    TBuf<TPosition::VECCALC> castQueue_;

    /* global memory address */
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> inputScaleGm_;
    GlobalTensor<T2> inputOffsetGm_;
    GlobalTensor<int8_t> yGm_;
};

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::Init(
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
    uint64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_ * endAxisLen_;

    // shield normal core and tail core
    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreProcessNum_ = tailCoreProcessNum_;
    } else {
        curCoreProcessNum_ = normalCoreProcessNum_;
    }

    xGm_.SetGlobalBuffer((__gm__ T1*)x + intraCoreOffset);
    inputScaleGm_.SetGlobalBuffer((__gm__ T2*)inputScale);
    inputOffsetGm_.SetGlobalBuffer((__gm__ T2*)inputOffset);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y + intraCoreOffset);

    pipe.InitBuffer(inQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(tempQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(castQueue_, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(outQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(scaleQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
    pipe.InitBuffer(offsetQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(float));
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t endAxisLoopNum = endAxisLen_ / coexistentNodeElementNum_;
    int64_t endAxisLoopTail = endAxisLen_ % coexistentNodeElementNum_;
    for (int64_t endAxisLoopIndex = 0; endAxisLoopIndex < endAxisLoopNum; endAxisLoopIndex++) {
        ProcessPerLoop(endAxisLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_);
    }
    if (endAxisLoopTail != 0) {
        ProcessPerLoop(endAxisLoopNum * coexistentNodeElementNum_, endAxisLoopTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::ProcessPerLoop(int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<float> scaleLocalFp32 = scaleQueue_.AllocTensor<float>();
    LocalTensor<float> offsetLocalFp32 = offsetQueue_.AllocTensor<float>();
    if (inputScaleType_ == NORMAL_TENSOR) {
        ProcessOptionalInput(scaleLocalFp32, inputScaleGm_, endAxisOffset, calCount);
    }

    if (inputOffsetType_ == NORMAL_TENSOR) {
        ProcessOptionalInput(offsetLocalFp32, inputOffsetGm_, endAxisOffset, calCount);
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        CopyIn(endAxisOffset, calCount, loopNum);
        Compute(scaleLocalFp32, offsetLocalFp32, calCount);
        CopyOut(endAxisOffset, calCount, loopNum);
    }
    scaleQueue_.FreeTensor(scaleLocalFp32);
    offsetQueue_.FreeTensor(offsetLocalFp32);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::CopyIn(int64_t endAxisOffset, int32_t calCount, int64_t loopNum)

{
    LocalTensor<T1> xLocal = inQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T1)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T1> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};
    DataCopyPad(xLocal, xGm_[endAxisLen_ * loopNum + endAxisOffset], copyParams, padParams);

    inQueue_.EnQue(xLocal);
}

// 搬入并cast成float类型
template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::ProcessOptionalInput(
    LocalTensor<float>& optionalLocalFp32, GlobalTensor<T2>& optionalGlobal, int64_t endAxisOffset, int32_t calCount)
{
    LocalTensor<T2> optionalInput = inQueue_.AllocTensor<T2>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T2)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T2> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};

    DataCopyPad(optionalInput, optionalGlobal[endAxisOffset], copyParams, padParams);

    inQueue_.EnQue(optionalInput);
    LocalTensor<T2> tempLocal = inQueue_.DeQue<T2>();

    if constexpr (IsSameType<T2, float>::value) {
        Muls(optionalLocalFp32, tempLocal, 1.0f, calCount);
    } else {
        Cast(optionalLocalFp32, tempLocal, RoundMode::CAST_NONE, calCount);
    }

    inQueue_.FreeTensor(optionalInput);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::Compute(
    LocalTensor<float>& scaleLocalFp32, LocalTensor<float>& offsetLocalFp32, int32_t calCount)
{
    LocalTensor<float> geluRes = tempQueue_.Get<float>();
    ComputeGelu(geluRes, calCount);
    ComputeQuant(geluRes, scaleLocalFp32, offsetLocalFp32, calCount);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::CopyOut(int64_t endAxisOffset, int32_t calCount, int64_t loopNum)
{
    LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParamsInt8{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(int8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[endAxisLen_ * loopNum + endAxisOffset], outLocal, copyParamsInt8);

    outQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::ComputeGelu(LocalTensor<float>& geluRes, int32_t calCount)
{
    LocalTensor<T1> xLocal = inQueue_.DeQue<T1>();

    if (approximate_ == APPROXIMATE_NONE) {
        ComputeGeluErf(xLocal, geluRes, calCount);
    } else {
        ComputeGeluTanh(xLocal, geluRes, calCount);
    }
    inQueue_.FreeTensor(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GeluQuant<T1, T2>::ComputeQuant(
    LocalTensor<float>& geluRes, LocalTensor<float>& scaleLocalFp32, LocalTensor<float>& offsetLocalFp32,
    int32_t calCount)
{
    if (inputScaleType_ == NORMAL_TENSOR) {
        Mul(geluRes, geluRes, scaleLocalFp32, calCount);
    }

    if (inputOffsetType_ == NORMAL_TENSOR) {
        Add(geluRes, geluRes, offsetLocalFp32, calCount);
    }

    LocalTensor<half> castFp16 = castQueue_.Get<half>();
    LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();

    if (dstType_ == DT_HIFLOAT8) {
        if (roundMode_ == AscendC::RoundMode::CAST_HYBRID) {
            Cast<hifloat8_t, float>(outLocal.ReinterpretCast<hifloat8_t>(), geluRes, RoundMode::CAST_HYBRID, calCount);
        } else if (roundMode_ == AscendC::RoundMode::CAST_ROUND) {
            Cast<hifloat8_t, float>(outLocal.ReinterpretCast<hifloat8_t>(), geluRes, RoundMode::CAST_ROUND, calCount);
        }
    } else if (dstType_ == DT_FLOAT8_E5M2) {
        Cast<fp8_e5m2_t, float>(outLocal.ReinterpretCast<fp8_e5m2_t>(), geluRes, RoundMode::CAST_RINT, calCount);
    } else if (dstType_ == DT_FLOAT8_E4M3FN) {
        Cast<fp8_e4m3fn_t, float>(outLocal.ReinterpretCast<fp8_e4m3fn_t>(), geluRes, RoundMode::CAST_RINT, calCount);
    } else if (dstType_ == DT_INT8) {
        Cast(castFp16, geluRes, RoundMode::CAST_ODD, calCount);
        if (roundMode_ == AscendC::RoundMode::CAST_RINT) {
            Cast(outLocal, castFp16, RoundMode::CAST_RINT, calCount);
        } else if (roundMode_ == AscendC::RoundMode::CAST_ROUND) {
            Cast(outLocal, castFp16, RoundMode::CAST_ROUND, calCount);
        }
    }

    outQueue_.EnQue<int8_t>(outLocal);
}
} // namespace GeluQuantALL
#endif