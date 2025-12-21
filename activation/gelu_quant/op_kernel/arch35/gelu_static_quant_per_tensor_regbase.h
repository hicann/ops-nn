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
 * \file gelu_static_quant_per_tensor_regbase.h
 * \brief
 */

#ifndef GELU_STATIC_QUANT_PER_TENSOR_REGBASE_H
#define GELU_STATIC_QUANT_PER_TENSOR_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "gelu_quant_base_regbase.h"

namespace GeluQuantALL {
using namespace AscendC;
// static quant per-tensor   elewise切分
template <typename T1, typename T2>
class StaticQuantPerTensor : public GeluQuantBase
{
public:
    __aicore__ inline StaticQuantPerTensor(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR inputScale, GM_ADDR inputOffset, GM_ADDR y, GM_ADDR outScale, GM_ADDR workspace,
        const GeluQuantTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessScalarInput();
    __aicore__ inline void ProcessPerLoop(int64_t offset, int32_t calCount);
    __aicore__ inline void CopyIn(int64_t offset, int32_t calCount);
    __aicore__ inline void Compute(int32_t calCount);
    __aicore__ inline void CopyOut(int64_t offset, int32_t calCount);
    __aicore__ inline void ComputeGelu(LocalTensor<float>& geluRes, int32_t calCount);
    __aicore__ inline void ComputePerTensorQuant(LocalTensor<float>& geluRes, int32_t calCount);

private:
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TBuf<TPosition::VECCALC> tempQueue_;
    TBuf<TPosition::VECCALC> castQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;

    /* global memory address */
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> inputScaleGm_;
    GlobalTensor<T2> inputOffsetGm_;
    GlobalTensor<int8_t> yGm_;
};

template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::Init(
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
    uint64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_;

    // shield normal core and tail core
    // curCoreProcessNum_为当前核处理多少个元素
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
    pipe.InitBuffer(outQueue_, BUFFER_NUM, coexistentNodeElementNum_ * sizeof(int8_t));
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::ProcessScalarInput()
{
    if (inputScaleType_ == SCALAR_TENSOR) {
        inputScaleScalar_ = GetFloatScalar(inputScaleGm_);
    }

    if (inputOffsetType_ == SCALAR_TENSOR) {
        inputOffsetScalar_ = GetFloatScalar(inputOffsetGm_);
    }
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    ProcessScalarInput();
    // coexistentNodeElementNum_为单个UB用满一次可以处理多少个元素
    int64_t ubLoopNum = curCoreProcessNum_ / coexistentNodeElementNum_;
    int64_t ubLoopTail = curCoreProcessNum_ % coexistentNodeElementNum_;
    for (int64_t ubLoopIndex = 0; ubLoopIndex < ubLoopNum; ubLoopIndex++) {
        ProcessPerLoop(ubLoopIndex * coexistentNodeElementNum_, coexistentNodeElementNum_);
    }
    if (ubLoopTail != 0) {
        ProcessPerLoop(ubLoopNum * coexistentNodeElementNum_, ubLoopTail);
    }
}

// 单个UB处理一次维度，第一个参数是偏移，第二个参数是处理的元素个数，这边每个核总计算量的GM偏移已经在init完成
template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::ProcessPerLoop(int64_t offset, int32_t calCount)
{
    CopyIn(offset, calCount);
    Compute(calCount);
    CopyOut(offset, calCount);
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::CopyIn(int64_t offset, int32_t calCount)
{
    LocalTensor<T1> xLocal = inQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(T1)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T1> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};
    DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);

    inQueue_.EnQue(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::Compute(int32_t calCount)
{
    LocalTensor<float> geluRes = tempQueue_.Get<float>();
    ComputeGelu(geluRes, calCount);
    ComputePerTensorQuant(geluRes, calCount);
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::CopyOut(int64_t offset, int32_t calCount)
{
    LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
    DataCopyExtParams copyParamsInt8{
        static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(int8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[offset], outLocal, copyParamsInt8);

    outQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::ComputeGelu(LocalTensor<float>& geluRes, int32_t calCount)
{
    LocalTensor<T1> xLocal = inQueue_.DeQue<T1>();
    LocalTensor<float> castFp32 = castQueue_.Get<float>();
    if constexpr (IsSameType<T1, float>::value) {
        Muls(castFp32, xLocal, 1.0f, calCount);
    } else {
        Cast(castFp32, xLocal, RoundMode::CAST_NONE, calCount);
    }

    inQueue_.FreeTensor(xLocal);

    if (approximate_ == APPROXIMATE_NONE) {
        ComputeGeluErf(castFp32, geluRes, calCount);
    } else {
        ComputeGeluTanh(castFp32, geluRes, calCount);
    }
}

template <typename T1, typename T2>
__aicore__ inline void StaticQuantPerTensor<T1, T2>::ComputePerTensorQuant(
    LocalTensor<float>& geluRes, int32_t calCount)
{
    if (inputScaleType_ == SCALAR_TENSOR) {
        Muls(geluRes, geluRes, inputScaleScalar_, calCount);
    }

    if (inputOffsetType_ == SCALAR_TENSOR) {
        Adds(geluRes, geluRes, inputOffsetScalar_, calCount);
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