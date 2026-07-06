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
 * \file fused_sgd_f32.h
 * \brief
 */

#ifndef FUSED_SGD_F32_H
#define FUSED_SGD_F32_H

#include "fused_sgd_base.h"

namespace FusedSgd {
using namespace AscendC;

template <typename T>
class FusedSgdF32 : public FusedSgdBase<T> {
public:
    __aicore__ inline FusedSgdF32(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR params, GM_ADDR grads, GM_ADDR momentum_buffer_list, GM_ADDR grad_scale,
                                GM_ADDR params_ref, GM_ADDR grads_ref, GM_ADDR momentum_buffer_list_out,
                                const FusedSgdTilingData& tiling, uint64_t tensorStart, uint64_t tensorEnd);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void Compute(const uint64_t index, const uint64_t dataCount);

    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;

    GlobalTensor<T> gmParams;
    GlobalTensor<T> gmGrads;
    GlobalTensor<T> gmMomentumBuffer;
    GlobalTensor<float> gmGradScale;
    GlobalTensor<float> gmFoundInf;
    GlobalTensor<T> gmParamsRef;
    GlobalTensor<T> gmGradsRef;
    GlobalTensor<T> gmMomentumBufferOut;

    ListTensorDesc paramsList_;
    ListTensorDesc gradsList_;
    ListTensorDesc momentumList_;
    ListTensorDesc paramsRefList_;
    ListTensorDesc gradsRefList_;
    ListTensorDesc momentumOutList_;
    TensorDesc<uint64_t> desc_;

    float gradScaleValue;
    uint64_t hasGradScale;
    uint64_t tensorStart_;
    uint64_t tensorEnd_;
    int64_t paramsOffset;
    int64_t gradsOffset;
    int64_t momentumOffset;
    TPipe* pipe_;
    const FusedSgdTilingData* tiling_;
};

template <typename T>
__aicore__ inline void FusedSgdF32<T>::Init(GM_ADDR params, GM_ADDR grads, GM_ADDR momentum_buffer_list,
                                            GM_ADDR grad_scale, GM_ADDR params_ref, GM_ADDR grads_ref,
                                            GM_ADDR momentum_buffer_list_out, const FusedSgdTilingData& tiling,
                                            uint64_t tensorStart, uint64_t tensorEnd)
{
    this->InitData(tiling);
    tiling_ = &tiling;
    tensorStart_ = tensorStart;
    tensorEnd_ = tensorEnd;

    paramsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(params));
    gradsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(grads));
    paramsRefList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(params_ref));
    gradsRefList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(grads_ref));
    if (this->useMomentum) {
        momentumList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(momentum_buffer_list));
        momentumOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(momentum_buffer_list_out));
    }

    pipe_->InitBuffer(inQue, BUFFER_NUM, this->coreCalcMax * sizeof(T) * 3);
    pipe_->InitBuffer(outQue, BUFFER_NUM, this->coreCalcMax * sizeof(T) * 3);

    paramsOffset = this->coreCalcMax * INDEX_PARAMS;
    gradsOffset = this->coreCalcMax * INDEX_GRADS;
    momentumOffset = this->coreCalcMax * INDEX_MOMENTUM_BUFFER;

    hasGradScale = 0;
    if (this->useGradScale) {
        gmGradScale.SetGlobalBuffer((__gm__ float*)grad_scale, 1);
        gradScaleValue = static_cast<float>(gmGradScale.GetValue(0));
        hasGradScale = 1;
    }
}

template <typename T>
__aicore__ inline void FusedSgdF32<T>::Compute(const uint64_t index, const uint64_t dataCount)
{
    uint64_t offset = index * this->coreCalcMax;
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};

    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<T> outLocal = outQue.AllocTensor<T>();

    PipeSync<AscendC::HardEvent::MTE3_MTE2>();
    PipeSync<AscendC::HardEvent::S_MTE2>();
    PipeSync<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(inLocal[paramsOffset], gmParams[offset], copyParams, padParams);
    DataCopyPad(inLocal[gradsOffset], gmGrads[offset], copyParams, padParams);
    if (this->useMomentum) {
        DataCopyPad(inLocal[momentumOffset], gmMomentumBuffer[offset], copyParams, padParams);
    }
    PipeSync<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();

    // Step 1: 梯度缩放 (grad = grad / grad_scale)，并回写缩放后的梯度
    if (hasGradScale) {
        float invGradScale = 1.0f / gradScaleValue;
        Muls(inLocal[gradsOffset], inLocal[gradsOffset], invGradScale, dataCount);
        PipeBarrier<PIPE_V>();
        PipeSync<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(gmGradsRef[offset], inLocal[gradsOffset], copyParams);
        PipeSync<AscendC::HardEvent::MTE3_V>();
    }
    // Step 2: 最大化处理 (grad = -grad)，用于最大化目标函数
    if (this->maximize) {
        Muls(inLocal[gradsOffset], inLocal[gradsOffset], -1.0f, dataCount);
        PipeBarrier<PIPE_V>();
    }
    // Step 3: 权重衰减 (grad = grad + weight_decay * param)
    if (this->weightDecay != 0.0f) {
        Muls(outLocal[gradsOffset], inLocal[paramsOffset], this->weightDecay, dataCount);
        PipeBarrier<PIPE_V>();
        Add(inLocal[gradsOffset], inLocal[gradsOffset], outLocal[gradsOffset], dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 4: 动量更新 (SGD with Momentum)
    if (this->useMomentum) {
        if (this->isFirstStep) {
            Muls(outLocal[momentumOffset], inLocal[gradsOffset], 1.0f, dataCount);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(outLocal[momentumOffset], inLocal[momentumOffset], this->momentum, dataCount);
            PipeBarrier<PIPE_V>();
            Muls(outLocal[paramsOffset], inLocal[gradsOffset], 1.0f - this->dampening, dataCount);
            PipeBarrier<PIPE_V>();
            Add(outLocal[momentumOffset], outLocal[momentumOffset], outLocal[paramsOffset], dataCount);
            PipeBarrier<PIPE_V>();
        }
        // 动量写回GM
        PipeSync<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(gmMomentumBufferOut[offset], outLocal[momentumOffset], copyParams);
        PipeSync<AscendC::HardEvent::MTE3_V>();
        // Nesterov: grad = grad + momentum * buf
        if (this->nesterov) {
            Muls(outLocal[momentumOffset], outLocal[momentumOffset], this->momentum, dataCount);
            PipeBarrier<PIPE_V>();
            Add(inLocal[gradsOffset], outLocal[momentumOffset], inLocal[gradsOffset], dataCount);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(inLocal[gradsOffset], outLocal[momentumOffset], 1.0f, dataCount);
            PipeBarrier<PIPE_V>();
        }
    }

    // Step 5: 参数更新 (param = param - lr * grad)，并写回GM
    Muls(outLocal[gradsOffset], inLocal[gradsOffset], this->lr, dataCount);
    PipeBarrier<PIPE_V>();
    Sub(outLocal[paramsOffset], inLocal[paramsOffset], outLocal[gradsOffset], dataCount);
    PipeBarrier<PIPE_V>();
    PipeSync<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(gmParamsRef[offset], outLocal[paramsOffset], copyParams);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void FusedSgdF32<T>::Process()
{
    for (uint64_t idx = tensorStart_; idx < tensorEnd_; idx++) {
        uint64_t buf[10];
        desc_.SetShapeAddr(buf);
        paramsList_.GetDesc(desc_, static_cast<uint32_t>(idx));

        uint64_t tensorDataNum = 1;
        for (uint32_t j = 0; j < desc_.GetDim(); j++) {
            tensorDataNum *= desc_.GetShape(j);
        }
        if (tensorDataNum == 0) {
            continue;
        }

        gmParams.SetGlobalBuffer(paramsList_.GetDataPtr<T>(idx));
        gmGrads.SetGlobalBuffer(gradsList_.GetDataPtr<T>(idx));
        gmParamsRef.SetGlobalBuffer(paramsRefList_.GetDataPtr<T>(idx));
        gmGradsRef.SetGlobalBuffer(gradsRefList_.GetDataPtr<T>(idx));
        if (this->useMomentum) {
            gmMomentumBuffer.SetGlobalBuffer(momentumList_.GetDataPtr<T>(idx));
            gmMomentumBufferOut.SetGlobalBuffer(momentumOutList_.GetDataPtr<T>(idx));
        }

        uint64_t loopNum = (tensorDataNum + this->coreCalcMax - 1) / this->coreCalcMax;
        for (uint64_t n = 0; n < loopNum - 1; n++) {
            Compute(n, this->coreCalcMax);
        }
        uint64_t lastCount = tensorDataNum - this->coreCalcMax * (loopNum - 1);
        Compute(loopNum - 1, lastCount);
    }
}

} // namespace FusedSgd

#endif // FUSED_SGD_F32_H
