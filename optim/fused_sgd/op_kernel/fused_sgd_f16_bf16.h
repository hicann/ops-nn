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
 * \file fused_sgd_f16_bf16.h
 * \brief
 */

#ifndef FUSED_SGD_F16_BF16_H
#define FUSED_SGD_F16_BF16_H

#include "fused_sgd_base.h"

namespace FusedSgd {
using namespace AscendC;

template <typename T>
class FusedSgdF16Bf16 : public FusedSgdBase<T> {
public:
    __aicore__ inline FusedSgdF16Bf16(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR params, GM_ADDR grads, GM_ADDR momentum_buffer_list, GM_ADDR grad_scale,
                                GM_ADDR params_ref, GM_ADDR grads_ref, GM_ADDR momentum_buffer_list_out,
                                const FusedSgdTilingData& tiling, uint64_t tensorStart, uint64_t tensorEnd);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void Compute(const uint64_t index, const uint64_t dataCount);

    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;

    GlobalTensor<T> gmParamsF16;
    GlobalTensor<T> gmGradsF16;
    GlobalTensor<T> gmMomentumBufferF16;
    GlobalTensor<float> gmGradScaleF16;
    GlobalTensor<float> gmFoundInfF16;
    GlobalTensor<T> gmParamsRefF16;
    GlobalTensor<T> gmGradsRefF16;
    GlobalTensor<T> gmMomentumBufferOutF16;

    ListTensorDesc paramsListF16_;
    ListTensorDesc gradsListF16_;
    ListTensorDesc momentumListF16_;
    ListTensorDesc paramsRefListF16_;
    ListTensorDesc gradsRefListF16_;
    ListTensorDesc momentumOutListF16_;
    TensorDesc<uint64_t> descF16_;

    float gradScaleValueF16;
    uint64_t hasGradScaleF16;
    uint64_t tensorStartF16_;
    uint64_t tensorEndF16_;
    int64_t paramsOffsetF16;
    int64_t gradsOffsetF16;
    int64_t momentumOffsetF16;
    int64_t paramsOffsetCF16;
    int64_t gradsOffsetCF16;
    int64_t momentumOffsetCF16;
    TPipe* pipe_;
    const FusedSgdTilingData* tiling_;
};

template <typename T>
__aicore__ inline void FusedSgdF16Bf16<T>::Init(GM_ADDR params, GM_ADDR grads, GM_ADDR momentum_buffer_list,
                                                GM_ADDR grad_scale, GM_ADDR params_ref, GM_ADDR grads_ref,
                                                GM_ADDR momentum_buffer_list_out, const FusedSgdTilingData& tiling,
                                                uint64_t tensorStart, uint64_t tensorEnd)
{
    this->InitData(tiling);
    tiling_ = &tiling;
    tensorStartF16_ = tensorStart;
    tensorEndF16_ = tensorEnd;

    paramsListF16_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(params));
    gradsListF16_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(grads));
    paramsRefListF16_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(params_ref));
    gradsRefListF16_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(grads_ref));
    if (this->useMomentum) {
        momentumListF16_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(momentum_buffer_list));
        momentumOutListF16_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(momentum_buffer_list_out));
    }

    // UB Buffer布局: inQue = [原始类型(param+grad+momentum)] + [FP32(param+grad+momentum)]
    // 前半存原始类型，后半(偏移3*sizeof(T))存Cast后的FP32
    pipe_->InitBuffer(inQue, BUFFER_NUM, this->coreCalcMax * (sizeof(T) + sizeof(float)) * 3);
    pipe_->InitBuffer(outQue, BUFFER_NUM, this->coreCalcMax * sizeof(float) * 3);

    paramsOffsetF16 = this->coreCalcMax * INDEX_PARAMS;
    gradsOffsetF16 = this->coreCalcMax * INDEX_GRADS;
    momentumOffsetF16 = this->coreCalcMax * INDEX_MOMENTUM_BUFFER;
    // FP32区域偏移 = 前半3份原始类型 + 对应的FP32偏移
    paramsOffsetCF16 = this->coreCalcMax * 3 + paramsOffsetF16;
    gradsOffsetCF16 = this->coreCalcMax * 3 + gradsOffsetF16;
    momentumOffsetCF16 = this->coreCalcMax * 3 + momentumOffsetF16;

    hasGradScaleF16 = 0;
    if (this->useGradScale) {
        gmGradScaleF16.SetGlobalBuffer((__gm__ float*)grad_scale, 1);
        gradScaleValueF16 = static_cast<float>(gmGradScaleF16.GetValue(0));
        hasGradScaleF16 = 1;
    }
}

template <typename T>
__aicore__ inline void FusedSgdF16Bf16<T>::Compute(const uint64_t index, const uint64_t dataCount)
{
    uint64_t offset = index * this->coreCalcMax;
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};

    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<float> outLocal = outQue.AllocTensor<float>();

    PipeSync<AscendC::HardEvent::MTE3_MTE2>();
    PipeSync<AscendC::HardEvent::S_MTE2>();
    PipeSync<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(inLocal[paramsOffsetF16], gmParamsF16[offset], copyParams, padParams);
    DataCopyPad(inLocal[gradsOffsetF16], gmGradsF16[offset], copyParams, padParams);
    if (this->useMomentum) {
        DataCopyPad(inLocal[momentumOffsetF16], gmMomentumBufferF16[offset], copyParams, padParams);
    }
    PipeSync<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();

    LocalTensor<float> inLocalC = inLocal[this->coreCalcMax * 3].template ReinterpretCast<float>();
    Cast(inLocalC[paramsOffsetF16], inLocal[paramsOffsetF16], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(inLocalC[gradsOffsetF16], inLocal[gradsOffsetF16], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(inLocalC[momentumOffsetF16], inLocal[momentumOffsetF16], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();

    // Step 1: 梯度缩放，并Cast回原始类型写回
    if (hasGradScaleF16) {
        float invGradScale = 1.0f / gradScaleValueF16;
        Muls(inLocalC[gradsOffsetF16], inLocalC[gradsOffsetF16], invGradScale, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(inLocal[gradsOffsetF16], inLocalC[gradsOffsetF16], RoundMode::CAST_RINT, dataCount);
        PipeSync<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(gmGradsRefF16[offset], inLocal[gradsOffsetF16], copyParams);
        PipeSync<AscendC::HardEvent::MTE3_V>();
    }
    // Step 2: 最大化处理
    if (this->maximize) {
        Muls(inLocalC[gradsOffsetF16], inLocalC[gradsOffsetF16], -1.0f, dataCount);
        PipeBarrier<PIPE_V>();
    }
    // Step 3: 权重衰减
    if (this->weightDecay != 0.0f) {
        Muls(outLocal[gradsOffsetF16], inLocalC[paramsOffsetF16], this->weightDecay, dataCount);
        PipeBarrier<PIPE_V>();
        Add(inLocalC[gradsOffsetF16], inLocalC[gradsOffsetF16], outLocal[gradsOffsetF16], dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 4: 动量更新 (FP32计算，写回时Cast回原始类型)
    if (this->useMomentum) {
        if (this->isFirstStep) {
            Muls(outLocal[momentumOffsetF16], inLocalC[gradsOffsetF16], 1.0f, dataCount);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(outLocal[momentumOffsetF16], inLocalC[momentumOffsetF16], this->momentum, dataCount);
            PipeBarrier<PIPE_V>();
            Muls(outLocal[paramsOffsetF16], inLocalC[gradsOffsetF16], 1.0f - this->dampening, dataCount);
            PipeBarrier<PIPE_V>();
            Add(outLocal[momentumOffsetF16], outLocal[momentumOffsetF16], outLocal[paramsOffsetF16], dataCount);
            PipeBarrier<PIPE_V>();
        }
        // 动量Cast回原始类型并写回GM
        Cast(inLocal[momentumOffsetF16], outLocal[momentumOffsetF16], RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
        PipeSync<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(gmMomentumBufferOutF16[offset], inLocal[momentumOffsetF16], copyParams);
        PipeSync<AscendC::HardEvent::MTE3_V>();
        // Nesterov: grad = grad + momentum * buf
        if (this->nesterov) {
            Muls(outLocal[momentumOffsetF16], outLocal[momentumOffsetF16], this->momentum, dataCount);
            PipeBarrier<PIPE_V>();
            Add(inLocalC[gradsOffsetF16], outLocal[momentumOffsetF16], inLocalC[gradsOffsetF16], dataCount);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(inLocalC[gradsOffsetF16], outLocal[momentumOffsetF16], 1.0f, dataCount);
            PipeBarrier<PIPE_V>();
        }
    }

    // Step 5: 参数更新 (param = param - lr * grad)，Cast回原始类型写回
    Muls(inLocalC[gradsOffsetF16], inLocalC[gradsOffsetF16], this->lr, dataCount);
    PipeBarrier<PIPE_V>();
    Sub(inLocalC[gradsOffsetF16], inLocalC[paramsOffsetF16], inLocalC[gradsOffsetF16], dataCount);
    PipeBarrier<PIPE_V>();
    Cast(inLocal[paramsOffsetF16], inLocalC[gradsOffsetF16], RoundMode::CAST_RINT, dataCount);
    PipeBarrier<PIPE_V>();
    PipeSync<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(gmParamsRefF16[offset], inLocal[paramsOffsetF16], copyParams);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void FusedSgdF16Bf16<T>::Process()
{
    for (uint64_t idx = tensorStartF16_; idx < tensorEndF16_; idx++) {
        uint64_t buf[10];
        descF16_.SetShapeAddr(&buf[0]);
        paramsListF16_.GetDesc(descF16_, static_cast<uint32_t>(idx));

        uint64_t tensorDataNum = 1;
        for (uint32_t j = 0; j < descF16_.GetDim(); j++) {
            tensorDataNum *= descF16_.GetShape(j);
        }
        if (tensorDataNum == 0) {
            continue;
        }

        gmParamsF16.SetGlobalBuffer(paramsListF16_.GetDataPtr<T>(idx), tensorDataNum);
        gmGradsF16.SetGlobalBuffer(gradsListF16_.GetDataPtr<T>(idx), tensorDataNum);
        gmParamsRefF16.SetGlobalBuffer(paramsRefListF16_.GetDataPtr<T>(idx), tensorDataNum);
        gmGradsRefF16.SetGlobalBuffer(gradsRefListF16_.GetDataPtr<T>(idx), tensorDataNum);
        if (this->useMomentum) {
            gmMomentumBufferF16.SetGlobalBuffer(momentumListF16_.GetDataPtr<T>(idx), tensorDataNum);
            gmMomentumBufferOutF16.SetGlobalBuffer(momentumOutListF16_.GetDataPtr<T>(idx), tensorDataNum);
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

#endif // FUSED_SGD_F16_BF16_H
