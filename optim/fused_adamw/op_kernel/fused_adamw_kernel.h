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
 * \file fused_adamw_kernel.h
 * \brief unified fused_adamw kernel for FP32/FP16/BF16
 */

#ifndef _FUSED_ADAMW_KERNEL_H_
#define _FUSED_ADAMW_KERNEL_H_

#include <type_traits>
#include "fused_adamw_base.h"

namespace FusedAdamW {
using namespace AscendC;

template <typename T>
class FusedAdamWKernel : public FusedAdamWBase<T> {
    static constexpr bool isFloat = std::is_same_v<T, float>;

public:
    __aicore__ inline FusedAdamWKernel(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR params, GM_ADDR grads, GM_ADDR exp_avgs, GM_ADDR exp_avg_sqs,
                                GM_ADDR max_exp_avg_sqs, GM_ADDR state_steps, GM_ADDR grad_scale, GM_ADDR found_inf,
                                GM_ADDR params_ref, GM_ADDR exp_avgs_ref, GM_ADDR exp_avg_sqs_ref,
                                GM_ADDR max_exp_avg_sqs_ref, const FusedAdamWTilingData& tiling, uint64_t tensorStart,
                                uint64_t tensorEnd);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void Compute(const uint64_t index, const uint64_t dataCount);
    __aicore__ inline void ComputeFP32(const uint64_t index, const uint64_t dataCount);
    __aicore__ inline void ComputeFP16Bf16(const uint64_t index, const uint64_t dataCount);
    // __aicore__ inline void ComputeBiasCorrectionUC(float& biasCorrection1, float& biasCorrection2Sqrt);
    __aicore__ inline float ScalarPow(float x, float y);

    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> tempBuf_;
    TBuf<QuePosition::VECCALC> tempBuf2_;

    GlobalTensor<T> gmParams;
    GlobalTensor<T> gmGrads;
    GlobalTensor<T> gmExpAvg;
    GlobalTensor<T> gmExpAvgSq;
    GlobalTensor<T> gmMaxExpAvgSq;
    GlobalTensor<T> gmMaxExpAvgSqOut;
    GlobalTensor<float> gmStateSteps;
    GlobalTensor<float> gmGradScale;
    GlobalTensor<float> gmFoundInf;
    GlobalTensor<T> gmParamsOut;
    GlobalTensor<T> gmExpAvgOut;
    GlobalTensor<T> gmExpAvgSqOut;

    ListTensorDesc paramsList_;
    ListTensorDesc gradsList_;
    ListTensorDesc expAvgsList_;
    ListTensorDesc expAvgSqsList_;
    ListTensorDesc maxExpAvgSqsList_;
    ListTensorDesc stateStepsList_;
    ListTensorDesc paramsOutList_;
    ListTensorDesc expAvgsOutList_;
    ListTensorDesc expAvgSqsOutList_;
    ListTensorDesc maxExpAvgSqsOutList_;
    TensorDesc<uint64_t> desc_;

    float gradScaleValue;
    float biasCorrection2Sqrt;
    float stepSize;
    uint64_t hasGradScale;
    uint64_t tensorStart_;
    uint64_t tensorEnd_;
    int64_t paramsOffset;
    int64_t gradsOffset;
    int64_t expAvgOffset;
    int64_t expAvgSqOffset;
    int64_t maxExpAvgSqOffset;
    int64_t paramsOutOffset;
    int64_t expAvgOutOffset;
    int64_t expAvgSqOutOffset;
    int64_t maxExpAvgSqOutOffset;
    TPipe* pipe_;
    const FusedAdamWTilingData* tiling_;
    int32_t tensorCount_;
    int32_t tensorCountOut_;
};

// ==================== Init ====================

template <typename T>
__aicore__ inline void FusedAdamWKernel<T>::Init(GM_ADDR params, GM_ADDR grads, GM_ADDR exp_avgs, GM_ADDR exp_avg_sqs,
                                                 GM_ADDR max_exp_avg_sqs, GM_ADDR state_steps, GM_ADDR grad_scale,
                                                 GM_ADDR found_inf, GM_ADDR params_ref, GM_ADDR exp_avgs_ref,
                                                 GM_ADDR exp_avg_sqs_ref, GM_ADDR max_exp_avg_sqs_ref,
                                                 const FusedAdamWTilingData& tiling, uint64_t tensorStart,
                                                 uint64_t tensorEnd)
{
    this->InitData(tiling);
    tiling_ = &tiling;
    tensorStart_ = tensorStart;
    tensorEnd_ = tensorEnd;
    paramsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(params));
    gradsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(grads));
    expAvgsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(exp_avgs));
    expAvgSqsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(exp_avg_sqs));
    stateStepsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(state_steps));
    paramsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(params_ref));
    expAvgsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(exp_avgs_ref));
    expAvgSqsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(exp_avg_sqs_ref));
    if (this->amsgrad) {
        maxExpAvgSqsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(max_exp_avg_sqs));
        maxExpAvgSqsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(max_exp_avg_sqs_ref));
    }

    tensorCount_ = this->amsgrad ? TENSOR_COUNT_AMSGRAD : TENSOR_COUNT_NO_AMSGRAD;
    tensorCountOut_ = this->amsgrad ? TENSOR_COUNT_OUT_AMSGRAD : TENSOR_COUNT_OUT_NO_AMSGRAD;

    if constexpr (isFloat) {
        // FP32: 直接使用float类型，无需额外的cast区域
        pipe_->InitBuffer(inQue, BUFFER_NUM, this->coreCalcMax * sizeof(float) * tensorCount_);
        pipe_->InitBuffer(outQue, BUFFER_NUM, this->coreCalcMax * sizeof(float) * tensorCountOut_);
        pipe_->InitBuffer(tempBuf_, 32);
        pipe_->InitBuffer(tempBuf2_, 32);
    } else {
        // FP16/BF16: 原始类型 + FP32 cast区域
        int32_t tensorCountC = tensorCount_ - 1; // 不含step
        pipe_->InitBuffer(inQue, BUFFER_NUM,
                          this->coreCalcMax * (sizeof(T) * tensorCount_ + sizeof(float) * tensorCountC));
        pipe_->InitBuffer(outQue, BUFFER_NUM, this->coreCalcMax * sizeof(float) * tensorCountOut_);
        pipe_->InitBuffer(tempBuf_, 32);
        pipe_->InitBuffer(tempBuf2_, 32);
    }

    paramsOffset = this->coreCalcMax * INDEX_PARAMS;
    gradsOffset = this->coreCalcMax * INDEX_GRADS;
    expAvgOffset = this->coreCalcMax * INDEX_EXP_AVG;
    expAvgSqOffset = this->coreCalcMax * INDEX_EXP_AVG_SQ;
    maxExpAvgSqOffset = this->coreCalcMax * INDEX_MAX_EXP_AVG_SQ;

    paramsOutOffset = this->coreCalcMax * 0;
    expAvgOutOffset = this->coreCalcMax * 1;
    expAvgSqOutOffset = this->coreCalcMax * 2;
    maxExpAvgSqOutOffset = this->coreCalcMax * 3;

    hasGradScale = 0;
    if (this->useGradScale) {
        gmGradScale.SetGlobalBuffer((__gm__ float*)grad_scale, 1);
        gradScaleValue = static_cast<float>(gmGradScale.GetValue(0));
        hasGradScale = 1;
    }
    if (this->useFoundInf) {
        gmFoundInf.SetGlobalBuffer((__gm__ float*)found_inf, 1);
    }
}

// ==================== Compute dispatch ====================

template <typename T>
__aicore__ inline void FusedAdamWKernel<T>::Compute(const uint64_t index, const uint64_t dataCount)
{
    if constexpr (isFloat) {
        ComputeFP32(index, dataCount);
    } else {
        ComputeFP16Bf16(index, dataCount);
    }
}

// ==================== ComputeFP32 ====================

template <typename T>
__aicore__ inline void FusedAdamWKernel<T>::ComputeFP32(const uint64_t index, const uint64_t dataCount)
{
    if (this->useFoundInf) {
        float foundInfValue = static_cast<float>(gmFoundInf.GetValue(0));
        if (foundInfValue == 1.0f) {
            return;
        }
    }
    uint64_t offset = index * this->coreCalcMax;
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(float)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};

    LocalTensor<float> inLocal = inQue.AllocTensor<float>();
    LocalTensor<float> outLocal = outQue.AllocTensor<float>();

    // 从GM拷贝数据到UB
    PipeSync<AscendC::HardEvent::MTE3_MTE2>();
    PipeSync<AscendC::HardEvent::S_MTE2>();
    PipeSync<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(inLocal[paramsOffset], gmParams[offset], copyParams, padParams);
    DataCopyPad(inLocal[gradsOffset], gmGrads[offset], copyParams, padParams);
    DataCopyPad(inLocal[expAvgOffset], gmExpAvg[offset], copyParams, padParams);
    DataCopyPad(inLocal[expAvgSqOffset], gmExpAvgSq[offset], copyParams, padParams);
    if (this->amsgrad) {
        DataCopyPad(inLocal[maxExpAvgSqOffset], gmMaxExpAvgSq[offset], copyParams, padParams);
    }
    PipeSync<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();

    // Step 1: 梯度缩放
    if (hasGradScale) {
        float invGradScale = 1.0f / gradScaleValue;
        Muls(inLocal[gradsOffset], inLocal[gradsOffset], invGradScale, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 2: 最大化处理
    if (this->maximize) {
        Muls(inLocal[gradsOffset], inLocal[gradsOffset], -1.0f, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 3: AdamW weight decay
    if (this->weightDecay != 0.0f) {
        float decayFactor = 1.0f - this->lr * this->weightDecay;
        Muls(inLocal[paramsOffset], inLocal[paramsOffset], decayFactor, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 4: 更新一阶动量
    Muls(outLocal[expAvgOutOffset], inLocal[gradsOffset], 1.0f - this->beta1, dataCount);
    PipeBarrier<PIPE_V>();
    Muls(inLocal[expAvgOffset], inLocal[expAvgOffset], this->beta1, dataCount);
    PipeBarrier<PIPE_V>();
    Add(inLocal[expAvgOffset], inLocal[expAvgOffset], outLocal[expAvgOutOffset], dataCount);
    PipeBarrier<PIPE_V>();

    // Step 5: 更新二阶动量
    Mul(outLocal[expAvgSqOutOffset], inLocal[gradsOffset], inLocal[gradsOffset], dataCount);
    PipeBarrier<PIPE_V>();
    Muls(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], 1.0f - this->beta2, dataCount);
    PipeBarrier<PIPE_V>();
    Muls(inLocal[expAvgSqOffset], inLocal[expAvgSqOffset], this->beta2, dataCount);
    PipeBarrier<PIPE_V>();
    Add(inLocal[expAvgSqOffset], inLocal[expAvgSqOffset], outLocal[expAvgSqOutOffset], dataCount);
    PipeBarrier<PIPE_V>();

    // Step 7: 分母 denom
    if (this->amsgrad) {
        Max(inLocal[maxExpAvgSqOffset], inLocal[maxExpAvgSqOffset], inLocal[expAvgSqOffset], dataCount);
        PipeBarrier<PIPE_V>();
        Sqrt(outLocal[expAvgSqOutOffset], inLocal[maxExpAvgSqOffset], dataCount);
        PipeBarrier<PIPE_V>();
        Muls(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], 1.0f / biasCorrection2Sqrt, dataCount);
        PipeBarrier<PIPE_V>();
        Adds(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], this->eps, dataCount);
        PipeBarrier<PIPE_V>();
    } else {
        Sqrt(outLocal[expAvgSqOutOffset], inLocal[expAvgSqOffset], dataCount);
        PipeBarrier<PIPE_V>();
        Muls(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], 1.0f / biasCorrection2Sqrt, dataCount);
        PipeBarrier<PIPE_V>();
        Adds(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], this->eps, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 8: param = param - step_size * exp_avg / denom
    Muls(outLocal[paramsOutOffset], inLocal[expAvgOffset], stepSize, dataCount);
    PipeBarrier<PIPE_V>();
    Div(outLocal[paramsOutOffset], outLocal[paramsOutOffset], outLocal[expAvgSqOutOffset], dataCount);
    PipeBarrier<PIPE_V>();
    Sub(inLocal[paramsOffset], inLocal[paramsOffset], outLocal[paramsOutOffset], dataCount);
    PipeBarrier<PIPE_V>();

    // 写回结果
    PipeSync<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(gmParamsOut[offset], inLocal[paramsOffset], copyParams);
    DataCopyPad(gmExpAvgOut[offset], inLocal[expAvgOffset], copyParams);
    DataCopyPad(gmExpAvgSqOut[offset], inLocal[expAvgSqOffset], copyParams);
    if (this->amsgrad) {
        DataCopyPad(gmMaxExpAvgSq[offset], inLocal[maxExpAvgSqOffset], copyParams);
    }

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

// ==================== ComputeFP16Bf16 ====================

template <typename T>
__aicore__ inline void FusedAdamWKernel<T>::ComputeFP16Bf16(const uint64_t index, const uint64_t dataCount)
{
    if (this->useFoundInf) {
        float foundInfValue = static_cast<float>(gmFoundInf.GetValue(0));
        if (foundInfValue == 1.0f) {
            return;
        }
    }

    uint64_t offset = index * this->coreCalcMax;
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};

    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<float> outLocal = outQue.AllocTensor<float>();

    // 从GM拷贝原始类型数据到UB
    PipeSync<AscendC::HardEvent::MTE3_MTE2>();
    PipeSync<AscendC::HardEvent::S_MTE2>();
    PipeSync<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(inLocal[paramsOffset], gmParams[offset], copyParams, padParams);
    DataCopyPad(inLocal[gradsOffset], gmGrads[offset], copyParams, padParams);
    DataCopyPad(inLocal[expAvgOffset], gmExpAvg[offset], copyParams, padParams);
    DataCopyPad(inLocal[expAvgSqOffset], gmExpAvgSq[offset], copyParams, padParams);
    if (this->amsgrad) {
        DataCopyPad(inLocal[maxExpAvgSqOffset], gmMaxExpAvgSq[offset], copyParams, padParams);
    }
    PipeSync<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();

    // Cast到FP32区域
    LocalTensor<float> inLocalC = inLocal[this->coreCalcMax * tensorCount_].template ReinterpretCast<float>();
    Cast(inLocalC[paramsOffset], inLocal[paramsOffset], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(inLocalC[gradsOffset], inLocal[gradsOffset], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(inLocalC[expAvgOffset], inLocal[expAvgOffset], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(inLocalC[expAvgSqOffset], inLocal[expAvgSqOffset], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    if (this->amsgrad) {
        Cast(inLocalC[maxExpAvgSqOffset], inLocal[maxExpAvgSqOffset], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 1: 梯度缩放
    if (hasGradScale) {
        float invGradScale = 1.0f / gradScaleValue;
        Muls(inLocalC[gradsOffset], inLocalC[gradsOffset], invGradScale, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 2: 最大化处理
    if (this->maximize) {
        Muls(inLocalC[gradsOffset], inLocalC[gradsOffset], -1.0f, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 3: AdamW weight decay
    if (this->weightDecay != 0.0f) {
        float decayFactor = 1.0f - this->lr * this->weightDecay;
        Muls(inLocalC[paramsOffset], inLocalC[paramsOffset], decayFactor, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 4: 更新一阶动量
    Muls(outLocal[expAvgOutOffset], inLocalC[gradsOffset], 1.0f - this->beta1, dataCount);
    PipeBarrier<PIPE_V>();
    Muls(inLocalC[expAvgOffset], inLocalC[expAvgOffset], this->beta1, dataCount);
    PipeBarrier<PIPE_V>();
    Add(inLocalC[expAvgOffset], inLocalC[expAvgOffset], outLocal[expAvgOutOffset], dataCount);
    PipeBarrier<PIPE_V>();

    // Step 5: 更新二阶动量
    Mul(outLocal[expAvgSqOutOffset], inLocalC[gradsOffset], inLocalC[gradsOffset], dataCount);
    PipeBarrier<PIPE_V>();
    Muls(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], 1.0f - this->beta2, dataCount);
    PipeBarrier<PIPE_V>();
    Muls(inLocalC[expAvgSqOffset], inLocalC[expAvgSqOffset], this->beta2, dataCount);
    PipeBarrier<PIPE_V>();
    Add(inLocalC[expAvgSqOffset], inLocalC[expAvgSqOffset], outLocal[expAvgSqOutOffset], dataCount);
    PipeBarrier<PIPE_V>();

    // Step 7: 分母 denom
    if (this->amsgrad) {
        Max(inLocalC[maxExpAvgSqOffset], inLocalC[maxExpAvgSqOffset], inLocalC[expAvgSqOffset], dataCount);
        PipeBarrier<PIPE_V>();
        Sqrt(outLocal[expAvgSqOutOffset], inLocalC[maxExpAvgSqOffset], dataCount);
        PipeBarrier<PIPE_V>();
        Muls(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], 1.0f / biasCorrection2Sqrt, dataCount);
        PipeBarrier<PIPE_V>();
        Adds(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], this->eps, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(inLocal[maxExpAvgSqOffset], inLocalC[maxExpAvgSqOffset], RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    } else {
        Sqrt(outLocal[expAvgSqOutOffset], inLocalC[expAvgSqOffset], dataCount);
        PipeBarrier<PIPE_V>();
        Muls(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], 1.0f / biasCorrection2Sqrt, dataCount);
        PipeBarrier<PIPE_V>();
        Adds(outLocal[expAvgSqOutOffset], outLocal[expAvgSqOutOffset], this->eps, dataCount);
        PipeBarrier<PIPE_V>();
    }

    // Step 8: param = param - step_size * exp_avg / denom
    Muls(outLocal[paramsOutOffset], inLocalC[expAvgOffset], stepSize, dataCount);
    PipeBarrier<PIPE_V>();
    Div(outLocal[paramsOutOffset], outLocal[paramsOutOffset], outLocal[expAvgSqOutOffset], dataCount);
    PipeBarrier<PIPE_V>();
    Sub(inLocalC[paramsOffset], inLocalC[paramsOffset], outLocal[paramsOutOffset], dataCount);
    PipeBarrier<PIPE_V>();

    // Cast回原始类型并写回
    Cast(inLocal[paramsOffset], inLocalC[paramsOffset], RoundMode::CAST_RINT, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(inLocal[expAvgOffset], inLocalC[expAvgOffset], RoundMode::CAST_RINT, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(inLocal[expAvgSqOffset], inLocalC[expAvgSqOffset], RoundMode::CAST_RINT, dataCount);
    PipeBarrier<PIPE_V>();

    PipeSync<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(gmParamsOut[offset], inLocal[paramsOffset], copyParams);
    DataCopyPad(gmExpAvgOut[offset], inLocal[expAvgOffset], copyParams);
    DataCopyPad(gmExpAvgSqOut[offset], inLocal[expAvgSqOffset], copyParams);
    if (this->amsgrad) {
        DataCopyPad(gmMaxExpAvgSq[offset], inLocal[maxExpAvgSqOffset], copyParams);
    }

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline float FusedAdamWKernel<T>::ScalarPow(float x, float y)
{
    LocalTensor<float> baseLocal = tempBuf_.Get<float>();
    LocalTensor<float> outLocal = tempBuf2_.Get<float>();
    PipeBarrier<PIPE_V>();
    Duplicate(baseLocal, x, BLOCK_SIZE_FOR_FLOAT32);
    PipeBarrier<PIPE_V>();
    Power<float, false>(outLocal, baseLocal, y, BLOCK_SIZE_FOR_FLOAT32);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float result = outLocal.GetValue(0);
    return result;
}

// ==================== Process (shared) ====================

template <typename T>
__aicore__ inline void FusedAdamWKernel<T>::Process()
{
    for (uint64_t idx = tensorStart_; idx < tensorEnd_; idx++) {
        uint64_t buf[10];
        desc_.SetShapeAddr(&buf[0]);
        paramsList_.GetDesc(desc_, static_cast<uint32_t>(idx));

        uint64_t tensorDataNum = 1;
        for (uint32_t j = 0; j < desc_.GetDim(); j++) {
            tensorDataNum *= desc_.GetShape(j);
        }
        if (tensorDataNum == 0) {
            continue;
        }
        gmParams.SetGlobalBuffer(paramsList_.GetDataPtr<T>(idx), tensorDataNum);
        gmGrads.SetGlobalBuffer(gradsList_.GetDataPtr<T>(idx), tensorDataNum);
        gmExpAvg.SetGlobalBuffer(expAvgsList_.GetDataPtr<T>(idx), tensorDataNum);
        gmExpAvgSq.SetGlobalBuffer(expAvgSqsList_.GetDataPtr<T>(idx), tensorDataNum);
        gmParamsOut.SetGlobalBuffer(paramsOutList_.GetDataPtr<T>(idx), tensorDataNum);
        gmExpAvgOut.SetGlobalBuffer(expAvgsOutList_.GetDataPtr<T>(idx), tensorDataNum);
        gmExpAvgSqOut.SetGlobalBuffer(expAvgSqsOutList_.GetDataPtr<T>(idx), tensorDataNum);
        if (this->amsgrad) {
            gmMaxExpAvgSq.SetGlobalBuffer(maxExpAvgSqsList_.GetDataPtr<T>(idx), tensorDataNum);
            gmMaxExpAvgSqOut.SetGlobalBuffer(maxExpAvgSqsOutList_.GetDataPtr<T>(idx), tensorDataNum);
        }

        gmStateSteps.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(stateStepsList_.GetDataPtr<float>(idx)), 1);
        this->stepCount = static_cast<float>(gmStateSteps.GetValue(0));
        this->stepCount += 1;
        float biasCorrection1 = 1.0f - ScalarPow(this->beta1, this->stepCount);
        float biasCorrection2 = 1.0f - ScalarPow(this->beta2, this->stepCount);
        stepSize = static_cast<float>(this->lr / biasCorrection1);
        biasCorrection2Sqrt = static_cast<float>(sqrt(biasCorrection2));

        uint64_t loopNum = (tensorDataNum + this->coreCalcMax - 1) / this->coreCalcMax;
        for (uint64_t n = 0; n < loopNum - 1; n++) {
            Compute(n, this->coreCalcMax);
        }
        uint64_t lastCount = tensorDataNum - this->coreCalcMax * (loopNum - 1);
        Compute(loopNum - 1, lastCount);
    }
}

} // namespace FusedAdamW

#endif // _FUSED_ADAMW_KERNEL_H_
