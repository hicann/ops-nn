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
 * \file fused_adam_regbase.h
 * \brief unified fused_adam regbase kernel for FP32/FP16/BF16
 */

#ifndef _FUSED_ADAM_KERNEL_REGBASE_H_
#define _FUSED_ADAM_KERNEL_REGBASE_H_

#include "kernel_operator.h"
#include "fused_adam_utils.h"
#include "fused_adam_tiling_data.h"

namespace FusedAdam {
using namespace AscendC;

constexpr int32_t BYTE_ONE_BLOCK = 32;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t INDEX_PARAMS = 0;
constexpr int32_t INDEX_GRADS = 1;
constexpr int32_t INDEX_EXP_AVG = 2;
constexpr int32_t INDEX_EXP_AVG_SQ = 3;
constexpr int32_t INDEX_MAX_EXP_AVG_SQ = 4;
constexpr int32_t INDEX_STEP = 5;
constexpr int32_t BLOCK_SIZE_FOR_FLOAT32 = 8;

constexpr uint64_t SINGLE_BUFFER_COUNT = 12 * 1024 / sizeof(float);

template <typename T, bool amsgrad_>
class FusedAdamKernelRegBase {
    static constexpr bool isFloat = std::is_same_v<T, float>;

public:
    __aicore__ inline FusedAdamKernelRegBase(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void InitSingleTensorParam(uint32_t idx);
    __aicore__ inline void Init(GM_ADDR params, GM_ADDR grads, GM_ADDR exp_avgs, GM_ADDR exp_avg_sqs,
                                GM_ADDR max_exp_avg_sqs, GM_ADDR state_steps, GM_ADDR grad_scale, GM_ADDR found_inf,
                                GM_ADDR params_ref, GM_ADDR grads_ref, GM_ADDR exp_avgs_ref, GM_ADDR exp_avg_sqs_ref,
                                GM_ADDR max_exp_avg_sqs_ref, const FusedAdamTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline float ScalarPow(float x, float y);
    __aicore__ inline void InitCoreData(const FusedAdamTilingData& tiling);
    __aicore__ inline void ProcessSingle(uint32_t tensorIndex, uint64_t tensorStartOffset, uint64_t tensorEndOffset);
    __aicore__ inline void ComputeFP32(uint64_t offset, uint64_t dataCount, bool pingpongflag, uint32_t oneRepeatSize);
    __aicore__ inline void ComputeFP16BF16(uint64_t offset, uint64_t dataCount, bool pingpongflag,
                                           uint32_t oneRepeatSize);

private:
    // param from tiling
    float lr_{0.0f};
    float beta1_{0.0f};
    float beta2_{0.0f};
    float weightDecay_{0.0f};
    float eps_{0.0f};
    uint64_t maximize_{0};
    uint64_t useGradScale_{0};
    uint64_t useFoundInf_{0};
    uint64_t usedCoreNum_{0};
    const uint64_t* tensorDataCountList_ = nullptr;
    const uint32_t* tensorStartList_ = nullptr;
    const uint32_t* tensorEndList_ = nullptr;
    const uint64_t* tensorStartOffsetList_ = nullptr;
    const uint64_t* tensorEndOffsetList_ = nullptr;
    // group2
    float stepCount_{0.0f};
    float invGradScaleValue_{1.0f};
    float stepSize_{0.0f};
    float biasCorrection2Sqrt_{1.0f};
    // GlobalTensor Group
    GlobalTensor<T> gmParams_;
    GlobalTensor<T> gmGrads_;
    GlobalTensor<T> gmExpAvg_;
    GlobalTensor<T> gmExpAvgSqs_;
    GlobalTensor<T> gmMaxExpAvgSqs_;
    GlobalTensor<float> gmStateSteps_;
    GlobalTensor<float> gmGradScale_;
    GlobalTensor<float> gmFoundInf_;
    GlobalTensor<T> gmParamsOut_;
    GlobalTensor<T> gmGradsOut_;
    GlobalTensor<T> gmExpAvgOut_;
    GlobalTensor<T> gmExpAvgSqsOut_;
    GlobalTensor<T> gmMaxExpAvgSqsOut_;
    // ListTensorDesc Group
    ListTensorDesc paramsList_;
    ListTensorDesc gradsList_;
    ListTensorDesc expAvgsList_;
    ListTensorDesc expAvgSqsList_;
    ListTensorDesc maxExpAvgSqsList_;
    ListTensorDesc stateStepsList_;
    ListTensorDesc paramsOutList_;
    ListTensorDesc gradsOutList_;
    ListTensorDesc expAvgsOutList_;
    ListTensorDesc expAvgSqsOutList_;
    ListTensorDesc maxExpAvgSqsOutList_;

    TPipe* pipe_;
    // buf group normal
    TBuf<QuePosition::VECCALC> paramBuf1_;
    TBuf<QuePosition::VECCALC> paramBuf2_;
    TBuf<QuePosition::VECCALC> gradBuf1_;
    TBuf<QuePosition::VECCALC> gradBuf2_;
    TBuf<QuePosition::VECCALC> mBuf1_;
    TBuf<QuePosition::VECCALC> mBuf2_;
    TBuf<QuePosition::VECCALC> vBuf1_;
    TBuf<QuePosition::VECCALC> vBuf2_;
    TBuf<QuePosition::VECCALC> maxvBuf1_;
    TBuf<QuePosition::VECCALC> maxvBuf2_;
    TBuf<QuePosition::VECCALC> grad2Buf1_;
    TBuf<QuePosition::VECCALC> grad2Buf2_;
    TBuf<QuePosition::VECCALC> powBuf1_;
    TBuf<QuePosition::VECCALC> powBuf2_;
    // buf group special for b16 dtype
    TBuf<QuePosition::VECCALC> paramB16Buf1_;
    TBuf<QuePosition::VECCALC> paramB16Buf2_;
    TBuf<QuePosition::VECCALC> gradB16Buf1_;
    TBuf<QuePosition::VECCALC> gradB16Buf2_;
    TBuf<QuePosition::VECCALC> mB16Buf1_;
    TBuf<QuePosition::VECCALC> mB16Buf2_;
    TBuf<QuePosition::VECCALC> vB16Buf1_;
    TBuf<QuePosition::VECCALC> vB16Buf2_;
    TBuf<QuePosition::VECCALC> maxvB16Buf1_;
    TBuf<QuePosition::VECCALC> maxvB16Buf2_;
};

template <typename T, bool amsgrad_>
__aicore__ inline void FusedAdamKernelRegBase<T, amsgrad_>::Init(
    GM_ADDR params, GM_ADDR grads, GM_ADDR exp_avgs, GM_ADDR exp_avg_sqs, GM_ADDR max_exp_avg_sqs, GM_ADDR state_steps,
    GM_ADDR grad_scale, GM_ADDR found_inf, GM_ADDR params_ref, GM_ADDR grads_ref, GM_ADDR exp_avgs_ref,
    GM_ADDR exp_avg_sqs_ref, GM_ADDR max_exp_avg_sqs_ref, const FusedAdamTilingData& tilingData)
{
    InitCoreData(tilingData);
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }
    paramsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(params));
    gradsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(grads));
    expAvgsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(exp_avgs));
    expAvgSqsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(exp_avg_sqs));
    stateStepsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(state_steps));
    paramsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(params_ref));
    gradsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(grads_ref));
    expAvgsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(exp_avgs_ref));
    expAvgSqsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(exp_avg_sqs_ref));
    if constexpr (amsgrad_) {
        maxExpAvgSqsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(max_exp_avg_sqs));
        maxExpAvgSqsOutList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(max_exp_avg_sqs_ref));
    }
    if (useGradScale_) {
        gmGradScale_.SetGlobalBuffer((__gm__ float*)grad_scale, 1);
        invGradScaleValue_ = 1.0f / static_cast<float>(gmGradScale_.GetValue(0));
    }
    if (useFoundInf_) {
        gmFoundInf_.SetGlobalBuffer((__gm__ float*)found_inf, 1);
    }
    // FP32: 直接使用float类型，无需额外的cast区域
    pipe_->InitBuffer(paramBuf1_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(paramBuf2_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(gradBuf1_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(gradBuf2_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(grad2Buf1_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(grad2Buf2_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(mBuf1_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(mBuf2_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(vBuf1_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(vBuf2_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(maxvBuf1_, SINGLE_BUFFER_COUNT * sizeof(float));
    pipe_->InitBuffer(maxvBuf2_, SINGLE_BUFFER_COUNT * sizeof(float));
    if constexpr (!isFloat) {
        // FP32: 直接使用float类型，无需额外的cast区域
        pipe_->InitBuffer(paramB16Buf1_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(paramB16Buf2_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(gradB16Buf1_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(gradB16Buf2_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(mB16Buf1_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(mB16Buf2_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(vB16Buf1_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(vB16Buf2_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(maxvB16Buf1_, SINGLE_BUFFER_COUNT * sizeof(T));
        pipe_->InitBuffer(maxvB16Buf2_, SINGLE_BUFFER_COUNT * sizeof(T));
    }
    pipe_->InitBuffer(powBuf1_, 32);
    pipe_->InitBuffer(powBuf2_, 32);
}

template <typename T, bool amsgrad_>
__aicore__ inline void FusedAdamKernelRegBase<T, amsgrad_>::InitCoreData(const FusedAdamTilingData& tilingData)
{
    lr_ = tilingData.lr;
    beta1_ = tilingData.beta1;
    beta2_ = tilingData.beta2;
    weightDecay_ = tilingData.weightDecay;
    eps_ = tilingData.eps;
    maximize_ = tilingData.maximize;
    useGradScale_ = tilingData.useGradScale;
    useFoundInf_ = tilingData.useFoundInf;
    usedCoreNum_ = tilingData.usedCoreNum;
    tensorDataCountList_ = tilingData.tensorDataCountList_;
    tensorStartList_ = tilingData.tensorStartList_;
    tensorEndList_ = tilingData.tensorEndList_;
    tensorStartOffsetList_ = tilingData.tensorStartOffsetList_;
    tensorEndOffsetList_ = tilingData.tensorEndOffsetList_;
}

template <typename T, bool amsgrad_>
__aicore__ inline void FusedAdamKernelRegBase<T, amsgrad_>::InitSingleTensorParam(uint32_t idx)
{
    uint64_t count = tensorDataCountList_[idx];
    gmParams_.SetGlobalBuffer(paramsList_.GetDataPtr<T>(idx), count);
    gmGrads_.SetGlobalBuffer(gradsList_.GetDataPtr<T>(idx), count);
    gmExpAvg_.SetGlobalBuffer(expAvgsList_.GetDataPtr<T>(idx), count);
    gmExpAvgSqs_.SetGlobalBuffer(expAvgSqsList_.GetDataPtr<T>(idx), count);
    gmParamsOut_.SetGlobalBuffer(paramsOutList_.GetDataPtr<T>(idx), count);
    gmGradsOut_.SetGlobalBuffer(gradsOutList_.GetDataPtr<T>(idx), count);
    gmExpAvgOut_.SetGlobalBuffer(expAvgsOutList_.GetDataPtr<T>(idx), count);
    gmExpAvgSqsOut_.SetGlobalBuffer(expAvgSqsOutList_.GetDataPtr<T>(idx), count);
    if constexpr (amsgrad_) {
        gmMaxExpAvgSqs_.SetGlobalBuffer(maxExpAvgSqsList_.GetDataPtr<T>(idx), count);
        gmMaxExpAvgSqsOut_.SetGlobalBuffer(maxExpAvgSqsOutList_.GetDataPtr<T>(idx), count);
    }

    gmStateSteps_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(stateStepsList_.GetDataPtr<float>(idx)), 1);
    stepCount_ = static_cast<float>(gmStateSteps_.GetValue(0));
    stepCount_ += 1;
    float biasCorrection1 = 1.0f - ScalarPow(beta1_, stepCount_);
    float biasCorrection2 = 1.0f - ScalarPow(beta2_, stepCount_);
    stepSize_ = static_cast<float>(lr_ / biasCorrection1);
    biasCorrection2Sqrt_ = static_cast<float>(sqrt(biasCorrection2));
}

template <typename T, bool amsgrad_>
__aicore__ inline float FusedAdamKernelRegBase<T, amsgrad_>::ScalarPow(float x, float y)
{
    LocalTensor<float> baseLocal = powBuf1_.Get<float>();
    LocalTensor<float> outLocal = powBuf2_.Get<float>();
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

template <typename T, bool amsgrad_>
__aicore__ inline void FusedAdamKernelRegBase<T, amsgrad_>::Process()
{
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }
    if (this->useFoundInf_) {
        float foundInfValue = static_cast<float>(gmFoundInf_.GetValue(0));
        if (foundInfValue == 1.0f) {
            return;
        }
    }
    uint32_t tensorStart = tensorStartList_[GetBlockIdx()];
    uint32_t tensorEnd = tensorEndList_[GetBlockIdx()];
    uint64_t tensorStartOffset = tensorStartOffsetList_[GetBlockIdx()];
    uint64_t tensorEndOffset = tensorEndOffsetList_[GetBlockIdx()];
    for (uint32_t idx = tensorStart; idx <= tensorEnd; idx++) {
        uint32_t start = 0;
        uint32_t end = tensorDataCountList_[idx];
        if (idx == tensorStart) {
            start = tensorStartOffset;
        }
        if (idx == tensorEnd) {
            end = tensorEndOffset;
        }
        ProcessSingle(idx, start, end);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, bool amsgrad_>
__aicore__ inline void FusedAdamKernelRegBase<T, amsgrad_>::ProcessSingle(uint32_t idx, uint64_t start, uint64_t end)
{
    // Init Single Tensor HyperData
    InitSingleTensorParam(idx);
    uint64_t count = end - start;
    if (count == 0) {
        return;
    }
    uint64_t loopNum = (count + SINGLE_BUFFER_COUNT - 1) / SINGLE_BUFFER_COUNT;
    uint64_t lastLoopCount = count - (loopNum - 1) * SINGLE_BUFFER_COUNT;
    // param
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    // grad
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    // m v maxv
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
    bool pingpongflag = 0;
    uint32_t oneRepeatSize = GetVecLen() / sizeof(float);
    uint64_t offset = start;
    for (uint64_t i = 0; i < loopNum - 1; i++) {
        if constexpr (isFloat) {
            ComputeFP32(offset, SINGLE_BUFFER_COUNT, pingpongflag, oneRepeatSize);
        } else {
            ComputeFP16BF16(offset, SINGLE_BUFFER_COUNT, pingpongflag, oneRepeatSize);
        }
        pingpongflag = !pingpongflag;
        offset += SINGLE_BUFFER_COUNT;
    }
    // last loop
    if constexpr (isFloat) {
        ComputeFP32(offset, lastLoopCount, pingpongflag, oneRepeatSize);
    } else {
        ComputeFP16BF16(offset, lastLoopCount, pingpongflag, oneRepeatSize);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
}

template <typename T, bool amsgrad_>
__aicore__ inline void FusedAdamKernelRegBase<T, amsgrad_>::ComputeFP32(uint64_t offset, uint64_t dataCount,
                                                                        bool pingpongflag, uint32_t oneRepeatSize)
{
    uint16_t repeatTimes = CeilDivision(static_cast<uint32_t>(dataCount), oneRepeatSize);
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(float)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};

    // init
    auto paramSyncID = pingpongflag ? EVENT_ID0 : EVENT_ID1;
    auto gradSyncID = pingpongflag ? EVENT_ID2 : EVENT_ID3;
    auto mvmaxvSyncID = pingpongflag ? EVENT_ID4 : EVENT_ID5;
    LocalTensor<float> paramBuf = pingpongflag ? paramBuf1_.Get<float>() : paramBuf2_.Get<float>();
    LocalTensor<float> gradBuf = pingpongflag ? gradBuf1_.Get<float>() : gradBuf2_.Get<float>();
    LocalTensor<float> grad2Buf = pingpongflag ? grad2Buf1_.Get<float>() : grad2Buf2_.Get<float>();
    LocalTensor<float> mBuf = pingpongflag ? mBuf1_.Get<float>() : mBuf2_.Get<float>();
    LocalTensor<float> vBuf = pingpongflag ? vBuf1_.Get<float>() : vBuf2_.Get<float>();
    LocalTensor<float> maxvBuf = pingpongflag ? maxvBuf1_.Get<float>() : maxvBuf2_.Get<float>();

    // COMPUTE
    WaitFlag<HardEvent::MTE3_MTE2>(gradSyncID);
    DataCopyPad(gradBuf, gmGrads_[offset], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(gradSyncID);

    WaitFlag<HardEvent::MTE2_V>(gradSyncID);
    if (useGradScale_) {
        Muls(gradBuf, gradBuf, invGradScaleValue_, dataCount);
    }

    WaitFlag<HardEvent::MTE3_MTE2>(paramSyncID);
    DataCopyPad(paramBuf, gmParams_[offset], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(paramSyncID);

    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(paramSyncID);
    if (maximize_) {
        asc_vf_call<MaximizeAndWeightDecayFP32>(
            (__ubuf__ float*)grad2Buf.GetPhyAddr(), (__ubuf__ float*)paramBuf.GetPhyAddr(),
            (__ubuf__ float*)gradBuf.GetPhyAddr(), weightDecay_, dataCount, oneRepeatSize, repeatTimes);
    } else {
        asc_vf_call<WeightDecayFP32>((__ubuf__ float*)grad2Buf.GetPhyAddr(), (__ubuf__ float*)paramBuf.GetPhyAddr(),
                                     (__ubuf__ float*)gradBuf.GetPhyAddr(), weightDecay_, dataCount, oneRepeatSize,
                                     repeatTimes);
    }
    SetFlag<HardEvent::V_MTE3>(gradSyncID);

    WaitFlag<HardEvent::V_MTE3>(gradSyncID);
    if (useGradScale_) {
        DataCopyPad(gmGradsOut_[offset], gradBuf, copyParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(gradSyncID);

    WaitFlag<HardEvent::MTE3_MTE2>(mvmaxvSyncID);
    DataCopyPad(mBuf, gmExpAvg_[offset], copyParams, padParams);
    DataCopyPad(vBuf, gmExpAvgSqs_[offset], copyParams, padParams);
    if constexpr (amsgrad_) {
        DataCopyPad(maxvBuf, gmMaxExpAvgSqs_[offset], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(mvmaxvSyncID);

    WaitFlag<HardEvent::MTE2_V>(mvmaxvSyncID);
    if constexpr (amsgrad_) {
        asc_vf_call<UpdateMVMaxVFP32>((__ubuf__ float*)mBuf.GetPhyAddr(), (__ubuf__ float*)vBuf.GetPhyAddr(),
                                      (__ubuf__ float*)maxvBuf.GetPhyAddr(), (__ubuf__ float*)grad2Buf.GetPhyAddr(),
                                      beta1_, beta2_, 1.0f - beta1_, 1.0f - beta2_, dataCount, oneRepeatSize,
                                      repeatTimes);
    } else {
        asc_vf_call<UpdateMVFP32>((__ubuf__ float*)mBuf.GetPhyAddr(), (__ubuf__ float*)vBuf.GetPhyAddr(),
                                  (__ubuf__ float*)grad2Buf.GetPhyAddr(), beta1_, beta2_, 1.0f - beta1_, 1.0f - beta2_,
                                  dataCount, oneRepeatSize, repeatTimes);
    }

    PipeBarrier<PIPE_V>();

    if constexpr (amsgrad_) {
        asc_vf_call<UpdateParamFP32>((__ubuf__ float*)paramBuf.GetPhyAddr(), (__ubuf__ float*)mBuf.GetPhyAddr(),
                                     (__ubuf__ float*)maxvBuf.GetPhyAddr(), -stepSize_, biasCorrection2Sqrt_, eps_,
                                     dataCount, oneRepeatSize, repeatTimes);
    } else {
        asc_vf_call<UpdateParamFP32>((__ubuf__ float*)paramBuf.GetPhyAddr(), (__ubuf__ float*)mBuf.GetPhyAddr(),
                                     (__ubuf__ float*)vBuf.GetPhyAddr(), -stepSize_, biasCorrection2Sqrt_, eps_,
                                     dataCount, oneRepeatSize, repeatTimes);
    }
    SetFlag<HardEvent::V_MTE3>(mvmaxvSyncID);
    SetFlag<HardEvent::V_MTE3>(paramSyncID);

    WaitFlag<HardEvent::V_MTE3>(mvmaxvSyncID);
    DataCopyPad(gmExpAvgOut_[offset], mBuf, copyParams);
    DataCopyPad(gmExpAvgSqsOut_[offset], vBuf, copyParams);
    if constexpr (amsgrad_) {
        DataCopyPad(gmMaxExpAvgSqsOut_[offset], maxvBuf, copyParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(mvmaxvSyncID);

    WaitFlag<HardEvent::V_MTE3>(paramSyncID);
    DataCopyPad(gmParamsOut_[offset], paramBuf, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(paramSyncID);
}

template <typename T, bool amsgrad_>
__aicore__ inline void FusedAdamKernelRegBase<T, amsgrad_>::ComputeFP16BF16(uint64_t offset, uint64_t dataCount,
                                                                            bool pingpongflag, uint32_t oneRepeatSize)
{
    uint16_t repeatTimes = CeilDivision(static_cast<uint32_t>(dataCount), oneRepeatSize);
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};

    // init fp16
    LocalTensor<T> paramB16Buf = pingpongflag ? paramB16Buf1_.Get<T>() : paramB16Buf2_.Get<T>();
    LocalTensor<T> gradB16Buf = pingpongflag ? gradB16Buf1_.Get<T>() : gradB16Buf2_.Get<T>();
    LocalTensor<T> mB16Buf = pingpongflag ? mB16Buf1_.Get<T>() : mB16Buf2_.Get<T>();
    LocalTensor<T> vB16Buf = pingpongflag ? vB16Buf1_.Get<T>() : vB16Buf2_.Get<T>();
    LocalTensor<T> maxvB16Buf = pingpongflag ? maxvB16Buf1_.Get<T>() : maxvB16Buf2_.Get<T>();
    // init fp32
    auto paramSyncID = pingpongflag ? EVENT_ID0 : EVENT_ID1;
    auto gradSyncID = pingpongflag ? EVENT_ID2 : EVENT_ID3;
    auto mvmaxvSyncID = pingpongflag ? EVENT_ID4 : EVENT_ID5;
    LocalTensor<float> paramBuf = pingpongflag ? paramBuf1_.Get<float>() : paramBuf2_.Get<float>();
    LocalTensor<float> gradBuf = pingpongflag ? gradBuf1_.Get<float>() : gradBuf2_.Get<float>();
    LocalTensor<float> grad2Buf = pingpongflag ? grad2Buf1_.Get<float>() : grad2Buf2_.Get<float>();
    LocalTensor<float> mBuf = pingpongflag ? mBuf1_.Get<float>() : mBuf2_.Get<float>();
    LocalTensor<float> vBuf = pingpongflag ? vBuf1_.Get<float>() : vBuf2_.Get<float>();
    LocalTensor<float> maxvBuf = pingpongflag ? maxvBuf1_.Get<float>() : maxvBuf2_.Get<float>();

    // COMPUTE
    WaitFlag<HardEvent::MTE3_MTE2>(gradSyncID);
    DataCopyPad(gradB16Buf, gmGrads_[offset], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(gradSyncID);

    WaitFlag<HardEvent::MTE2_V>(gradSyncID);
    Cast(gradBuf, gradB16Buf, RoundMode::CAST_NONE, dataCount);
    if (useGradScale_) {
        Muls(gradBuf, gradBuf, invGradScaleValue_, dataCount);
    }
    PipeBarrier<PIPE_V>();

    WaitFlag<HardEvent::MTE3_MTE2>(paramSyncID);
    DataCopyPad(paramB16Buf, gmParams_[offset], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(paramSyncID);

    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(paramSyncID);
    Cast(paramBuf, paramB16Buf, RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    if (maximize_) {
        asc_vf_call<MaximizeAndWeightDecayFP32>(
            (__ubuf__ float*)grad2Buf.GetPhyAddr(), (__ubuf__ float*)paramBuf.GetPhyAddr(),
            (__ubuf__ float*)gradBuf.GetPhyAddr(), weightDecay_, dataCount, oneRepeatSize, repeatTimes);
    } else {
        asc_vf_call<WeightDecayFP32>((__ubuf__ float*)grad2Buf.GetPhyAddr(), (__ubuf__ float*)paramBuf.GetPhyAddr(),
                                     (__ubuf__ float*)gradBuf.GetPhyAddr(), weightDecay_, dataCount, oneRepeatSize,
                                     repeatTimes);
    }
    PipeBarrier<PIPE_V>();
    Cast(gradB16Buf, gradBuf, RoundMode::CAST_RINT, dataCount);
    SetFlag<HardEvent::V_MTE3>(gradSyncID);

    WaitFlag<HardEvent::V_MTE3>(gradSyncID);
    if (useGradScale_) {
        DataCopyPad(gmGradsOut_[offset], gradB16Buf, copyParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(gradSyncID);

    WaitFlag<HardEvent::MTE3_MTE2>(mvmaxvSyncID);
    DataCopyPad(mB16Buf, gmExpAvg_[offset], copyParams, padParams);
    DataCopyPad(vB16Buf, gmExpAvgSqs_[offset], copyParams, padParams);
    if constexpr (amsgrad_) {
        DataCopyPad(maxvB16Buf, gmMaxExpAvgSqs_[offset], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(mvmaxvSyncID);

    WaitFlag<HardEvent::MTE2_V>(mvmaxvSyncID);
    Cast(mBuf, mB16Buf, RoundMode::CAST_NONE, dataCount);
    Cast(vBuf, vB16Buf, RoundMode::CAST_NONE, dataCount);
    if constexpr (amsgrad_) {
        Cast(maxvBuf, maxvB16Buf, RoundMode::CAST_NONE, dataCount);
    }
    PipeBarrier<PIPE_V>();
    if constexpr (amsgrad_) {
        asc_vf_call<UpdateMVMaxVFP32>((__ubuf__ float*)mBuf.GetPhyAddr(), (__ubuf__ float*)vBuf.GetPhyAddr(),
                                      (__ubuf__ float*)maxvBuf.GetPhyAddr(), (__ubuf__ float*)grad2Buf.GetPhyAddr(),
                                      beta1_, beta2_, 1.0f - beta1_, 1.0f - beta2_, dataCount, oneRepeatSize,
                                      repeatTimes);
    } else {
        asc_vf_call<UpdateMVFP32>((__ubuf__ float*)mBuf.GetPhyAddr(), (__ubuf__ float*)vBuf.GetPhyAddr(),
                                  (__ubuf__ float*)grad2Buf.GetPhyAddr(), beta1_, beta2_, 1.0f - beta1_, 1.0f - beta2_,
                                  dataCount, oneRepeatSize, repeatTimes);
    }

    PipeBarrier<PIPE_V>();

    if constexpr (amsgrad_) {
        asc_vf_call<UpdateParamFP32>((__ubuf__ float*)paramBuf.GetPhyAddr(), (__ubuf__ float*)mBuf.GetPhyAddr(),
                                     (__ubuf__ float*)maxvBuf.GetPhyAddr(), -stepSize_, biasCorrection2Sqrt_, eps_,
                                     dataCount, oneRepeatSize, repeatTimes);
    } else {
        asc_vf_call<UpdateParamFP32>((__ubuf__ float*)paramBuf.GetPhyAddr(), (__ubuf__ float*)mBuf.GetPhyAddr(),
                                     (__ubuf__ float*)vBuf.GetPhyAddr(), -stepSize_, biasCorrection2Sqrt_, eps_,
                                     dataCount, oneRepeatSize, repeatTimes);
    }
    PipeBarrier<PIPE_V>();
    Cast(mB16Buf, mBuf, RoundMode::CAST_RINT, dataCount);
    Cast(vB16Buf, vBuf, RoundMode::CAST_RINT, dataCount);
    if constexpr (amsgrad_) {
        Cast(maxvB16Buf, maxvBuf, RoundMode::CAST_RINT, dataCount);
    }
    Cast(paramB16Buf, paramBuf, RoundMode::CAST_RINT, dataCount);
    SetFlag<HardEvent::V_MTE3>(mvmaxvSyncID);
    SetFlag<HardEvent::V_MTE3>(paramSyncID);

    WaitFlag<HardEvent::V_MTE3>(mvmaxvSyncID);
    DataCopyPad(gmExpAvgOut_[offset], mB16Buf, copyParams);
    DataCopyPad(gmExpAvgSqsOut_[offset], vB16Buf, copyParams);
    if constexpr (amsgrad_) {
        DataCopyPad(gmMaxExpAvgSqsOut_[offset], maxvB16Buf, copyParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(mvmaxvSyncID);

    WaitFlag<HardEvent::V_MTE3>(paramSyncID);
    DataCopyPad(gmParamsOut_[offset], paramB16Buf, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(paramSyncID);
}

} // namespace FusedAdam
#endif // _FUSED_ADAM_KERNEL_REGBASE_H_
