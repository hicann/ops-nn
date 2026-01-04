/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nll_loss_grad.h
 * \brief nll_loss_grad kernel
 */

#ifndef ASCENDC_NLL_LOSS_GRAD_H_
#define ASCENDC_NLL_LOSS_GRAD_H_

#include "kernel_operator.h"
#include "../../inc/platform.h"

namespace NLLLossGrad
{
using namespace AscendC;

constexpr uint32_t THREAD_DIM = 1024;
constexpr uint32_t NONE_MODE = 0;
constexpr uint32_t MEAN_MODE = 1;
constexpr uint32_t SUM_MODE = 2;
constexpr int32_t MAGIC_AND_SPECIAL_CONSTANT = 128;
constexpr uint32_t NUMBER_TWO = 2;
constexpr uint32_t NUMBER_FOUR = 4;

template <typename T, typename F>
class KernelNLLLossGrad
{
public:
    __aicore__ inline KernelNLLLossGrad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y_grad, GM_ADDR target, GM_ADDR weight, GM_ADDR total_weight,
                                GM_ADDR x_grad, NLLLossGradSimtTilingData tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMemset();

private:
    TPipe pipe;
    AscendC::GlobalTensor<T> outputGm_;
    AscendC::GlobalTensor<T> inputXGm_;
    AscendC::GlobalTensor<T> yGrad_;
    AscendC::GlobalTensor<F> target_;
    AscendC::GlobalTensor<T> weight_;
    AscendC::GlobalTensor<T> total_weight_;

    TBuf<TPosition::VECCALC> localTotalWeightBuf_;
    TBuf<TPosition::VECCALC> singleGradBuf_;
    TBuf<TPosition::VECCALC> tmpOutBuf_;

    LocalTensor<float> localTotalWeight_;
    LocalTensor<float> singleGrad_;
    LocalTensor<float> tmpOut_;

    uint64_t batchNum_{1};
    uint64_t classNum_{1};
    uint64_t height_{1};
    uint64_t width_{1};
    int32_t ignoreIdx_{-100};
    uint32_t reductionMode_{1};
    uint32_t usedThread_{THREAD_DIM};
    uint32_t xDims_{1};
    uint32_t blockId_;
    uint32_t blockNums_;
    uint64_t productOfNHW_{0};
    uint64_t productOfHW_{0};

    bool isMemsetCore_;
    bool isComputeCore_;

    uint64_t blockPerCore_;
    uint64_t blockTailCore_;
};

template <typename T, typename F>
__aicore__ inline void KernelNLLLossGrad<T, F>::Init(GM_ADDR x, GM_ADDR y_grad, GM_ADDR target, GM_ADDR weight,
                                                     GM_ADDR total_weight, GM_ADDR x_grad,
                                                     NLLLossGradSimtTilingData tilingData)
{
    inputXGm_.SetGlobalBuffer((__gm__ T*)(x));
    outputGm_.SetGlobalBuffer((__gm__ T*)(x_grad));
    yGrad_.SetGlobalBuffer((__gm__ T*)(y_grad));
    target_.SetGlobalBuffer((__gm__ F*)(target));
    weight_.SetGlobalBuffer((__gm__ T*)(weight));
    total_weight_.SetGlobalBuffer((__gm__ T*)(total_weight));

    batchNum_ = tilingData.batchNum;
    classNum_ = tilingData.classNum;
    height_ = tilingData.height;
    width_ = tilingData.width;
    xDims_ = tilingData.xDims;
    ignoreIdx_ = tilingData.ignoreIdx;
    reductionMode_ = tilingData.reductionMode;
    usedThread_ = tilingData.usedThread;
    productOfNHW_ = batchNum_ * height_ * width_;
    productOfHW_ = height_ * width_;

    this->blockId_ = GetBlockIdx();
    this->blockNums_ = GetBlockNum();

    this->isMemsetCore_ = blockId_ < tilingData.notVeryImportantProcessCoreNums;
    this->isComputeCore_ = blockId_ < tilingData.veryImportantProcessCoreNums;
    this->blockPerCore_ = tilingData.blockPerCore;
    this->blockTailCore_ = tilingData.blockTailCore;
}

__aicore__ inline void TargetCheck(const int64_t idx, const int64_t classNum)
{
    ASSERT((0 <= idx && idx < classNum) &&
           ("Currert target is %ld, which should be in range (0, classNum:%ld)", idx, classNum));
}

template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNoReduce2d(uint64_t classNum_, uint64_t batchNum_,
                                                                                __gm__ F* target_, int32_t ignoreIdx_,
                                                                                __gm__ T* outputGm_, __gm__ T* yGrad_,
                                                                                __gm__ T* weight_, uint32_t blockId_,
                                                                                uint32_t blockNums_)
{
    int64_t localClassNum = classNum_;
    for (uint64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < batchNum_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        int64_t chosenClass = target_[i];
        if (chosenClass == ignoreIdx_) {
            continue;
        }
        TargetCheck(chosenClass, localClassNum);
        outputGm_[chosenClass + i * localClassNum] =
            -static_cast<T>(static_cast<float>(yGrad_[i]) * static_cast<float>(weight_[chosenClass]));
    }
}

template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeReduceMean2d(
    uint64_t classNum_, uint64_t batchNum_, __gm__ F* target_, int32_t ignoreIdx_, __ubuf__ float* tmpOut_,
    __gm__ T* outputGm_, __gm__ T* yGrad_, __gm__ T* weight_, uint32_t blockId_, uint32_t blockNums_)
{
    int64_t localClassNum = classNum_;
    ASSERT(localTotalWeight != 0 && "Input total weight must not be 0!");
    for (uint64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < batchNum_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        int64_t chosenClass = target_[i];
        if (chosenClass == ignoreIdx_) {
            continue;
        }
        TargetCheck(chosenClass, localClassNum);
        outputGm_[chosenClass + i * localClassNum] =
            -static_cast<T>(tmpOut_[0] * static_cast<float>(weight_[chosenClass]));
    }
}

template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeReduceSum2d(
    uint64_t classNum_, uint64_t batchNum_, __gm__ F* target_, int32_t ignoreIdx_, __ubuf__ float* singleGrad_,
    __gm__ T* outputGm_, __gm__ T* yGrad_, __gm__ T* weight_, uint32_t blockId_, uint32_t blockNums_)
{
    int64_t localClassNum = classNum_;
    for (uint64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < batchNum_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        int64_t chosenClass = target_[i];
        if (chosenClass == ignoreIdx_) {
            continue;
        }
        TargetCheck(chosenClass, localClassNum);
        outputGm_[chosenClass + i * localClassNum] =
            -static_cast<T>(singleGrad_[0] * static_cast<float>(weight_[chosenClass]));
    }
}


template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNoReduce4d(uint64_t classNum_, uint64_t batchNum_, uint64_t height_, uint64_t width_,
                                                                                __gm__ F* target_, int32_t ignoreIdx_,
                                                                                __gm__ T* outputGm_, __gm__ T* yGrad_,
                                                                                __gm__ T* weight_, uint32_t blockId_,
                                                                                uint32_t blockNums_, uint64_t productOfNHW_, uint64_t productOfHW_)
{
    int64_t localClassNum = classNum_;
    for (uint64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < productOfNHW_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        int64_t chosenClass = target_[i];
        if (chosenClass == ignoreIdx_) {
            continue;
        }
        auto n = i / productOfHW_;
        auto rem = i % productOfHW_;
        TargetCheck(chosenClass, localClassNum);

        outputGm_[n * localClassNum * productOfHW_ + chosenClass * productOfHW_ + rem] =
            -static_cast<T>(static_cast<float>(yGrad_[i]) * static_cast<float>(weight_[chosenClass]));
    }
}

template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeReduceMean4d(
    uint64_t classNum_, uint64_t batchNum_, uint64_t height_, uint64_t width_, __gm__ F* target_, int32_t ignoreIdx_, __ubuf__ float* tmpOut_,
    __gm__ T* outputGm_, __gm__ T* yGrad_, __gm__ T* weight_, uint32_t blockId_, uint32_t blockNums_, uint64_t productOfNHW_, uint64_t productOfHW_)
{
    int64_t localClassNum = classNum_;
    ASSERT(localTotalWeight != 0 && "Input total weight must not be 0!");
    for (uint64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < productOfNHW_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        int64_t chosenClass = target_[i];
        if (chosenClass == ignoreIdx_) {
            continue;
        }
        auto n = i / productOfHW_;
        auto rem = i % productOfHW_;
        TargetCheck(chosenClass, localClassNum);
        outputGm_[n * localClassNum * productOfHW_ + chosenClass * productOfHW_ + rem] =
            -static_cast<T>(tmpOut_[0] * static_cast<float>(weight_[chosenClass]));
    }
}

template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeReduceSum4d(
    uint64_t classNum_, uint64_t batchNum_, uint64_t height_, uint64_t width_, __gm__ F* target_, int32_t ignoreIdx_, __ubuf__ float* singleGrad_,
    __gm__ T* outputGm_, __gm__ T* yGrad_, __gm__ T* weight_, uint32_t blockId_, uint32_t blockNums_, uint64_t productOfNHW_, uint64_t productOfHW_)
{
    int64_t localClassNum = classNum_;
    for (uint64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < productOfNHW_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        int64_t chosenClass = target_[i];
        if (chosenClass == ignoreIdx_) {
            continue;
        }
        auto n = i / productOfHW_;
        auto rem = i % productOfHW_;
        TargetCheck(chosenClass, localClassNum);
        outputGm_[n * localClassNum * productOfHW_ + chosenClass * productOfHW_ + rem] =
            -static_cast<T>(singleGrad_[0] * static_cast<float>(weight_[chosenClass]));
    }
}


template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void ComputeSetValue(
    uint64_t classNum_, uint64_t batchNum_, uint32_t blockId_, uint32_t blockNums_, __ubuf__ float* singleGrad_,
    __gm__ T* outputGm_, __gm__ T* yGrad_, __ubuf__ float* localTotalWeight_, __gm__ T* total_weight_)
{
    if (Simt::GetThreadIdx() == 0) {
        localTotalWeight_[0] = static_cast<float>(total_weight_[0]);
        singleGrad_[0] = static_cast<float>(yGrad_[0]);
    }
}


template <typename T, typename F>
__aicore__ inline void KernelNLLLossGrad<T, F>::ProcessMemset()
{
    if(this->isMemsetCore_){
        uint64_t initCoreNum = this->blockPerCore_;
        uint64_t initOffset = initCoreNum * this->blockId_;
        if(this->blockId_ < this->blockTailCore_){
            initCoreNum += 1;
            initOffset += this->blockId_;
        }else {
            initOffset += this->blockTailCore_;
        }
        InitOutput<T>(outputGm_[initOffset], initCoreNum, static_cast<T>(0));
        SyncAll();
    }
}

template <typename T, typename F>
__aicore__ inline void KernelNLLLossGrad<T, F>::Process()
{
    ProcessMemset();

    pipe.InitBuffer(localTotalWeightBuf_, 1 * sizeof(float));
    pipe.InitBuffer(singleGradBuf_, 1 * sizeof(float));
    pipe.InitBuffer(tmpOutBuf_, 1 * sizeof(float));
    localTotalWeight_ = localTotalWeightBuf_.Get<float>();
    singleGrad_ = singleGradBuf_.Get<float>();
    tmpOut_ = tmpOutBuf_.Get<float>();

    if (this->isComputeCore_) {
        AscendC::Simt::VF_CALL<ComputeSetValue<T, F>>(
            AscendC::Simt::Dim3{usedThread_}, classNum_, batchNum_, blockId_, blockNums_,
            (__ubuf__ float*)singleGrad_.GetPhyAddr(), (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)yGrad_.GetPhyAddr(),
            (__ubuf__ float*)localTotalWeight_.GetPhyAddr(), (__gm__ T*)total_weight_.GetPhyAddr());
    }

    SyncAll();

    if (!this->isComputeCore_) {
        return;
    }

    tmpOut_(0) = singleGrad_(0) / localTotalWeight_(0);
    PipeBarrier<PIPE_ALL>();
    if(xDims_ == 1 || xDims_ == NUMBER_TWO){
        if (reductionMode_ == NONE_MODE) {
            AscendC::Simt::VF_CALL<SimtComputeNoReduce2d<T, F>>(
                AscendC::Simt::Dim3{usedThread_}, classNum_, batchNum_, (__gm__ F*)target_.GetPhyAddr(), ignoreIdx_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)yGrad_.GetPhyAddr(), (__gm__ T*)weight_.GetPhyAddr(),
                blockId_, blockNums_);
        } else if (reductionMode_ == MEAN_MODE) {
            AscendC::Simt::VF_CALL<SimtComputeReduceMean2d<T, F>>(
                AscendC::Simt::Dim3{usedThread_}, classNum_, batchNum_, (__gm__ F*)target_.GetPhyAddr(), ignoreIdx_,
                (__ubuf__ float*)tmpOut_.GetPhyAddr(), (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)yGrad_.GetPhyAddr(),
                (__gm__ T*)weight_.GetPhyAddr(), blockId_, blockNums_);
        } else if (reductionMode_ == SUM_MODE) {
            AscendC::Simt::VF_CALL<SimtComputeReduceSum2d<T, F>>(
                AscendC::Simt::Dim3{usedThread_}, classNum_, batchNum_, (__gm__ F*)target_.GetPhyAddr(), ignoreIdx_,
                (__ubuf__ float*)singleGrad_.GetPhyAddr(), (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)yGrad_.GetPhyAddr(), (__gm__ T*)weight_.GetPhyAddr(), blockId_, blockNums_);
        }
    }else if(xDims_ == NUMBER_FOUR){
        if (reductionMode_ == NONE_MODE) {
            AscendC::Simt::VF_CALL<SimtComputeNoReduce4d<T, F>>(
                AscendC::Simt::Dim3{usedThread_}, classNum_, batchNum_, height_, width_, (__gm__ F*)target_.GetPhyAddr(), ignoreIdx_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)yGrad_.GetPhyAddr(), (__gm__ T*)weight_.GetPhyAddr(),
                blockId_, blockNums_, productOfNHW_, productOfHW_);
        } else if (reductionMode_ == MEAN_MODE) {
            AscendC::Simt::VF_CALL<SimtComputeReduceMean4d<T, F>>(
                AscendC::Simt::Dim3{usedThread_}, classNum_, batchNum_, height_, width_, (__gm__ F*)target_.GetPhyAddr(), ignoreIdx_,
                (__ubuf__ float*)tmpOut_.GetPhyAddr(), (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)yGrad_.GetPhyAddr(),
                (__gm__ T*)weight_.GetPhyAddr(), blockId_, blockNums_, productOfNHW_, productOfHW_);
        } else if (reductionMode_ == SUM_MODE) {
            AscendC::Simt::VF_CALL<SimtComputeReduceSum4d<T, F>>(
                AscendC::Simt::Dim3{usedThread_}, classNum_, batchNum_, height_, width_, (__gm__ F*)target_.GetPhyAddr(), ignoreIdx_,
                (__ubuf__ float*)singleGrad_.GetPhyAddr(), (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)yGrad_.GetPhyAddr(), (__gm__ T*)weight_.GetPhyAddr(), blockId_, blockNums_, productOfNHW_, productOfHW_);
        }
    }
}
}  // namespace NLLLossGrad

#endif