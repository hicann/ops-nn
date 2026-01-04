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
 * \file nll_loss_simt.h
 * \brief nll_loss_simt
 */

#ifndef ASCENDC_NLL_LOSS_SIMT_H_
#define ASCENDC_NLL_LOSS_SIMT_H_

#include <cmath>
#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

constexpr int64_t THREAD_DIM = 512;
constexpr uint32_t REDUCTION_NONE =0;
constexpr uint32_t REDUCTION_MEAN = 1;
constexpr uint32_t REDUCTION_SUM = 2;
constexpr uint32_t BINARY_HALF = 2;
constexpr int64_t THREAD_DIM_INTERBLOCK = 64;
constexpr int64_t NUMBER_TWO = 2;
constexpr int64_t NUMBER_FOUR = 4;

template <typename U>
__aicore__ inline void TargetCheck(const U idx, const uint32_t xDimC_)
{
    ASSERT((0 <= idx && idx < xDimC_) && "Target is invalid value");
}

__aicore__ inline void SimtComputeBinaryReduction(__ubuf__ float* tmpOut_, __ubuf__ float* tmpWeight_)
{
    uint32_t countBR = Simt::GetThreadNum();
    while (countBR > 1) {
        uint32_t halfBR = (countBR + 1) / BINARY_HALF;
        if (Simt::GetThreadIdx() < halfBR) {
            tmpOut_[Simt::GetThreadIdx()] = tmpOut_[Simt::GetThreadIdx() + halfBR] + tmpOut_[Simt::GetThreadIdx()];

            tmpWeight_[Simt::GetThreadIdx()] =
                tmpWeight_[Simt::GetThreadIdx() + halfBR] + tmpWeight_[Simt::GetThreadIdx()];
        }
        Simt::ThreadBarrier();
        countBR = halfBR;
    }
}

template <typename U, typename T, uint64_t reduction>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNLLLoss1d(
    __gm__ U* targetGm_, int64_t ignoreIndex_, __gm__ T* yGm_, __gm__ T* totalWeight_, uint32_t isWeightPresent_,
    __gm__ T* weightGm_, __gm__ T* xGm_, uint32_t xDimC_)
{
    U targetIndex = targetGm_[0];
    T curWeight = 0;
    if (targetIndex == ignoreIndex_) {
        yGm_[0] = static_cast<T>(0);
        totalWeight_[0] = static_cast<T>(0);
    } else {
        TargetCheck(targetIndex, xDimC_);
        if (isWeightPresent_ == 1) {
            curWeight = weightGm_[targetIndex];
        } else {
            curWeight = static_cast<T>(1);
        }
        totalWeight_[0] = curWeight;
        if constexpr (reduction == REDUCTION_MEAN) {
            if (curWeight == 0) {
                yGm_[0] = NAN;
            } else {
                yGm_[0] = -xGm_[targetIndex];
            }
        } else {
            yGm_[0] =
                -static_cast<T>(static_cast<float>(weightGm_[targetIndex]) * static_cast<float>(xGm_[targetIndex]));
        }
    }
}

template <typename U, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNLLLoss2dNone(
    int64_t blockId_, int64_t xDimN_, int64_t blockNums_, __gm__ U* targetGm_, int64_t ignoreIndex_, __gm__ T* yGm_,
    uint32_t isWeightPresent_, __gm__ T* weightGm_, __gm__ T* xGm_, uint32_t xDimC_)
{
    for (int64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < xDimN_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        U targetIndex = targetGm_[i];
        if (targetIndex == ignoreIndex_) {
            yGm_[i] = static_cast<T>(0);
            continue;
        }
        TargetCheck(targetIndex, xDimC_);
        T curWeight = isWeightPresent_ == 1 ? weightGm_[targetIndex] : static_cast<T>(1);
        yGm_[i] = -static_cast<T>(static_cast<float>(curWeight) * static_cast<float>(xGm_[(i)*xDimC_ + targetIndex]));
    }
}

template <typename U, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNLLLoss2dSum(
    int64_t blockId_, int64_t xDimN_, int64_t blockNums_, __gm__ U* targetGm_, int64_t ignoreIndex_, uint32_t xDimC_,
    uint32_t isWeightPresent_, __gm__ T* weightGm_, __ubuf__ float* tmpOut_, __gm__ T* xGm_, __ubuf__ float* tmpWeight_,
    __gm__ volatile float* tmpSumGm_, __gm__ volatile float* tmpWeightGm_)
{
    for (int64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < xDimN_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        U targetIndex = targetGm_[i];
        int64_t ubIdx = (i % (blockNums_ * Simt::GetThreadNum())) % THREAD_DIM;
        if (targetIndex == ignoreIndex_) {
            continue;
        }
        TargetCheck(targetIndex, xDimC_);
        T curWeight = isWeightPresent_ == 1 ? weightGm_[targetIndex] : static_cast<T>(1);
        tmpOut_[ubIdx] =
            tmpOut_[ubIdx] - static_cast<float>(curWeight) * static_cast<float>(xGm_[i * xDimC_ + targetIndex]);
        tmpWeight_[ubIdx] = static_cast<float>(curWeight) + tmpWeight_[ubIdx];
    }
    Simt::ThreadBarrier();
    SimtComputeBinaryReduction(tmpOut_, tmpWeight_);

    if (Simt::GetThreadIdx() == 0) {
        tmpSumGm_[blockId_] = tmpOut_[0];
        tmpWeightGm_[blockId_] = tmpWeight_[0];
    }
}

template <typename U, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNLLLoss2dMean(
    int64_t blockId_, int64_t xDimN_, int64_t blockNums_, __gm__ U* targetGm_, int64_t ignoreIndex_,
    __ubuf__ float* tmpOut_, __ubuf__ float* tmpWeight_, uint32_t isWeightPresent_, __gm__ T* weightGm_, __gm__ T* xGm_,
    __gm__ volatile float* tmpSumGm_, uint32_t xDimC_, __gm__ volatile float* tmpWeightGm_)
{
    for (int64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < xDimN_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        U targetIndex = targetGm_[i];
        int64_t ubIdx = (i % (blockNums_ * Simt::GetThreadNum())) % THREAD_DIM;
        if (targetIndex == ignoreIndex_) {
            continue;
        }
        TargetCheck(targetIndex, xDimC_);
        T curWeight = isWeightPresent_ == 1 ? weightGm_[targetIndex] : static_cast<T>(1);
        tmpOut_[ubIdx] =
            tmpOut_[ubIdx] - static_cast<float>(curWeight) * static_cast<float>(xGm_[i * xDimC_ + targetIndex]);
        tmpWeight_[ubIdx] = static_cast<float>(curWeight) + tmpWeight_[ubIdx];
    }
    Simt::ThreadBarrier();
    SimtComputeBinaryReduction(tmpOut_, tmpWeight_);

    if (Simt::GetThreadIdx() == 0) {
        tmpSumGm_[blockId_] = tmpOut_[0];
        tmpWeightGm_[blockId_] = tmpWeight_[0];
    }
}

template <typename U, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNLLLoss4dNone(
    __gm__ U* targetGm, int64_t ignoreIndex, __ubuf__ T* outUb, int64_t offset, uint32_t isWeightPresent,
    __gm__ T* weightGm, __gm__ T* xGm, uint32_t xDimC, int64_t dealingNum, int64_t productOfCHW,
    int64_t productOfHW)
{
    for (int64_t i = Simt::GetThreadIdx(); i < dealingNum; i = i + Simt::GetThreadNum()) {
        int64_t index = offset + i;
        U targetIndex = targetGm[index]; //blockid_*
        if (targetIndex == ignoreIndex) {
            outUb[i] = static_cast<T>(0);
            continue;
        }
        int64_t n = index / productOfHW;
        int64_t rem = index % productOfHW;
        int64_t xOffset = n * productOfCHW + targetIndex * productOfHW + rem;
        TargetCheck(targetIndex, xDimC);
        T curWeight = isWeightPresent == 1 ? weightGm[targetIndex] : static_cast<T>(1);
        outUb[i] = -static_cast<T>(static_cast<float>(curWeight) * static_cast<float>(xGm[xOffset]));
    }
}

template <typename U, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNLLLoss4dSum(
    int64_t blockId_, int64_t xDimN_, int64_t xDimH_, int64_t xDimW_, int64_t blockNums_, __gm__ U* targetGm_,
    int64_t ignoreIndex_, uint32_t xDimC_, uint32_t isWeightPresent_, __gm__ T* weightGm_, __ubuf__ float* tmpOut_,
    __gm__ T* xGm_, __ubuf__ float* tmpWeight_, __gm__ volatile float* tmpSumGm_, __gm__ volatile float* tmpWeightGm_,
    int64_t productOfNHW_, int64_t productOfCHW_, int64_t productOfHW_)
{
    for (int64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < productOfNHW_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        U targetIndex = targetGm_[i];
        int64_t ubIdx = (i % (blockNums_ * Simt::GetThreadNum())) % THREAD_DIM;
        if (targetIndex == ignoreIndex_) {
            continue;
        }
        int64_t n = i / productOfHW_;
        int64_t rem = i % productOfHW_;
        TargetCheck(targetIndex, xDimC_);
        T curWeight = isWeightPresent_ == 1 ? weightGm_[targetIndex] : static_cast<T>(1);
        tmpOut_[ubIdx] =
            tmpOut_[ubIdx] - static_cast<float>(curWeight) *
                                 static_cast<float>(xGm_[n * productOfCHW_ + targetIndex * productOfHW_ + rem]);
        tmpWeight_[ubIdx] = static_cast<float>(curWeight) + tmpWeight_[ubIdx];
    }
    Simt::ThreadBarrier();
    SimtComputeBinaryReduction(tmpOut_, tmpWeight_);

    if (Simt::GetThreadIdx() == 0) {
        tmpSumGm_[blockId_] = tmpOut_[0];
        tmpWeightGm_[blockId_] = tmpWeight_[0];
    }
}

template <typename U, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNLLLoss4dMean(
    int64_t blockId_, int64_t xDimN_, int64_t xDimH_, int64_t xDimW_, int64_t blockNums_, __gm__ U* targetGm_,
    int64_t ignoreIndex_, __ubuf__ float* tmpOut_, __ubuf__ float* tmpWeight_, uint32_t isWeightPresent_,
    __gm__ T* weightGm_, __gm__ T* xGm_, __gm__ volatile float* tmpSumGm_, uint32_t xDimC_,
    __gm__ volatile float* tmpWeightGm_, int64_t productOfNHW_, int64_t productOfCHW_, int64_t productOfHW_)
{
    for (int64_t i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < productOfNHW_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        U targetIndex = targetGm_[i];
        int64_t ubIdx = (i % (blockNums_ * Simt::GetThreadNum())) % THREAD_DIM;
        if (targetIndex == ignoreIndex_) {
            continue;
        }
        int64_t n = i / productOfHW_;
        int64_t rem = i % productOfHW_;
        TargetCheck(targetIndex, xDimC_);
        T curWeight = isWeightPresent_ == 1 ? weightGm_[targetIndex] : static_cast<T>(1);
        tmpOut_[ubIdx] =
            tmpOut_[ubIdx] - static_cast<float>(curWeight) *
                                 static_cast<float>(xGm_[n * productOfCHW_ + targetIndex * productOfHW_ + rem]);
        tmpWeight_[ubIdx] = static_cast<float>(curWeight) + tmpWeight_[ubIdx];
    }
    Simt::ThreadBarrier();
    SimtComputeBinaryReduction(tmpOut_, tmpWeight_);

    if (Simt::GetThreadIdx() == 0) {
        tmpSumGm_[blockId_] = tmpOut_[0];
        tmpWeightGm_[blockId_] = tmpWeight_[0];
    }
}

template <typename U, typename T, uint64_t reduction>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void CalBinaryReductionInterBlock(
    int64_t blockNums_, uint32_t calNumPower_, __gm__ volatile float* tmpSumGm_, __gm__ volatile float* tmpWeightGm_,
    __gm__ T* yGm_, __gm__ T* totalWeight_)
{
    uint32_t valueLarger2Power = blockNums_ - calNumPower_;
    if (Simt::GetThreadIdx() < valueLarger2Power) {
        tmpSumGm_[Simt::GetThreadIdx()] =
            tmpSumGm_[Simt::GetThreadIdx() + calNumPower_] + tmpSumGm_[Simt::GetThreadIdx()];
        tmpWeightGm_[Simt::GetThreadIdx()] =
            tmpWeightGm_[Simt::GetThreadIdx() + calNumPower_] + tmpWeightGm_[Simt::GetThreadIdx()];
    }
    Simt::ThreadBarrier();

    int64_t countBR = calNumPower_;
    while (countBR > 1) {
        int64_t halfBR = (countBR + 1) / BINARY_HALF;
        if (Simt::GetThreadIdx() < halfBR) {
            tmpSumGm_[Simt::GetThreadIdx()] =
                tmpSumGm_[Simt::GetThreadIdx() + halfBR] + tmpSumGm_[Simt::GetThreadIdx()];
            tmpWeightGm_[Simt::GetThreadIdx()] =
                tmpWeightGm_[Simt::GetThreadIdx() + halfBR] + tmpWeightGm_[Simt::GetThreadIdx()];
        }
        Simt::ThreadBarrier();
        countBR = halfBR;
    }
    if (Simt::GetThreadIdx() == 0) {
        if constexpr (is_same<bfloat16_t, T>::value) {
            if constexpr (reduction == REDUCTION_MEAN) {
                yGm_[0] = Simt::Cast<bfloat16_t, float, RoundMode::CAST_EVEN>((tmpSumGm_[0] / tmpWeightGm_[0]));
            } else {
                yGm_[0] = Simt::Cast<bfloat16_t, float, RoundMode::CAST_EVEN>(tmpSumGm_[0]);
            }
        } else if constexpr (is_same<half, T>::value) {
            if constexpr (reduction == REDUCTION_MEAN) {
                yGm_[0] = Simt::Cast<half, float, RoundMode::CAST_EVEN>((tmpSumGm_[0] / tmpWeightGm_[0]));
            } else {
                yGm_[0] = Simt::Cast<half, float, RoundMode::CAST_EVEN>(tmpSumGm_[0]);
            }
        } else {
            if constexpr (reduction == REDUCTION_MEAN) {
                yGm_[0] = tmpSumGm_[0] / tmpWeightGm_[0];
            } else {
                yGm_[0] = tmpSumGm_[0];
            }
        }
        if constexpr (is_same<bfloat16_t, T>::value) {
            totalWeight_[0] = Simt::Cast<bfloat16_t, float, RoundMode::CAST_EVEN>(tmpWeightGm_[0]);
        } else if constexpr (is_same<half, T>::value) {
            totalWeight_[0] = Simt::Cast<half, float, RoundMode::CAST_EVEN>(tmpWeightGm_[0]);
        } else {
            totalWeight_[0] = tmpWeightGm_[0];
        }
    }
}

template <typename T, typename U, uint64_t schId, uint64_t xDims, uint64_t reduction>
class KernelNLLLossSimt {
public:
    __aicore__ inline KernelNLLLossSimt()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR totalWeight, GM_ADDR workspace,
        NLLLossACTilingData tilingData)
    {
        xGm_.SetGlobalBuffer((__gm__ T*)(x));
        yGm_.SetGlobalBuffer((__gm__ T*)(y));
        totalWeight_.SetGlobalBuffer((__gm__ T*)(totalWeight));
        targetGm_.SetGlobalBuffer((__gm__ U*)(target));
        isWeightPresent_ = tilingData.isWeightPresent;
        if (isWeightPresent_ == 1) {
            weightGm_.SetGlobalBuffer((__gm__ T*)(weight));
        }

        xDims_ = tilingData.xDims;
        ignoreIndex_ = tilingData.ignoreIndex;
        reduction_ = tilingData.reduction;
        xDimN_ = tilingData.xDimN;
        xDimC_ = tilingData.xDimC;
        xDimH_ = tilingData.xDimH;
        xDimW_ = tilingData.xDimW;
        fullIndex_ = tilingData.fullIndex;
        coreLoopCnt_ = tilingData.coreLoopCnt;
        calNumPower_ = tilingData.calNumInterBlock;

        productOfNHW_ = xDimN_ * xDimH_ * xDimW_;
        productOfCHW_ = xDimC_ * xDimH_ * xDimW_;
        productOfHW_ = xDimH_ * xDimW_;

        blockId_ = GetBlockIdx();
        blockNums_ = GetBlockNum();

        if constexpr (reduction == REDUCTION_MEAN || reduction == REDUCTION_SUM) {
            tmpSumGm_.SetGlobalBuffer((__gm__ float*)workspace, blockNums_);
            tmpWeightGm_.SetGlobalBuffer((__gm__ float*)workspace + blockNums_, blockNums_);
            pipe.InitBuffer(tmpOutBuffer_, THREAD_DIM * sizeof(float));
            pipe.InitBuffer(tmpWeightBuffer_, THREAD_DIM * sizeof(float));
            tmpOut_ = tmpOutBuffer_.Get<float>();
            tmpWeight_ = tmpWeightBuffer_.Get<float>();
        }
        if constexpr (reduction == REDUCTION_NONE && xDims == 4) {
            coreNum_ = tilingData.coreNum;
            dealingNumOnce_ = tilingData.dealingNumOnce;
            dealingNumOneCore_ = tilingData.dealingNumOneCore + 1;
            offsetGmY_ = blockId_ * dealingNumOneCore_;
            if (blockId_ >= tilingData.frontCore) {
                dealingNumOneCore_ = tilingData.dealingNumOneCore;
                offsetGmY_ = tilingData.frontCore + blockId_ * tilingData.dealingNumOneCore;
            }
            loopTimes_ = dealingNumOneCore_ / dealingNumOnce_;
            dealingNumTail_ = dealingNumOneCore_ % dealingNumOnce_;
            pipe.InitBuffer(outQueue_, 1, dealingNumOnce_ * sizeof(T));
        }
    }

    // 此处 对BF16类型的除法是临时方案，等编译器提供BF16的除法会进行替换
    __aicore__ inline float bfloat162float(bfloat16_t in) const
    {
        uint16_t tmpIn16 = (uint16_t&)(in);
        uint32_t tmpIn32 = tmpIn16 << 16;
        float out = (float&)tmpIn32;
        return out;
    }

    __aicore__ inline bfloat16_t float2bfloat16(float in) const
    {
        uint32_t tmpIn32 = (uint32_t&)(in);
        uint16_t tmpIn16 = tmpIn32 >> 16;
        bfloat16_t out = (bfloat16_t&)tmpIn16;
        return out;
    }

    __aicore__ inline bfloat16_t div(bfloat16_t src0, bfloat16_t src1) const
    {
        float src0Float = bfloat162float(src0);
        float src1Float = bfloat162float(src1);
        float out = 0;
        if (src1Float == 0) {
            return NAN;
        } else {
            out = src0Float / src1Float;
        }
        return float2bfloat16(out);
    }

    __aicore__ inline void SimdDataCopyOutFor4d(int32_t nhwNum, int64_t offset)
    {
        LocalTensor<T> outUb = outQueue_.DeQue<T>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = nhwNum * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm_[offset], outUb, copyParams);
        outQueue_.FreeTensor(outUb);
    }

    __aicore__ inline void DealingRedutionNoneFor4d()
    {
        for (int32_t i = 0; i < loopTimes_; i++) {
            LocalTensor<T> outUb = outQueue_.AllocTensor<T>();
            int64_t offset = offsetGmY_ + i * dealingNumOnce_;
            AscendC::Simt::VF_CALL<SimtComputeNLLLoss4dNone<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)},
                    (__gm__ U*)targetGm_.GetPhyAddr(), ignoreIndex_, (__ubuf__ T*)outUb.GetPhyAddr(), offset, isWeightPresent_,
                    (__gm__ T*)weightGm_.GetPhyAddr(), (__gm__ T*)xGm_.GetPhyAddr(), xDimC_, dealingNumOnce_, productOfCHW_, productOfHW_);
            outQueue_.EnQue(outUb);
            SimdDataCopyOutFor4d(dealingNumOnce_, offset);
        }
        if (dealingNumTail_ != 0) {
            LocalTensor<T> outUb = outQueue_.AllocTensor<T>();
            int64_t offset = offsetGmY_ + loopTimes_ * dealingNumOnce_;
            AscendC::Simt::VF_CALL<SimtComputeNLLLoss4dNone<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)},
                    (__gm__ U*)targetGm_.GetPhyAddr(), ignoreIndex_, (__ubuf__ T*)outUb.GetPhyAddr(), offset, isWeightPresent_,
                    (__gm__ T*)weightGm_.GetPhyAddr(), (__gm__ T*)xGm_.GetPhyAddr(), xDimC_, dealingNumTail_, productOfCHW_, productOfHW_);
            outQueue_.EnQue(outUb);
            SimdDataCopyOutFor4d(dealingNumTail_, offset);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (xDims == 1) {
            AscendC::Simt::VF_CALL<SimtComputeNLLLoss1d<U, T, reduction>>(
                AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, (__gm__ U*)targetGm_.GetPhyAddr(), ignoreIndex_,
                (__gm__ T*)yGm_.GetPhyAddr(), (__gm__ T*)totalWeight_.GetPhyAddr(), isWeightPresent_,
                (__gm__ T*)weightGm_.GetPhyAddr(), (__gm__ T*)xGm_.GetPhyAddr(), xDimC_);
        } else if constexpr (xDims == NUMBER_TWO) {
            if (reduction == 0) {
                AscendC::Simt::VF_CALL<SimtComputeNLLLoss2dNone<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockId_, xDimN_, blockNums_,
                    (__gm__ U*)targetGm_.GetPhyAddr(), ignoreIndex_, (__gm__ T*)yGm_.GetPhyAddr(), isWeightPresent_,
                    (__gm__ T*)weightGm_.GetPhyAddr(), (__gm__ T*)xGm_.GetPhyAddr(), xDimC_);
            } else if constexpr (reduction == REDUCTION_SUM) {
                Duplicate(tmpOut_, 0.0f, THREAD_DIM);
                Duplicate(tmpWeight_, 0.0f, THREAD_DIM);
                AscendC::Simt::VF_CALL<SimtComputeNLLLoss2dSum<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockId_, xDimN_, blockNums_,
                    (__gm__ U*)targetGm_.GetPhyAddr(), ignoreIndex_, xDimC_, isWeightPresent_,
                    (__gm__ T*)weightGm_.GetPhyAddr(), (__ubuf__ float*)tmpOut_.GetPhyAddr(), (__gm__ T*)xGm_.GetPhyAddr(),
                    (__ubuf__ float*)tmpWeight_.GetPhyAddr(), (__gm__ volatile float*)tmpSumGm_.GetPhyAddr(),
                    (__gm__ volatile float*)tmpWeightGm_.GetPhyAddr());
                SyncAll();
                if (blockId_ == 0) {
                    AscendC::Simt::VF_CALL<CalBinaryReductionInterBlock<U, T, reduction>>(
                        AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockNums_, calNumPower_,
                        (__gm__ volatile float*)tmpSumGm_.GetPhyAddr(), (__gm__ volatile float*)tmpWeightGm_.GetPhyAddr(),
                        (__gm__ T*)yGm_.GetPhyAddr(), (__gm__ T*)totalWeight_.GetPhyAddr());
                }
            } else if constexpr (reduction == REDUCTION_MEAN) {
                Duplicate(tmpOut_, 0.0f, THREAD_DIM);
                Duplicate(tmpWeight_, 0.0f, THREAD_DIM);
                AscendC::Simt::VF_CALL<SimtComputeNLLLoss2dMean<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockId_, xDimN_, blockNums_,
                    (__gm__ U*)targetGm_.GetPhyAddr(), ignoreIndex_, (__ubuf__ float*)tmpOut_.GetPhyAddr(),
                    (__ubuf__ float*)tmpWeight_.GetPhyAddr(), isWeightPresent_, (__gm__ T*)weightGm_.GetPhyAddr(),
                    (__gm__ T*)xGm_.GetPhyAddr(), (__gm__ volatile float*)tmpSumGm_.GetPhyAddr(), xDimC_,
                    (__gm__ volatile float*)tmpWeightGm_.GetPhyAddr());
                SyncAll();
                if (blockId_ == 0) {
                    AscendC::Simt::VF_CALL<CalBinaryReductionInterBlock<U, T, reduction>>(
                        AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockNums_, calNumPower_,
                        (__gm__ volatile float*)tmpSumGm_.GetPhyAddr(), (__gm__ volatile float*)tmpWeightGm_.GetPhyAddr(),
                        (__gm__ T*)yGm_.GetPhyAddr(), (__gm__ T*)totalWeight_.GetPhyAddr());
                }
            }
        } else if constexpr (xDims == NUMBER_FOUR){
            if constexpr (reduction == 0) {
                if (blockId_ < coreNum_) {
                    DealingRedutionNoneFor4d();
                }
            } else if constexpr (reduction == REDUCTION_SUM) {

                Duplicate(tmpOut_, 0.0f, THREAD_DIM);
                Duplicate(tmpWeight_, 0.0f, THREAD_DIM);
                AscendC::Simt::VF_CALL<SimtComputeNLLLoss4dSum<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockId_, xDimN_, xDimH_, xDimW_,
                    blockNums_, (__gm__ U*)targetGm_.GetPhyAddr(), ignoreIndex_, xDimC_, isWeightPresent_,
                    (__gm__ T*)weightGm_.GetPhyAddr(), (__ubuf__ float*)tmpOut_.GetPhyAddr(),
                    (__gm__ T*)xGm_.GetPhyAddr(), (__ubuf__ float*)tmpWeight_.GetPhyAddr(),
                    (__gm__ volatile float*)tmpSumGm_.GetPhyAddr(), (__gm__ volatile float*)tmpWeightGm_.GetPhyAddr(),
                    productOfNHW_, productOfCHW_, productOfHW_);
                SyncAll();
                if (blockId_ == 0) {
                    AscendC::Simt::VF_CALL<CalBinaryReductionInterBlock<U, T, reduction>>(
                        AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockNums_, calNumPower_,
                        (__gm__ volatile float*)tmpSumGm_.GetPhyAddr(), (__gm__ volatile float*)tmpWeightGm_.GetPhyAddr(),
                        (__gm__ T*)yGm_.GetPhyAddr(), (__gm__ T*)totalWeight_.GetPhyAddr());
                }
            } else if constexpr (reduction == REDUCTION_MEAN) {
                Duplicate(tmpOut_, 0.0f, THREAD_DIM);
                Duplicate(tmpWeight_, 0.0f, THREAD_DIM);
                AscendC::Simt::VF_CALL<SimtComputeNLLLoss4dMean<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockId_, xDimN_, xDimH_, xDimW_, blockNums_,
                    (__gm__ U*)targetGm_.GetPhyAddr(), ignoreIndex_, (__ubuf__ float*)tmpOut_.GetPhyAddr(),
                    (__ubuf__ float*)tmpWeight_.GetPhyAddr(), isWeightPresent_, (__gm__ T*)weightGm_.GetPhyAddr(),
                    (__gm__ T*)xGm_.GetPhyAddr(), (__gm__ volatile float*)tmpSumGm_.GetPhyAddr(), xDimC_,
                    (__gm__ volatile float*)tmpWeightGm_.GetPhyAddr(), productOfNHW_, productOfCHW_, productOfHW_);

                SyncAll();
                if (blockId_ == 0) {
                    AscendC::Simt::VF_CALL<CalBinaryReductionInterBlock<U, T, reduction>>(
                        AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, blockNums_, calNumPower_,
                        (__gm__ volatile float*)tmpSumGm_.GetPhyAddr(), (__gm__ volatile float*)tmpWeightGm_.GetPhyAddr(),
                        (__gm__ T*)yGm_.GetPhyAddr(), (__gm__ T*)totalWeight_.GetPhyAddr());
                }
            }
        }
    }

private:
    TPipe pipe;
    AscendC::GlobalTensor<T> xGm_, weightGm_, yGm_, totalWeight_;
    AscendC::GlobalTensor<U> targetGm_;
    AscendC::GlobalTensor<float> tmpSumGm_, tmpWeightGm_;

    TBuf<TPosition::VECCALC> tmpOutBuffer_;
    TBuf<TPosition::VECCALC> tmpWeightBuffer_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    LocalTensor<float> tmpOut_;
    LocalTensor<float> tmpWeight_;

    int64_t ignoreIndex_ = 0;
    uint32_t reduction_ = 0;
    int64_t xDimN_ = 0;
    uint32_t xDimC_ = 1;
    int64_t xDimH_ = 1;
    int64_t xDimW_ = 1;
    uint32_t coreNum_ = 0;
    uint32_t isWeightPresent_ = 0;
    int64_t blockId_ = 0;
    int64_t blockNums_ = 0;
    int64_t xDims_ = 0;
    uint32_t fullIndex_ = 0;
    uint32_t coreLoopCnt_ = 0;
    uint32_t blockOffset_ = 0;
    uint32_t calNumPower_ = 0;
    int64_t productOfNHW_ = 0;
    int64_t productOfCHW_ = 0;
    int64_t productOfHW_ = 0;
    int64_t dealingNumOneCore_ = 0;
    int64_t dealingNumOnce_ = 0;
    int64_t loopTimes_ = 0;
    int64_t dealingNumTail_ = 0;
    int64_t offsetGmY_ = 0;
};
#endif