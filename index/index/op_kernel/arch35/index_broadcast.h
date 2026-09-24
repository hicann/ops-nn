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
 * \file index_broadcast.h
 * \brief Ascendc index indices-broadcast kernel (only for Index, IS_BROADCAST=1)
 */

#ifndef ASCENDC_INDEX_BROADCAST_H_
#define ASCENDC_INDEX_BROADCAST_H_

#include "index.h"

template <typename T>
struct BroadcastCalcParams {
    uint64_t indexedDimStride_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t nonIndexedStride_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t fusedNonIndexedStride_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    T indexedShape_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t indexList_[8];
    T shift_[4];
    T m_[4];
    T fusedM_[8] = {0};
    T fusedShift_[8] = {0};

    uint32_t broadcastDimNum_{0};
    T broadcastDivisor_[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    T broadcastM_[8] = {0};
    T broadcastShift_[8] = {0};
    T effectiveStride_[8][8] = {{0}};
};

namespace Index {

constexpr uint32_t INDEX_THREE = 3;
constexpr uint32_t BC_LARGE_TAIL_FACTOR = 2;
constexpr uint32_t BC_SUB_GROUPS = 4;

template <typename T, typename F, typename P, int Dim, typename T2, uint16_t LAUNCH_BOUND_LIMIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(LAUNCH_BOUND_LIMIT) inline void SimtComputeBroadcastContinue(
    uint32_t blockId_, T2 outputLength_, uint32_t blockNums_, uint32_t secondThirdLoopLength_, T2 formerNonIndexStride_,
    uint32_t thirdLoopLength_, __gm__ T* outputGm_, __gm__ T* inputXGm_,
    __ubuf__ BroadcastCalcParams<T2>* calcParamsPtr_)
{
    const __gm__ P* localIndexList[Dim];
    T2 localIndexedShape[Dim];
    T2 localIndexedDimStride[Dim];
    const uint32_t bDimNum = calcParamsPtr_->broadcastDimNum_;

#pragma unroll
    for (uint16_t i = 0; i < Dim; ++i) {
        localIndexList[i] = (__gm__ P*)calcParamsPtr_->indexList_[i];
        localIndexedShape[i] = calcParamsPtr_->indexedShape_[i];
        localIndexedDimStride[i] = calcParamsPtr_->indexedDimStride_[i];
    }

    for (T2 i = blockId_ * blockDim.x + threadIdx.x; i < outputLength_; i = i + blockNums_ * blockDim.x) {
        T2 inputIndex = 0;
        T2 remLength = i;
        if (formerNonIndexStride_ != 1) {
            T2 firstLoopIdx = AscendC::Simt::UintDiv(remLength, calcParamsPtr_->m_[0], calcParamsPtr_->shift_[0]);

            inputIndex += firstLoopIdx * formerNonIndexStride_;
            remLength = remLength - firstLoopIdx * secondThirdLoopLength_;
        }

        T2 secondLoopIdx = AscendC::Simt::UintDiv(remLength, calcParamsPtr_->m_[1], calcParamsPtr_->shift_[1]);
        T2 thirdLoopIdx = remLength - thirdLoopLength_ * secondLoopIdx;
        T2 coord[8];
        T2 remIdx = secondLoopIdx;
        for (uint32_t d = 0; d + 1 < bDimNum; ++d) {
            coord[d] = AscendC::Simt::UintDiv(remIdx, calcParamsPtr_->broadcastM_[d],
                                              calcParamsPtr_->broadcastShift_[d]);
            remIdx -= coord[d] * calcParamsPtr_->broadcastDivisor_[d];
        }
        coord[bDimNum - 1] = remIdx;

#pragma unroll
        for (uint16_t j = 0; j < Dim; ++j) {
            T2 localIdx = 0;
            for (uint32_t d = 0; d < bDimNum; ++d) {
                localIdx += coord[d] * calcParamsPtr_->effectiveStride_[j][d];
            }

            int64_t curIdx = localIndexList[j][localIdx];
            if (curIdx < 0) {
                curIdx += localIndexedShape[j];
            }
            inputIndex += curIdx * localIndexedDimStride[j];
        }
        inputIndex += thirdLoopIdx;
        F()(outputGm_, inputXGm_, i, inputIndex);
    }
}

template <typename T, typename F, typename P, int Dim, typename T2, uint16_t LAUNCH_BOUND_LIMIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(LAUNCH_BOUND_LIMIT) inline void SimtComputeBroadcastContinueLargeTail(
    uint32_t blockId_, T2 outputLength_, uint32_t blockNums_, T2 bSize_, T2 formerNonIndexStride_, T2 thirdLoopLength_,
    __gm__ T* outputGm_, __gm__ T* inputXGm_, __ubuf__ BroadcastCalcParams<T2>* calcParamsPtr_)
{
    const __gm__ P* localIndexList[Dim];
    T2 localIndexedShape[Dim];
    T2 localIndexedDimStride[Dim];
    const uint32_t bDimNum = calcParamsPtr_->broadcastDimNum_;

#pragma unroll
    for (uint16_t i = 0; i < Dim; ++i) {
        localIndexList[i] = (__gm__ P*)calcParamsPtr_->indexList_[i];
        localIndexedShape[i] = calcParamsPtr_->indexedShape_[i];
        localIndexedDimStride[i] = calcParamsPtr_->indexedDimStride_[i];
    }

    const uint32_t subSize = blockDim.x / BC_SUB_GROUPS;
    const uint32_t subId = threadIdx.x / subSize;
    const uint32_t subTid = threadIdx.x - subId * subSize;
    const T2 groupCount = outputLength_ / thirdLoopLength_;
    for (T2 g = blockId_ * BC_SUB_GROUPS + subId; g < groupCount; g = g + blockNums_ * BC_SUB_GROUPS) {
        T2 inputBase = 0;
        T2 secondLoopIdx = g;
        if (formerNonIndexStride_ != 1) {
            T2 firstLoopIdx = AscendC::Simt::UintDiv(g, calcParamsPtr_->m_[INDEX_THREE],
                                                     calcParamsPtr_->shift_[INDEX_THREE]);

            inputBase += firstLoopIdx * formerNonIndexStride_;
            secondLoopIdx = g - firstLoopIdx * bSize_;
        }

        T2 coord[8];
        T2 remIdx = secondLoopIdx;
        for (uint32_t d = 0; d + 1 < bDimNum; ++d) {
            coord[d] = AscendC::Simt::UintDiv(remIdx, calcParamsPtr_->broadcastM_[d],
                                              calcParamsPtr_->broadcastShift_[d]);
            remIdx -= coord[d] * calcParamsPtr_->broadcastDivisor_[d];
        }
        coord[bDimNum - 1] = remIdx;

#pragma unroll
        for (uint16_t j = 0; j < Dim; ++j) {
            T2 localIdx = 0;
            for (uint32_t d = 0; d < bDimNum; ++d) {
                localIdx += coord[d] * calcParamsPtr_->effectiveStride_[j][d];
            }

            int64_t curIdx = localIndexList[j][localIdx];
            if (curIdx < 0) {
                curIdx += localIndexedShape[j];
            }
            inputBase += curIdx * localIndexedDimStride[j];
        }

        const T2 outBase = g * thirdLoopLength_;
        for (T2 t = subTid; t < thirdLoopLength_; t = t + subSize) {
            F()(outputGm_, inputXGm_, outBase + t, inputBase + t);
        }
    }
}

template <typename T, typename F, typename P, int IndexedDim, int NoneIndexedDim, typename T2,
          uint16_t LAUNCH_BOUND_LIMIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(LAUNCH_BOUND_LIMIT) inline void SimtComputeBroadcastNonContinue(
    uint32_t blockId_, T2 outputLength_, uint32_t blockNums_, T2 innerLoopLength_, T2 m3_, T2 shift3_,
    __gm__ T* outputGm_, __gm__ T* inputXGm_, __ubuf__ BroadcastCalcParams<T2>* calcParamsPtr_)
{
    const __gm__ P* localIndexList[IndexedDim];
    T2 localIndexedShape[IndexedDim];
    T2 localIndexedDimStride[IndexedDim];
    T2 localFusedNonIndexedStride[NoneIndexedDim];
    T2 localNonIndexedStride[NoneIndexedDim];
    T2 localFusedM[NoneIndexedDim];
    T2 localFusedShift[NoneIndexedDim];
    const uint32_t bDimNum = calcParamsPtr_->broadcastDimNum_;

#pragma unroll
    for (uint16_t i = 0; i < IndexedDim; ++i) {
        localIndexList[i] = (__gm__ P*)calcParamsPtr_->indexList_[i];
        localIndexedShape[i] = calcParamsPtr_->indexedShape_[i];
        localIndexedDimStride[i] = calcParamsPtr_->indexedDimStride_[i];
    }

#pragma unroll
    for (uint16_t i = 0; i < NoneIndexedDim; ++i) {
        localFusedNonIndexedStride[i] = calcParamsPtr_->fusedNonIndexedStride_[i];
        localNonIndexedStride[i] = calcParamsPtr_->nonIndexedStride_[i];
        localFusedM[i] = calcParamsPtr_->fusedM_[i];
        localFusedShift[i] = calcParamsPtr_->fusedShift_[i];
    }

    for (T2 i = blockId_ * blockDim.x + threadIdx.x; i < outputLength_; i = i + blockNums_ * blockDim.x) {
        T2 outLoopIdx = AscendC::Simt::UintDiv(i, m3_, shift3_);
        T2 inputIndex = 0;
        T2 coord[8];
        T2 remIdx = outLoopIdx;

        for (uint32_t d = 0; d + 1 < bDimNum; ++d) {
            coord[d] = AscendC::Simt::UintDiv(remIdx, calcParamsPtr_->broadcastM_[d],
                                              calcParamsPtr_->broadcastShift_[d]);
            remIdx -= coord[d] * calcParamsPtr_->broadcastDivisor_[d];
        }
        coord[bDimNum - 1] = remIdx;

#pragma unroll
        for (uint16_t j = 0; j < IndexedDim; ++j) {
            T2 localIdx = 0;
            for (uint32_t d = 0; d < bDimNum; ++d) {
                localIdx += coord[d] * calcParamsPtr_->effectiveStride_[j][d];
            }

            int64_t curIdx = localIndexList[j][localIdx];
            if (curIdx < 0) {
                curIdx += localIndexedShape[j];
            }
            inputIndex += curIdx * localIndexedDimStride[j];
        }

        T2 remLength = i - outLoopIdx * innerLoopLength_;
#pragma unroll
        for (uint16_t k = 0; k < NoneIndexedDim; ++k) {
            T2 current_loop_idx = (k == NoneIndexedDim - 1) ?
                                      remLength :
                                      AscendC::Simt::UintDiv(remLength, localFusedM[k], localFusedShift[k]);
            inputIndex += current_loop_idx * localNonIndexedStride[k];
            remLength = remLength - current_loop_idx * localFusedNonIndexedStride[k];
        }
        F()(outputGm_, inputXGm_, i, inputIndex);
    }
}

template <typename T, typename F, typename P, typename T2>
class KernelIndexBroadcast {
public:
    __aicore__ inline KernelIndexBroadcast(){};
    __aicore__ inline void Init(GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides,
                                GM_ADDR indices, IndexBroadcastTilingData tilingData);
    __aicore__ inline void ComputeStrides(IndexBroadcastTilingData tilingData);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessBroadcastContinueLargeTail();
    __aicore__ inline __gm__ P* GetInputTensorAddr(uint16_t index);

private:
    TPipe pipe;
    AscendC::GlobalTensor<T> outputGm_;
    AscendC::GlobalTensor<T> inputXGm_;
    AscendC::GlobalTensor<int64_t> indexedDim_;
    TBuf<TPosition::VECCALC> Buf_;

    GM_ADDR inTensorPtr_ = nullptr;

    uint64_t inputLength_{1};
    uint64_t outputLength_{1};
    uint32_t indexedDimNum_{1};
    uint64_t indexSize_{1};
    uint32_t nonIndexedDimNum_{0};
    uint32_t indexedSizesNum_{0};
    uint32_t indexContinue_{1};
    uint64_t formerNonIndexStride_{1};
    uint32_t threadDim_{1};
    uint32_t smallThreadDim_{1};
    uint32_t blockId_;
    uint32_t blockNums_;
    bool isSmallIndexSize{false};

    T2 thirdLoopLength_{0};
    T2 firstLoopLength_{1};
    T2 secondThirdLoopLength_{0};
    T2 innerLoopLength_{0};

    T2 shift1_{0};
    T2 shift2_{0};
    T2 shift3_{0};
    T2 m1_{0};
    T2 m2_{0};
    T2 m3_{0};

    __ubuf__ BroadcastCalcParams<T2>* calcParamsPtr_;
};

template <typename T, typename F, typename P, typename T2>
__aicore__ inline __gm__ P* KernelIndexBroadcast<T, F, P, T2>::GetInputTensorAddr(uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(inTensorPtr_);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ P*>(*(tensorPtr + index));
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexBroadcast<T, F, P, T2>::ComputeStrides(IndexBroadcastTilingData tilingData)
{
    uint64_t currentStride = 1;
    uint64_t fusedStride = 1;
    int32_t firstIndexedDim = -1;
    uint32_t nonContinueNum = 0;
    uint32_t nonIndexedNum = nonIndexedDimNum_;
    uint32_t indexedNum = tilingData.indexedDimNum;

    for (int i = tilingData.inputDimNum - 1; i >= 0; --i) {
        if (i >= indexedSizesNum_ || !indexedDim_(i)) {
            calcParamsPtr_->nonIndexedStride_[nonIndexedNum - 1] = currentStride;
            calcParamsPtr_->fusedNonIndexedStride_[nonIndexedNum - 1] = fusedStride;
            fusedStride *= tilingData.inputShape[i];
            --nonIndexedNum;
        } else {
            calcParamsPtr_->indexedShape_[indexedNum - 1] = tilingData.inputShape[i];
            calcParamsPtr_->indexedDimStride_[indexedNum - 1] = currentStride;
            --indexedNum;
            firstIndexedDim = i;
            if (i == tilingData.inputDimNum - 1 || (i > indexedSizesNum_) || !indexedDim_(i + 1)) {
                ++nonContinueNum;
            }
        }
        currentStride *= tilingData.inputShape[i];
    }
    if (nonContinueNum > 1) {
        indexContinue_ = 0;
    }

    if (indexContinue_ && (indexedSizesNum_ == 0 || !indexedDim_(0))) {
        formerNonIndexStride_ = calcParamsPtr_->indexedDimStride_[0] * tilingData.inputShape[firstIndexedDim];
    }
    thirdLoopLength_ = outputLength_ / indexSize_;
    if (formerNonIndexStride_ != 1) {
        firstLoopLength_ = inputLength_ / formerNonIndexStride_;
    }
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexBroadcast<T, F, P, T2>::Init(GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes,
                                                               GM_ADDR indexedStrides, GM_ADDR indices,
                                                               IndexBroadcastTilingData tilingData)
{
    inputXGm_.SetGlobalBuffer((__gm__ T*)(inputX), tilingData.inputLength);
    outputGm_.SetGlobalBuffer((__gm__ T*)(output), tilingData.outputLength);
    indexedDim_.SetGlobalBuffer((__gm__ int64_t*)(indexedSizes), tilingData.inputDimNum);
    pipe.InitBuffer(Buf_, sizeof(BroadcastCalcParams<T2>));
    LocalTensor<int64_t> calcParamsUb = Buf_.Get<int64_t>();
    calcParamsPtr_ = (__ubuf__ BroadcastCalcParams<T2>*)calcParamsUb.GetPhyAddr();
    inTensorPtr_ = indices;
    for (size_t i = 0; i < tilingData.indexedDimNum; ++i) {
        calcParamsPtr_->indexList_[i] = (uint64_t)GetInputTensorAddr(i);
    }
    inputLength_ = tilingData.inputLength;
    outputLength_ = tilingData.outputLength;
    indexedDimNum_ = tilingData.indexedDimNum;
    indexSize_ = tilingData.indexSize;
    nonIndexedDimNum_ = tilingData.inputDimNum - tilingData.indexedDimNum;
    indexedSizesNum_ = tilingData.indexedSizesNum;
    this->blockId_ = GetBlockIdx();
    this->blockNums_ = GetBlockNum();

    isSmallIndexSize = indexSize_ < INDEX_THRESHOLD;
    if (isSmallIndexSize) {
        threadDim_ = THREAD_DIM_SMALL;
        smallThreadDim_ = SMALL_THREAD_DIM;
    } else {
        threadDim_ = THREAD_DIM;
        smallThreadDim_ = SMALL_THREAD_DIM_LAUNCH_BOUND;
    }

    ComputeStrides(tilingData);
    thirdLoopLength_ /= firstLoopLength_;
    secondThirdLoopLength_ = outputLength_ / firstLoopLength_;
    innerLoopLength_ = outputLength_ / indexSize_;

    GetUintDivMagicAndShift(m1_, shift1_, secondThirdLoopLength_);
    GetUintDivMagicAndShift(m2_, shift2_, thirdLoopLength_);
    GetUintDivMagicAndShift(m3_, shift3_, innerLoopLength_);
    calcParamsPtr_->shift_[INDEX_ZERO] = shift1_;
    calcParamsPtr_->shift_[INDEX_ONE] = shift2_;
    calcParamsPtr_->shift_[INDEX_TWO] = shift3_;
    calcParamsPtr_->m_[INDEX_ZERO] = m1_;
    calcParamsPtr_->m_[INDEX_ONE] = m2_;
    calcParamsPtr_->m_[INDEX_TWO] = m3_;

    T2 m4;
    T2 shift4;
    GetUintDivMagicAndShift(m4, shift4, static_cast<T2>(indexSize_));
    calcParamsPtr_->m_[INDEX_THREE] = m4;
    calcParamsPtr_->shift_[INDEX_THREE] = shift4;

    for (uint32_t k = 0; k < nonIndexedDimNum_; ++k) {
        T2 fusedM;
        T2 fusedShift;
        GetUintDivMagicAndShift(fusedM, fusedShift, static_cast<T2>(calcParamsPtr_->fusedNonIndexedStride_[k]));
        calcParamsPtr_->fusedM_[k] = fusedM;
        calcParamsPtr_->fusedShift_[k] = fusedShift;
    }

    calcParamsPtr_->broadcastDimNum_ = tilingData.broadcastDimNum;
    T2 cumStride[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t bDim = tilingData.broadcastDimNum;
    for (int32_t d = static_cast<int32_t>(bDim) - 2; d >= 0; --d) {
        cumStride[d] = cumStride[d + 1] * static_cast<T2>(tilingData.broadcastShape[d + 1]);
    }
    for (uint32_t d = 0; d < 8; ++d) {
        T2 m;
        T2 shift;
        T2 divisor = (d < bDim) ? cumStride[d] : 1;
        GetUintDivMagicAndShift(m, shift, divisor);
        calcParamsPtr_->broadcastM_[d] = m;
        calcParamsPtr_->broadcastShift_[d] = shift;
        calcParamsPtr_->broadcastDivisor_[d] = divisor;

        for (uint32_t j = 0; j < tilingData.indexedDimNum; ++j) {
            calcParamsPtr_->effectiveStride_[j][d] = static_cast<T2>(tilingData.indexBcStride[j][d]);
        }
    }
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexBroadcast<T, F, P, T2>::ProcessBroadcastContinueLargeTail()
{
    const T2 bSize = static_cast<T2>(indexSize_);
    if (indexedDimNum_ == DIM_NUMS_ONE) {
        if (isSmallIndexSize) {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_ONE, T2, THREAD_DIM_SMALL>>(
                dim3{threadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_, thirdLoopLength_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                dim3{threadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_, thirdLoopLength_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    } else if (indexedDimNum_ == DIM_NUMS_TWO) {
        if (isSmallIndexSize) {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_TWO, T2, THREAD_DIM_SMALL>>(
                dim3{threadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_, thirdLoopLength_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_TWO, T2, THREAD_DIM>>(
                dim3{threadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_, thirdLoopLength_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    } else if (indexedDimNum_ == DIM_NUMS_THREE) {
        if (isSmallIndexSize) {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_THREE, T2, THREAD_DIM_SMALL>>(
                dim3{threadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_, thirdLoopLength_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_THREE, T2, THREAD_DIM>>(
                dim3{threadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_, thirdLoopLength_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    } else if (indexedDimNum_ == DIM_NUMS_FOUR) {
        if (isSmallIndexSize) {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_FOUR, T2, THREAD_DIM_SMALL>>(
                dim3{threadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_, thirdLoopLength_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_FOUR, T2, THREAD_DIM>>(
                dim3{threadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_, thirdLoopLength_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    } else if (indexedDimNum_ == DIM_NUMS_FIVE) {
        if (isSmallIndexSize) {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_FIVE, T2, SMALL_THREAD_DIM>>(
                dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_,
                thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else {
            asc_vf_call<
                SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_FIVE, T2, SMALL_THREAD_DIM_LAUNCH_BOUND>>(
                dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_,
                thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    } else if (indexedDimNum_ == DIM_NUMS_SIX) {
        if (isSmallIndexSize) {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_SIX, T2, SMALL_THREAD_DIM>>(
                dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_,
                thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else {
            asc_vf_call<
                SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_SIX, T2, SMALL_THREAD_DIM_LAUNCH_BOUND>>(
                dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_,
                thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    } else if (indexedDimNum_ == DIM_NUMS_SEVEN) {
        if (isSmallIndexSize) {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_SEVEN, T2, SMALL_THREAD_DIM>>(
                dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_,
                thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else {
            asc_vf_call<
                SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_SEVEN, T2, SMALL_THREAD_DIM_LAUNCH_BOUND>>(
                dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_,
                thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    } else if (indexedDimNum_ == DIM_NUMS_EIGHT) {
        if (isSmallIndexSize) {
            asc_vf_call<SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_EIGHT, T2, SMALL_THREAD_DIM>>(
                dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_,
                thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else {
            asc_vf_call<
                SimtComputeBroadcastContinueLargeTail<T, F, P, DIM_NUMS_EIGHT, T2, SMALL_THREAD_DIM_LAUNCH_BOUND>>(
                dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, bSize, formerNonIndexStride_,
                thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    }
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexBroadcast<T, F, P, T2>::Process()
{
    DataSyncBarrier<MemDsbT::UB>();
    if (indexContinue_) {
        const uint32_t continueLaunchedDim = (indexedDimNum_ <= DIM_NUMS_FOUR) ? threadDim_ : smallThreadDim_;
        if (thirdLoopLength_ >= static_cast<T2>(continueLaunchedDim) * static_cast<T2>(BC_LARGE_TAIL_FACTOR)) {
            ProcessBroadcastContinueLargeTail();
        } else if (indexedDimNum_ == DIM_NUMS_ONE) {
            if (isSmallIndexSize) {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_ONE, T2, THREAD_DIM_SMALL>>(
                    dim3{threadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            } else {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                    dim3{threadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            }
        } else if (indexedDimNum_ == DIM_NUMS_TWO) {
            if (isSmallIndexSize) {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_TWO, T2, THREAD_DIM_SMALL>>(
                    dim3{threadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            } else {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_TWO, T2, THREAD_DIM>>(
                    dim3{threadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            }
        } else if (indexedDimNum_ == DIM_NUMS_THREE) {
            if (isSmallIndexSize) {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_THREE, T2, THREAD_DIM_SMALL>>(
                    dim3{threadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            } else {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_THREE, T2, THREAD_DIM>>(
                    dim3{threadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            }
        } else if (indexedDimNum_ == DIM_NUMS_FOUR) {
            if (isSmallIndexSize) {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_FOUR, T2, THREAD_DIM_SMALL>>(
                    dim3{threadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            } else {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_FOUR, T2, THREAD_DIM>>(
                    dim3{threadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            }
        } else if (indexedDimNum_ == DIM_NUMS_FIVE) {
            if (isSmallIndexSize) {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_FIVE, T2, SMALL_THREAD_DIM>>(
                    dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            } else {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_FIVE, T2, SMALL_THREAD_DIM_LAUNCH_BOUND>>(
                    dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            }
        } else if (indexedDimNum_ == DIM_NUMS_SIX) {
            if (isSmallIndexSize) {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_SIX, T2, SMALL_THREAD_DIM>>(
                    dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            } else {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_SIX, T2, SMALL_THREAD_DIM_LAUNCH_BOUND>>(
                    dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            }
        } else if (indexedDimNum_ == DIM_NUMS_SEVEN) {
            if (isSmallIndexSize) {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_SEVEN, T2, SMALL_THREAD_DIM>>(
                    dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            } else {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_SEVEN, T2, SMALL_THREAD_DIM_LAUNCH_BOUND>>(
                    dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            }
        } else if (indexedDimNum_ == DIM_NUMS_EIGHT) {
            if (isSmallIndexSize) {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_EIGHT, T2, SMALL_THREAD_DIM>>(
                    dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            } else {
                asc_vf_call<SimtComputeBroadcastContinue<T, F, P, DIM_NUMS_EIGHT, T2, SMALL_THREAD_DIM_LAUNCH_BOUND>>(
                    dim3{smallThreadDim_}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                    formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                    (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
            }
        }
    } else {
        if (indexedDimNum_ == DIM_NUMS_ONE && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_ONE, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_ONE && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_ONE, DIM_NUMS_TWO, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_ONE && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_ONE, DIM_NUMS_THREE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_ONE && nonIndexedDimNum_ == DIM_NUMS_FOUR) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_ONE, DIM_NUMS_FOUR, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_TWO, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_THREE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_FOUR) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_FOUR, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_TWO, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_THREE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_FOUR) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_FOUR, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_FOUR, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_FOUR, DIM_NUMS_TWO, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_FOUR, DIM_NUMS_THREE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR && nonIndexedDimNum_ == DIM_NUMS_FOUR) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_FOUR, DIM_NUMS_FOUR, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FIVE && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_FIVE, DIM_NUMS_TWO, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FIVE && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_FIVE, DIM_NUMS_THREE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_FIVE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_FIVE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_FIVE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_FIVE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_SIX) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_SIX, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_SIX && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_SIX, DIM_NUMS_TWO, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_SIX && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_SIX, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_SEVEN && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_SEVEN, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FIVE && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            asc_vf_call<SimtComputeBroadcastNonContinue<T, F, P, DIM_NUMS_FIVE, DIM_NUMS_ONE, T2, THREAD_DIM>>(
                dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    }
}
} // namespace Index

#endif // ASCENDC_INDEX_BROADCAST_H_
