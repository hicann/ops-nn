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
 * \file index.h
 * \brief Ascendc index kernel
 */

#ifndef ASCENDC_INDEX_SIMT_H_
#define ASCENDC_INDEX_SIMT_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

template <typename T>
struct calcParams {
    uint64_t indexedDimStride_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t nonIndexedStride_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t fusedNonIndexedStride_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    T indexedShape_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t indexList_[8];
    T shift_[3];
    T m_[3];
};

namespace Index {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_DIM = 128;
constexpr uint32_t SMALL_THREAD_DIM = 64;
constexpr uint32_t THREAD_DIM_LAUNCH_BOUND = 512;
constexpr uint32_t SMALL_THREAD_DIM_LAUNCH_BOUND = 256;
#else
constexpr uint32_t THREAD_DIM = 512;
constexpr uint32_t SMALL_THREAD_DIM = 256;
constexpr uint32_t THREAD_DIM_LAUNCH_BOUND = 512;
constexpr uint32_t SMALL_THREAD_DIM_LAUNCH_BOUND = 256;
#endif

constexpr uint32_t DIM_NUMS_ONE = 1;
constexpr uint32_t DIM_NUMS_TWO = 2;
constexpr uint32_t DIM_NUMS_THREE = 3;
constexpr uint32_t DIM_NUMS_FOUR = 4;
constexpr uint32_t DIM_NUMS_FIVE = 5;
constexpr uint32_t DIM_NUMS_SIX = 6;
constexpr uint32_t DIM_NUMS_SEVEN = 7;
constexpr uint32_t DIM_NUMS_EIGHT = 8;
constexpr uint32_t INDEX_ZERO = 0;
constexpr uint32_t INDEX_ONE = 1;
constexpr uint32_t INDEX_TWO = 2;

typedef int32_t int4 __attribute__((ext_vector_type(4)));

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

// index assignment function
template <typename T, typename T2>
struct IndexAssign {
    __aicore__ inline IndexAssign()
    {}
    __aicore__ inline void operator()(__gm__ T* output, __gm__ T* input, T2 i, T2 idx)
    {
        output[i] = input[idx];
    }
};

// indexPut assignment function
template <typename T>
struct IndexPutAssign {
    __aicore__ inline IndexPutAssign()
    {}
    __aicore__ inline void operator()(__gm__ T* output, __gm__ T* input, uint32_t i, uint32_t idx)
    {
        output[idx] = input[i];
    }
};

// indexPut add function for accumulate mode
template <typename T>
struct IndexPutAdd {
    __aicore__ inline IndexPutAdd()
    {}
    __aicore__ inline void operator()(__gm__ T* output, __gm__ T* input, uint32_t i, uint32_t idx)
    {
        if constexpr (is_same<bool, T>::value) {
            if(input[i]){
                output[idx] = true;
            }
        } else {
            Simt::AtomicAdd(output + idx, input[i]);
        }
    }
};

/*
occasion 1: those indexed dimensions are continuous
    x shape = (a, b, c, d)
    only 2 dims in the middle are indexed
        Index1 shape: (e, f)
        Index2 shape: (e, f)
    y = x[:, index1, index2, :]
    output shape （a, e, f, d)

    stride1 = c*d

compute logic：
    for (n = 0; n < a; ++n) {
        for (i = 0; i < e*f; ++i) {
            for (j = 0; j < d; ++j) {
                y[n * e*f*d + i * d + j] = x[n * b*c*d + index1[i] * stride1 + index2[i] * stride2 + j]
            }
        }
    }
*/
template <typename T, typename F, typename P, int Dim, typename T2>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeContinue(
    uint32_t blockId_, uint64_t outputLength_, uint32_t blockNums_, uint32_t secondThirdLoopLength_,
    uint64_t formerNonIndexStride_, uint32_t thirdLoopLength_, __gm__ T* outputGm_, __gm__ T* inputXGm_,
    __ubuf__ calcParams<T2>* calcParamsPtr_)
{
    const __gm__ P* localIndexList[Dim];
    T2 localIndexedShape[Dim];
    int64_t localIndexedDimStride[Dim];

#pragma unroll
    for (uint32_t i = 0; i < Dim; ++i) {
        localIndexList[i] = (__gm__ P*)calcParamsPtr_->indexList_[i];
        localIndexedShape[i] = calcParamsPtr_->indexedShape_[i];
        localIndexedDimStride[i] = calcParamsPtr_->indexedDimStride_[i];
    }

    for (T2 i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outputLength_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        T2 inputIndex = 0;
        T2 remLength = i;
        if (formerNonIndexStride_ != 1) {
            // quick div for i / secondThirdLoopLength_
            T2 firstLoopIdx = AscendC::Simt::UintDiv(remLength, calcParamsPtr_->m_[0], calcParamsPtr_->shift_[0]);

            inputIndex += firstLoopIdx * formerNonIndexStride_;
            remLength = remLength - firstLoopIdx * secondThirdLoopLength_;
        }

        // quick div for remLength / thirdLoopLength_
        T2 secondLoopIdx = AscendC::Simt::UintDiv(remLength, calcParamsPtr_->m_[1], calcParamsPtr_->shift_[1]);

        T2 thirdLoopIdx = remLength - thirdLoopLength_ * secondLoopIdx;

#pragma unroll
        for (uint32_t j = 0; j < Dim; ++j) {
            int64_t curIdx = localIndexList[j][secondLoopIdx];
            if (curIdx < 0) {
                curIdx += localIndexedShape[j];
            }
            inputIndex += curIdx * localIndexedDimStride[j];
        }
        inputIndex += thirdLoopIdx;
        F()(outputGm_, inputXGm_, i, inputIndex);
    }
}

/*
occasion 2: those indexed dimensions are not continuous
    x shape = (a, b, c, d)
    the first and the third dim are indexed
        Index1 shape: (e, f)
        Index2 shape: (e, f)
    y = x[index1, :, index2, :]
    output shape （e, f, b, d)

compute logic：
    for (i = 0; i < e*f; ++i) {
        for (j = 0; j < b; ++j) {
            for (k = 0; k < d; ++) {
                y[i * b*d + j * d + k] = x[index1[i] * b*c*d + j * c*d + index2[i] * d + k]
            }
        }
    }
*/
template <typename T, typename F, typename P, int IndexedDim, int NoneIndexedDim, typename T2>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeNonContinue(
    uint32_t blockId_, uint64_t outputLength_, uint32_t blockNums_, T2 innerLoopLength_, T2 m3_, T2 shift3_,
    __gm__ T* outputGm_, __gm__ T* inputXGm_, __ubuf__ calcParams<T2>* calcParamsPtr_)
{
    const __gm__ P* localIndexList[IndexedDim];
    T2 localIndexedShape[IndexedDim];
    T2 localIndexedDimStride[IndexedDim];
    T2 localFusedNonIndexedStride[NoneIndexedDim];
    T2 localNonIndexedStride[NoneIndexedDim];
#pragma unroll
    for (uint32_t i = 0; i < IndexedDim; ++i) {
        localIndexList[i] = (__gm__ P*)calcParamsPtr_->indexList_[i];
        localIndexedShape[i] = calcParamsPtr_->indexedShape_[i];
        localIndexedDimStride[i] = calcParamsPtr_->indexedDimStride_[i];
    }

#pragma unroll
    for (uint32_t i = 0; i < NoneIndexedDim; ++i) {
        localFusedNonIndexedStride[i] = calcParamsPtr_->fusedNonIndexedStride_[i];
        localNonIndexedStride[i] = calcParamsPtr_->nonIndexedStride_[i];
    }

    for (T2 i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outputLength_;
         i = i + blockNums_ * Simt::GetThreadNum()) {
        // quick div for i / innerLoopLength_
        T2 outLoopIdx = AscendC::Simt::UintDiv(i, m3_, shift3_);

        T2 inputIndex = 0;

#pragma unroll
        for (uint16_t j = 0; j < IndexedDim; ++j) {
            int64_t curIdx = localIndexList[j][outLoopIdx];
            if (curIdx < 0) {
                curIdx += localIndexedShape[j];
            }
            inputIndex += curIdx * localIndexedDimStride[j];
        }

        T2 remLength = i - outLoopIdx * innerLoopLength_;
#pragma unroll
        for (uint16_t k = 0; k < NoneIndexedDim; ++k) {
            T2 current_loop_idx = remLength / localFusedNonIndexedStride[k];
            inputIndex += current_loop_idx * localNonIndexedStride[k];
            remLength = remLength - current_loop_idx * localFusedNonIndexedStride[k];
        }
        F()(outputGm_, inputXGm_, i, inputIndex);
    }
}

template <typename T, typename F, typename P, typename T2>
class KernelIndex {
public:
    __aicore__ inline KernelIndex(){};
    __aicore__ inline void Init(
        GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
        IndexSimtTilingData tilingData, GM_ADDR value = nullptr);
    __aicore__ inline void ComputeStrides(IndexSimtTilingData tilingData);

    __aicore__ inline void Process();
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
    uint32_t blockId_;
    uint32_t blockNums_;

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

    bool isIndexPut_{false};
    bool accumulateMode_{false};
    __ubuf__ calcParams<T2>* calcParamsPtr_;
};

template <typename T, typename F, typename P, typename T2>
__aicore__ inline __gm__ P* KernelIndex<T, F, P, T2>::GetInputTensorAddr(uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(inTensorPtr_);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ P*>(*(tensorPtr + index));
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndex<T, F, P, T2>::ComputeStrides(IndexSimtTilingData tilingData)
{
    // cpmpute different kinds of strides
    uint64_t currentStride = 1;
    uint64_t fusedStride = 1;
    int32_t firstIndexedDim = -1;
    uint32_t nonContinueNum = 0;
    uint32_t nonIndexedNum = nonIndexedDimNum_;
    uint32_t indexedNum = tilingData.indexedDimNum;

    for (int i = tilingData.inputDimNum - 1; i >= 0; --i) {
        if (i >= indexedSizesNum_ || !indexedDim_(i)) {
            calcParamsPtr_->nonIndexedStride_[nonIndexedNum - 1] = currentStride;
            // the strides of the new tensor formed by those dim that are not indexed
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

    // for the continuous occasion whose highest dimension is not indexed, acquire those dimensions' lowest strides
    if (indexContinue_ && (indexedSizesNum_ == 0 || !indexedDim_(0))) {
        formerNonIndexStride_ = calcParamsPtr_->indexedDimStride_[0] * tilingData.inputShape[firstIndexedDim];
    }
    thirdLoopLength_ = outputLength_ / indexSize_;
    // the highest dim is not indexed, update the first loop length and second loop length
    if (formerNonIndexStride_ != 1) {
        firstLoopLength_ = inputLength_ / formerNonIndexStride_;
    }
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndex<T, F, P, T2>::Init(
    GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
    IndexSimtTilingData tilingData, GM_ADDR value)
{
    if (value != nullptr) {
        isIndexPut_ = true;
        accumulateMode_ = tilingData.accumulateMode;
        inputXGm_.SetGlobalBuffer((__gm__ T*)(value), tilingData.outputLength);
        outputGm_.SetGlobalBuffer((__gm__ T*)(output), tilingData.inputLength);
    } else {
        inputXGm_.SetGlobalBuffer((__gm__ T*)(inputX), tilingData.inputLength);
        outputGm_.SetGlobalBuffer((__gm__ T*)(output), tilingData.outputLength);
    }
    indexedDim_.SetGlobalBuffer((__gm__ int64_t*)(indexedSizes), tilingData.inputDimNum);
    pipe.InitBuffer(Buf_, sizeof(calcParams<T2>));
    LocalTensor<int64_t> calcParamsUb = Buf_.Get<int64_t>();
    calcParamsPtr_ = (__ubuf__ calcParams<T2>*)calcParamsUb.GetPhyAddr();
    // get dynamci input tensor
    inTensorPtr_ = indices;
    for (size_t i = 0; i < tilingData.indexedDimNum; ++i) {
        // indexList_[i].SetGlobalBuffer((__gm__ P *)(GetInputTensorAddr(i)), tilingData.indexSize);
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
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndex<T, F, P, T2>::Process()
{
    DataSyncBarrier<MemDsbT::UB>();
    if (indexContinue_) {
        if (indexedDimNum_ == DIM_NUMS_ONE) {
            AscendC::Simt::VF_CALL<SimtComputeContinue<T, F, P, DIM_NUMS_ONE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO) {
            AscendC::Simt::VF_CALL<SimtComputeContinue<T, F, P, DIM_NUMS_TWO, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE) {
            AscendC::Simt::VF_CALL<SimtComputeContinue<T, F, P, DIM_NUMS_THREE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR) {
            AscendC::Simt::VF_CALL<SimtComputeContinue<T, F, P, DIM_NUMS_FOUR, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, secondThirdLoopLength_,
                formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FIVE) {
            AscendC::Simt::VF_CALL<SimtComputeContinue<T, F, P, DIM_NUMS_FIVE, T2>>(
                AscendC::Simt::Dim3{SMALL_THREAD_DIM_LAUNCH_BOUND}, blockId_, outputLength_, blockNums_,
                secondThirdLoopLength_, formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_SIX) {
            AscendC::Simt::VF_CALL<SimtComputeContinue<T, F, P, DIM_NUMS_SIX, T2>>(
                AscendC::Simt::Dim3{SMALL_THREAD_DIM_LAUNCH_BOUND}, blockId_, outputLength_, blockNums_,
                secondThirdLoopLength_, formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_SEVEN) {
            AscendC::Simt::VF_CALL<SimtComputeContinue<T, F, P, DIM_NUMS_SEVEN, T2>>(
                AscendC::Simt::Dim3{SMALL_THREAD_DIM_LAUNCH_BOUND}, blockId_, outputLength_, blockNums_,
                secondThirdLoopLength_, formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_EIGHT) {
            AscendC::Simt::VF_CALL<SimtComputeContinue<T, F, P, DIM_NUMS_EIGHT, T2>>(
                AscendC::Simt::Dim3{SMALL_THREAD_DIM_LAUNCH_BOUND}, blockId_, outputLength_, blockNums_,
                secondThirdLoopLength_, formerNonIndexStride_, thirdLoopLength_, (__gm__ T*)outputGm_.GetPhyAddr(),
                (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    } else {
        if (indexedDimNum_ == DIM_NUMS_ONE && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_ONE, DIM_NUMS_ONE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_ONE && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_ONE, DIM_NUMS_TWO, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_ONE && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_ONE, DIM_NUMS_THREE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_ONE && nonIndexedDimNum_ == DIM_NUMS_FOUR) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_ONE, DIM_NUMS_FOUR, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_ONE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_TWO, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_THREE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_FOUR) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_FOUR, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_ONE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_TWO, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_THREE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_FOUR) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_FOUR, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_FOUR, DIM_NUMS_ONE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_FOUR, DIM_NUMS_TWO, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_FOUR, DIM_NUMS_THREE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FOUR && nonIndexedDimNum_ == DIM_NUMS_FOUR) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_FOUR, DIM_NUMS_FOUR, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FIVE && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            // add missing non-continuous scenarios
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_FIVE, DIM_NUMS_TWO, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FIVE && nonIndexedDimNum_ == DIM_NUMS_THREE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_FIVE, DIM_NUMS_THREE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_THREE && nonIndexedDimNum_ == DIM_NUMS_FIVE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_THREE, DIM_NUMS_FIVE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_FIVE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_FIVE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_TWO && nonIndexedDimNum_ == DIM_NUMS_SIX) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_TWO, DIM_NUMS_SIX, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_SIX && nonIndexedDimNum_ == DIM_NUMS_TWO) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_SIX, DIM_NUMS_TWO, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_SIX && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_SIX, DIM_NUMS_ONE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_SEVEN && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_SEVEN, DIM_NUMS_ONE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        } else if (indexedDimNum_ == DIM_NUMS_FIVE && nonIndexedDimNum_ == DIM_NUMS_ONE) {
            AscendC::Simt::VF_CALL<SimtComputeNonContinue<T, F, P, DIM_NUMS_FIVE, DIM_NUMS_ONE, T2>>(
                AscendC::Simt::Dim3{THREAD_DIM}, blockId_, outputLength_, blockNums_, innerLoopLength_, m3_, shift3_,
                (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
        }
    }
}
} // namespace Index

#endif // ASCENDC_INDEX_SIMT_H_