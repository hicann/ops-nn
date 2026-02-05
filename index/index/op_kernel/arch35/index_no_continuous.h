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
 * \file index_no_continuous.h
 * \brief Ascendc index kernel
 */

#ifndef ASCENDC_NOCONTINUOUS_INDEX_SIMT_H_
#define ASCENDC_NOCONTINUOUS_INDEX_SIMT_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"


template <typename T>
struct noConcalcParams {
    uint64_t indexedDimStride_[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0},
                                        {0, 0, 0, 0}, {0, 0, 0, 0}};
    uint64_t indexList_[4] = {0, 0, 0, 0};
    uint64_t usedIndexInputShape[4] = {0, 0, 0, 0};
    T inputShift_[4];
    T inputM_[4];
    T indexShift_[4];
    T indexM_[4];
    T valueShift_[8];
    T valueM_[8];
    T inputPara[4] = {0, 0, 0, 0};
    T indexPara[4] = {0, 0, 0, 0};
    uint64_t usedIndexStride_[4] = {0, 0, 0, 0};
    uint64_t nonIndexStride_[4] = {0, 0, 0, 0};
    uint64_t valueStride_[8] = {0, 0, 0, 0};
    uint64_t valueViewStride_[8] = {0, 0, 0, 0};
    bool isIndexPut_{false};
    int32_t valueDimNum_{0};
};

namespace Index {
using namespace AscendC;

constexpr uint32_t NOCON_DIM_NUMS_ONE = 1;
constexpr uint32_t NOCON_DIM_NUMS_TWO = 2;
constexpr uint32_t NOCON_DIM_NUMS_THREE = 3;
constexpr uint32_t NOCON_DIM_NUMS_FOUR = 4;
constexpr uint32_t NOCON_COUNT_NUMS_ONE = 1;
constexpr uint32_t NOCON_COUNT_NUMS_TWO = 2;
constexpr uint32_t NOCON_COUNT_NUMS_THREE = 3;
constexpr uint32_t NOCON_COUNT_NUMS_FOUR = 4;
constexpr uint32_t OFFSET = 3;
constexpr uint32_t NONCON_THREAD_DIM = 256;

template <typename T, typename F, typename P, int IdxCount, int IdxDim, int InputDim, typename T2>     
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeNoContinuous(
    uint32_t blockId_, uint64_t outputLength_, uint32_t blockNums_, uint32_t firstIndexedDim_, __gm__ T* outputGm_, 
    __gm__ T* inputXGm_, __ubuf__ noConcalcParams<T2> *calcParamsPtr_)
{
    const __gm__ P* localIndexList[IdxCount];
    uint64_t indexFactor[IdxDim];

    for (uint32_t i = 0; i < IdxCount; ++i) {
        localIndexList[i] = (__gm__ P*)calcParamsPtr_->indexList_[i];
    }

    for (T2 i = blockId_ * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outputLength_;
        i = i + blockNums_ * Simt::GetThreadNum()) {
            T2 inputIndex = 0;
            T2 length = i;
            T2 index = calcParamsPtr_-> isIndexPut_ == false ? i : 0;

            for (uint32_t j = 0; j < IdxDim; j++) {
                T2 lastIdx = length / calcParamsPtr_->indexPara[j];
                T2 remLength = length - lastIdx * calcParamsPtr_->indexPara[j];
                T2 mulFactor = AscendC::Simt::UintDiv(calcParamsPtr_->indexPara[j], calcParamsPtr_->indexM_[j], calcParamsPtr_->indexShift_[j]);
                indexFactor[j] = remLength / mulFactor;
            }

            int64_t indexValue = 0;
            for (uint16_t j = 0; j < IdxCount; j++) {
                T2 idx = 0;
                for (uint32_t k = 0; k < IdxDim; k++) {
                    idx += indexFactor[k] * calcParamsPtr_->indexedDimStride_[j][k];
                }
                indexValue = localIndexList[j][idx];
                if (indexValue < 0) {
                    indexValue += calcParamsPtr_->usedIndexInputShape[j];
                }
                inputIndex += indexValue * calcParamsPtr_->usedIndexStride_[j];
            }
            for (int32_t j = 0; j < InputDim - IdxCount; j++) {
                T2 lastIdx = length / calcParamsPtr_->inputPara[j];
                T2 remLength = length - lastIdx * calcParamsPtr_->inputPara[j];
                T2 mulFactor = AscendC::Simt::UintDiv(calcParamsPtr_->inputPara[j], calcParamsPtr_->inputM_[j], calcParamsPtr_->inputShift_[j]);
                indexValue = remLength / mulFactor;
                inputIndex += indexValue * calcParamsPtr_->nonIndexStride_[j];
            }
            
            if (calcParamsPtr_-> isIndexPut_) {
                for (uint32_t j = 0; j < calcParamsPtr_->valueDimNum_; j++) {
                    T2 valueFactor = AscendC::Simt::UintDiv(length, calcParamsPtr_->valueM_[j], calcParamsPtr_->valueShift_[j]);
                    length = length - calcParamsPtr_->valueViewStride_[j] * valueFactor;
                    index += valueFactor * calcParamsPtr_->valueStride_[j];
                }
            }
            F()(outputGm_, inputXGm_, index, inputIndex);
        }
}

template <typename T, typename F, typename P, typename T2>
class KernelIndexNoContiguous {
public:
    __aicore__ inline KernelIndexNoContiguous(){};
    __aicore__ inline void Init(
        GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
        IndexNonContinuousTilingData tilingData, GM_ADDR value = nullptr);
    __aicore__ inline void ComputeParameter(IndexNonContinuousTilingData tilingData);

    __aicore__ inline void Process();
    __aicore__ inline __gm__ P* GetInputTensorAddr(uint16_t index);

private:
    TPipe pipe;
    AscendC::GlobalTensor<T> outputGm_;
    AscendC::GlobalTensor<T> inputXGm_;
    AscendC::GlobalTensor<int64_t> indexedSizeGm_;
    TBuf<TPosition::VECCALC> Buf_;  

    GM_ADDR inTensorPtr_ = nullptr;

    uint64_t outputLength_{1};
    uint32_t indexedDimNum_{1};
    uint32_t inputDimNum_{1};
    uint32_t valueDimNum_{1};
    uint32_t blockId_;
    uint32_t blockNums_;    
    uint32_t firstIndexedDim_{0};
    uint32_t indexedNum_{0};
    uint32_t indexContinue_{1};
    uint32_t indexedSizesNum_{0};
    bool isIndexPut_{false};

    __ubuf__ noConcalcParams<T2> *calcParamsPtr_;
};

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexNoContiguous<T, F, P, T2>::Init(
    GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
    IndexNonContinuousTilingData tilingData, GM_ADDR value)
{
    if (value != nullptr) {
        isIndexPut_ = true;
        inputXGm_.SetGlobalBuffer((__gm__ T*)(value));
        outputGm_.SetGlobalBuffer((__gm__ T*)(output));
    } else {
        inputXGm_.SetGlobalBuffer((__gm__ T*)(inputX));
        outputGm_.SetGlobalBuffer((__gm__ T*)(output));
    }
    indexedSizeGm_.SetGlobalBuffer((__gm__ int64_t*)(indexedSizes));
    pipe.InitBuffer(Buf_, sizeof(noConcalcParams<T2>));
    LocalTensor<int64_t> calcParamsUb = Buf_.Get<int64_t>();
    calcParamsPtr_ = (__ubuf__ noConcalcParams<T2>*)calcParamsUb.GetPhyAddr();

    calcParamsPtr_->isIndexPut_ = false;
    inTensorPtr_ = indices;
    outputLength_ = tilingData.outputLength;
    indexedDimNum_ = tilingData.indexedDimNum;
    inputDimNum_ = tilingData.inputDimNum;
    indexedNum_ = tilingData.indexSize;
    indexedSizesNum_ = tilingData.indexedSizesNum;
    valueDimNum_ = tilingData.valueDimNum;
    blockId_ = GetBlockIdx();
    blockNums_ = GetBlockNum();

    ComputeParameter(tilingData);
    for (size_t i = 0; i < indexedNum_; ++i) {
        calcParamsPtr_->indexList_[i] = (uint64_t)GetInputTensorAddr(i);
    }
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexNoContiguous<T, F, P, T2>::ComputeParameter(IndexNonContinuousTilingData tilingData)
{
    T2 m;
    T2 shift;
    T2 factor;
    uint32_t nonContinueNum = 0;

    for (int32_t i = inputDimNum_ - 1; i >= 0; --i) {
        if (i < indexedSizesNum_ && indexedSizeGm_(i) != 0) {
            firstIndexedDim_ = i;
            if (i == tilingData.inputDimNum - 1 || (i > indexedSizesNum_) || !indexedSizeGm_(i + 1)) {
                ++nonContinueNum;
            }
        }
    }

    if (nonContinueNum > 1) {
        indexContinue_ = 0;
    }

    for (uint16_t i = 0; i < indexedDimNum_; i++) {
        calcParamsPtr_->indexedDimStride_[0][i] = tilingData.indexStride1[i];
        calcParamsPtr_->indexedDimStride_[1][i] = tilingData.indexStride2[i];
        calcParamsPtr_->indexedDimStride_[2][i] = tilingData.indexStride3[i];
        calcParamsPtr_->indexedDimStride_[3][i] = tilingData.indexStride4[i];
        factor = tilingData.indexShape[i];
        GetUintDivMagicAndShift(m, shift, factor);
        calcParamsPtr_->indexM_[i] = m;
        calcParamsPtr_->indexShift_[i] = shift;
    }
 
    T2 mulFactor = outputLength_;
    uint32_t idN{0};
    uint32_t nonIdN{0};
    if (indexContinue_ == 0) {
        for (uint16_t i = 0; i < indexedDimNum_; i++) {
            calcParamsPtr_->indexPara[i] = mulFactor;
            mulFactor /= tilingData.indexShape[i];
        }
    }
    for (uint16_t i = 0; i < inputDimNum_; i++) {
        if (indexContinue_ != 0 && i == firstIndexedDim_) {
            for (uint16_t j = 0; j < indexedDimNum_; j++) {
                calcParamsPtr_->indexPara[j] = mulFactor;
                mulFactor /= tilingData.indexShape[j];
            }
        }
        if (i >= indexedSizesNum_ || indexedSizeGm_(i) == 0) {
            calcParamsPtr_->inputPara[nonIdN] = mulFactor;
            mulFactor /= tilingData.xShape[i];
            factor = tilingData.xShape[i];
            GetUintDivMagicAndShift(m, shift, factor);
            calcParamsPtr_->inputM_[nonIdN] = m;
            calcParamsPtr_->inputShift_[nonIdN] = shift;
            calcParamsPtr_->nonIndexStride_[nonIdN] = tilingData.xStride[i];
            nonIdN++;
        } else {
            calcParamsPtr_->usedIndexStride_[idN] = tilingData.xStride[i];
            calcParamsPtr_->usedIndexInputShape[idN] = tilingData.xShape[i];
            idN++;
        }
    }

    if (isIndexPut_) {
        T2 viewStride = 1;
        calcParamsPtr_->isIndexPut_ = true;
        calcParamsPtr_->valueDimNum_ = valueDimNum_;
        for (int32_t i = valueDimNum_ - 1; i >= 0; --i) {
            factor = viewStride;
            calcParamsPtr_->valueViewStride_[i] = viewStride;
            GetUintDivMagicAndShift(m, shift, factor);
            calcParamsPtr_->valueM_[i] = m;
            calcParamsPtr_->valueShift_[i] = shift;
            viewStride *= tilingData.valueShape[i];
            calcParamsPtr_->valueStride_[i] = tilingData.valueStride[i];
        }
    }
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline __gm__ P* KernelIndexNoContiguous<T, F, P, T2>::GetInputTensorAddr(uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(inTensorPtr_);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> OFFSET);
    return reinterpret_cast<__gm__ P*>(*(tensorPtr + index));
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexNoContiguous<T, F, P, T2>::Process()
{
    DataSyncBarrier<MemDsbT::UB>();
    if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_ONE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_ONE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);        
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_TWO, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);   
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_ONE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_ONE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_TWO, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_ONE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_ONE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_TWO, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_ONE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_ONE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_TWO, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);    
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_TWO, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);    
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_TWO, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_TWO, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_TWO, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_THREE, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_FOUR && indexedDimNum_ == NOCON_DIM_NUMS_ONE && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_FOUR, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_FOUR && indexedDimNum_ == NOCON_DIM_NUMS_TWO && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_FOUR, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_FOUR && indexedDimNum_ == NOCON_DIM_NUMS_THREE && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_FOUR, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_FOUR && indexedDimNum_ == NOCON_DIM_NUMS_FOUR && inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        AscendC::Simt::VF_CALL<SimtComputeNoContinuous<T, F, P, NOCON_COUNT_NUMS_FOUR, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_FOUR, T2>>(
            AscendC::Simt::Dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, firstIndexedDim_,
            (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_);
    }
}

}

#endif