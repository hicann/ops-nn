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
 * \file index_nocon_broadcast.h
 * \brief Ascendc index kernel for strided view x indices broadcast (only for Index,
 *        IS_NOCON=1 && IS_BROADCAST=1, dim <= 4)
 */

#ifndef ASCENDC_NOCON_BROADCAST_INDEX_SIMT_H_
#define ASCENDC_NOCON_BROADCAST_INDEX_SIMT_H_

#include "index_no_continuous.h"

template <typename T>
struct noConBcCalcParams {
    uint64_t indexBcStride_[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    uint64_t indexList_[4] = {0, 0, 0, 0};
    uint64_t usedIndexInputShape[4] = {0, 0, 0, 0};
    T inputShift_[4];
    T inputM_[4];
    T indexShift_[4];
    T indexM_[4];
    T inputPara[4] = {0, 0, 0, 0};
    T indexPara[4] = {0, 0, 0, 0};
    uint64_t usedIndexStride_[4] = {0, 0, 0, 0};
    uint64_t nonIndexStride_[4] = {0, 0, 0, 0};
};

namespace Index {
using namespace AscendC;

#define NOCON_BC_VF_CALL(IdxCount, BcDim, InputDim)                                                      \
    asc_vf_call<SimtComputeNoConBroadcast<T, F, P, IdxCount, BcDim, InputDim, T2>>(                      \
        dim3{NONCON_THREAD_DIM}, blockId_, outputLength_, blockNums_, (__gm__ T*)outputGm_.GetPhyAddr(), \
        (__gm__ T*)inputXGm_.GetPhyAddr(), calcParamsPtr_)

template <typename T, typename F, typename P, int IdxCount, int BcDim, int InputDim, typename T2>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeNoConBroadcast(
    uint32_t blockId_, uint64_t outputLength_, uint32_t blockNums_, __gm__ T* outputGm_, __gm__ T* inputXGm_,
    __ubuf__ noConBcCalcParams<T2>* calcParamsPtr_)
{
    const __gm__ P* localIndexList[IdxCount];
    uint64_t indexFactor[BcDim];

    for (uint32_t i = 0; i < IdxCount; ++i) {
        localIndexList[i] = (__gm__ P*)calcParamsPtr_->indexList_[i];
    }

    for (T2 i = blockId_ * blockDim.x + threadIdx.x; i < outputLength_; i = i + blockNums_ * blockDim.x) {
        T2 inputIndex = 0;
        T2 length = i;

        for (uint32_t j = 0; j < BcDim; j++) {
            T2 lastIdx = length / calcParamsPtr_->indexPara[j];
            T2 remLength = length - lastIdx * calcParamsPtr_->indexPara[j];
            T2 mulFactor = AscendC::Simt::UintDiv(calcParamsPtr_->indexPara[j], calcParamsPtr_->indexM_[j],
                                                  calcParamsPtr_->indexShift_[j]);
            indexFactor[j] = remLength / mulFactor;
        }

        int64_t indexValue = 0;
        for (uint16_t j = 0; j < IdxCount; j++) {
            T2 idx = 0;
            for (uint32_t k = 0; k < BcDim; k++) {
                idx += indexFactor[k] * calcParamsPtr_->indexBcStride_[j][k];
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
            T2 mulFactor = AscendC::Simt::UintDiv(calcParamsPtr_->inputPara[j], calcParamsPtr_->inputM_[j],
                                                  calcParamsPtr_->inputShift_[j]);
            indexValue = remLength / mulFactor;
            inputIndex += indexValue * calcParamsPtr_->nonIndexStride_[j];
        }

        F()(outputGm_, inputXGm_, i, inputIndex);
    }
}

template <typename T, typename F, typename P, typename T2>
class KernelIndexNoConBroadcast {
public:
    __aicore__ inline KernelIndexNoConBroadcast(){};
    __aicore__ inline void Init(GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides,
                                GM_ADDR indices, IndexNoConBroadcastTilingData tilingData);
    __aicore__ inline void ComputeParameter(IndexNoConBroadcastTilingData tilingData);

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
    uint32_t broadcastDimNum_{1};
    uint32_t inputDimNum_{1};
    uint32_t blockId_;
    uint32_t blockNums_;
    uint32_t firstIndexedDim_{0};
    uint32_t indexedNum_{0};
    uint32_t indexContinue_{1};
    uint32_t indexedSizesNum_{0};

    __ubuf__ noConBcCalcParams<T2>* calcParamsPtr_;
};

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexNoConBroadcast<T, F, P, T2>::Init(GM_ADDR output, GM_ADDR inputX,
                                                                    GM_ADDR indexedSizes, GM_ADDR indexedStrides,
                                                                    GM_ADDR indices,
                                                                    IndexNoConBroadcastTilingData tilingData)
{
    inputXGm_.SetGlobalBuffer((__gm__ T*)(inputX));
    outputGm_.SetGlobalBuffer((__gm__ T*)(output));
    indexedSizeGm_.SetGlobalBuffer((__gm__ int64_t*)(indexedSizes));
    pipe.InitBuffer(Buf_, sizeof(noConBcCalcParams<T2>));
    LocalTensor<int64_t> calcParamsUb = Buf_.Get<int64_t>();
    calcParamsPtr_ = (__ubuf__ noConBcCalcParams<T2>*)calcParamsUb.GetPhyAddr();

    inTensorPtr_ = indices;
    outputLength_ = tilingData.outputLength;
    broadcastDimNum_ = tilingData.broadcastDimNum;
    inputDimNum_ = tilingData.inputDimNum;
    indexedNum_ = tilingData.indexedDimNum;
    indexedSizesNum_ = tilingData.indexedSizesNum;
    blockId_ = GetBlockIdx();
    blockNums_ = GetBlockNum();

    ComputeParameter(tilingData);
    for (size_t i = 0; i < indexedNum_; ++i) {
        calcParamsPtr_->indexList_[i] = (uint64_t)GetInputTensorAddr(i);
    }
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexNoConBroadcast<T, F, P, T2>::ComputeParameter(
    IndexNoConBroadcastTilingData tilingData)
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

    for (uint16_t i = 0; i < broadcastDimNum_; i++) {
        calcParamsPtr_->indexBcStride_[0][i] = tilingData.indexBcStride[0][i];
        calcParamsPtr_->indexBcStride_[1][i] = tilingData.indexBcStride[1][i];
        calcParamsPtr_->indexBcStride_[2][i] = tilingData.indexBcStride[2][i];
        calcParamsPtr_->indexBcStride_[3][i] = tilingData.indexBcStride[3][i];
        factor = tilingData.broadcastShape[i];
        GetUintDivMagicAndShift(m, shift, factor);
        calcParamsPtr_->indexM_[i] = m;
        calcParamsPtr_->indexShift_[i] = shift;
    }

    T2 mulFactor = outputLength_;
    uint32_t idN{0};
    uint32_t nonIdN{0};
    if (indexContinue_ == 0) {
        for (uint16_t i = 0; i < broadcastDimNum_; i++) {
            calcParamsPtr_->indexPara[i] = mulFactor;
            mulFactor /= tilingData.broadcastShape[i];
        }
    }
    for (uint16_t i = 0; i < inputDimNum_; i++) {
        if (indexContinue_ != 0 && i == firstIndexedDim_) {
            for (uint16_t j = 0; j < broadcastDimNum_; j++) {
                calcParamsPtr_->indexPara[j] = mulFactor;
                mulFactor /= tilingData.broadcastShape[j];
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
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline __gm__ P* KernelIndexNoConBroadcast<T, F, P, T2>::GetInputTensorAddr(uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(inTensorPtr_);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> OFFSET);
    return reinterpret_cast<__gm__ P*>(*(tensorPtr + index));
}

template <typename T, typename F, typename P, typename T2>
__aicore__ inline void KernelIndexNoConBroadcast<T, F, P, T2>::Process()
{
    DataSyncBarrier<MemDsbT::UB>();
    if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
        inputDimNum_ == NOCON_DIM_NUMS_ONE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_ONE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_TWO);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_ONE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_ONE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_TWO);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_ONE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_ONE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_TWO);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_ONE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_ONE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_TWO);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_ONE && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_ONE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_TWO);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_TWO);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_TWO);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_TWO) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_TWO);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_TWO && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_TWO, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_THREE) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_THREE);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_THREE && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_THREE, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_FOUR && broadcastDimNum_ == NOCON_DIM_NUMS_ONE &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_FOUR, NOCON_DIM_NUMS_ONE, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_FOUR && broadcastDimNum_ == NOCON_DIM_NUMS_TWO &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_FOUR, NOCON_DIM_NUMS_TWO, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_FOUR && broadcastDimNum_ == NOCON_DIM_NUMS_THREE &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_FOUR, NOCON_DIM_NUMS_THREE, NOCON_DIM_NUMS_FOUR);
    } else if (indexedNum_ == NOCON_COUNT_NUMS_FOUR && broadcastDimNum_ == NOCON_DIM_NUMS_FOUR &&
               inputDimNum_ == NOCON_DIM_NUMS_FOUR) {
        NOCON_BC_VF_CALL(NOCON_COUNT_NUMS_FOUR, NOCON_DIM_NUMS_FOUR, NOCON_DIM_NUMS_FOUR);
    }
}

} // namespace Index

#endif // ASCENDC_NOCON_BROADCAST_INDEX_SIMT_H_
