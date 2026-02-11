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
 * \file index_perf_no_continuous.h
 * \brief
 */

#ifndef INDEX_PERF_NO_CONTINUOUS_H_
#define INDEX_PERF_NO_CONTINUOUS_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "kernel_operator_list_tensor_intf.h"

struct SimtParams {
    uint64_t indexedStride[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0},
                                        {0, 0, 0, 0}, {0, 0, 0, 0}};
    int64_t indexShape[4] = {0, 0, 0, 0};
    int64_t xShape[4] = {0, 0, 0, 0};
    int64_t xStride[4] = {0, 0, 0, 0};
};

namespace Index {
using namespace AscendC;

constexpr uint32_t NOCON_IDX_MAX_DIM_NUM = 4;
constexpr uint32_t INPUTDIMNUMTWO = 2;
constexpr uint32_t INPUTDIMNUMTHREE = 3;
constexpr uint32_t INPUTDIMNUMFOUR = 4;
constexpr uint32_t INDEX_DIM_NUM_ONE = 1;
constexpr uint32_t INDEX_DIM_NUM_TWO = 2;
constexpr uint32_t INDEX_DIM_NUM_THREE = 3;
constexpr uint32_t INDEX_DIM_NUM_FOUR = 4;
constexpr uint32_t LIST_INDEX_ZERO = 0;
constexpr uint32_t LIST_INDEX_ONE = 1;
constexpr uint32_t LIST_INDEX_TWO = 2;
constexpr uint32_t LIST_INDEX_THREE = 3;

template <typename T, typename P, typename T2, int IdxDim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_LAUNCH_BOUND) inline void SimtDim2Compute(
    __gm__ T* outputGm_, __gm__ T* inputXGm_, __gm__ P* indexTensor0, __gm__ P* indexTensor1,
    __ubuf__ SimtParams *simtParamsPtr_, uint64_t outputLength)
{
    for (T2 i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outputLength;
         i = i + Simt::GetBlockNum() * Simt::GetThreadNum()) {
        T2 indexDim[NOCON_IDX_MAX_DIM_NUM] = {0, 0, 0, 0};
        T2 offset = i;
        T2 inputIndex = 0;
        for (uint32_t j = 0; j < IdxDim; j++) {
            T2 stride = 1;
            for (uint32_t k = j + 1; k < IdxDim; k++) {
                stride *= simtParamsPtr_->indexShape[k];
            }
            indexDim[j] = offset / stride;
            offset = offset - indexDim[j] * stride;
        }

        T2 idx1 = 0;
        T2 idx2 = 0;
        for (uint32_t j = 0; j < IdxDim; j++) {
            idx1 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_ZERO][j];
        }
        int64_t indexValue1 = indexTensor0[idx1];
        if (indexValue1 < 0) {
            indexValue1 += simtParamsPtr_->xShape[LIST_INDEX_ZERO];
        }

        for (uint32_t j = 0; j < IdxDim; j++) {
            idx2 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_ONE][j];
        }
        int64_t indexValue2 = indexTensor1[idx2];
        if (indexValue2 < 0) {
            indexValue2 += simtParamsPtr_->xShape[LIST_INDEX_ONE];
        }

        inputIndex = indexValue1 * simtParamsPtr_->xStride[LIST_INDEX_ZERO] + indexValue2 * simtParamsPtr_->xStride[LIST_INDEX_ONE];
        outputGm_[i] = inputXGm_[inputIndex];
    }
}

template <typename T, typename P, typename T2, int IdxDim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_LAUNCH_BOUND) inline void SimtDim3Compute(
    __gm__ T* outputGm_, __gm__ T* inputXGm_, __gm__ P* indexTensor0, __gm__ P* indexTensor1,
    __gm__ P* indexTensor2, __ubuf__ SimtParams *simtParamsPtr_, uint64_t outputLength)
{
    for (T2 i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outputLength;
         i = i + Simt::GetBlockNum() * Simt::GetThreadNum()) {
        T2 indexDim[NOCON_IDX_MAX_DIM_NUM] = {0, 0, 0, 0};
        T2 offset = i;
        T2 inputIndex = 0;
        for (uint32_t j = 0; j < IdxDim; j++) {
            T2 stride = 1;
            for (uint32_t k = j + 1; k < IdxDim; k++) {
                stride *= simtParamsPtr_->indexShape[k];
            }
            indexDim[j] = offset / stride;
            offset = offset - indexDim[j] * stride;
        }

        T2 idx1 = 0;
        T2 idx2 = 0;
        T2 idx3 = 0;
        for (uint32_t j = 0; j < IdxDim; j++) {
            idx1 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_ZERO][j];
        }
        int64_t indexValue1 = indexTensor0[idx1];
        if (indexValue1 < 0) {
            indexValue1 += simtParamsPtr_->xShape[LIST_INDEX_ZERO];
        }

        for (uint32_t j = 0; j < IdxDim; j++) {
            idx2 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_ONE][j];
        }
        int64_t indexValue2 = indexTensor1[idx2];
        if (indexValue2 < 0) {
            indexValue2 += simtParamsPtr_->xShape[LIST_INDEX_ONE];
        }

        for (uint32_t j = 0; j < IdxDim; j++) {
            idx3 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_TWO][j];
        }
        int64_t indexValue3 = indexTensor2[idx3];
        if (indexValue3 < 0) {
            indexValue3 += simtParamsPtr_->xShape[LIST_INDEX_TWO];
        }

        inputIndex = indexValue1 * simtParamsPtr_->xStride[LIST_INDEX_ZERO] + indexValue2 * simtParamsPtr_->xStride[LIST_INDEX_ONE] +
                     indexValue3 * simtParamsPtr_->xStride[LIST_INDEX_TWO];
        outputGm_[i] = inputXGm_[inputIndex];
    }
}

template <typename T, typename P, typename T2, int IdxDim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_LAUNCH_BOUND) inline void SimtDim4Compute(
    __gm__ T* outputGm_, __gm__ T* inputXGm_, __gm__ P* indexTensor0, __gm__ P* indexTensor1,
    __gm__ P* indexTensor2, __gm__ P* indexTensor3, __ubuf__ SimtParams *simtParamsPtr_,
    uint64_t outputLength)
{
    for (T2 i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outputLength;
         i = i + Simt::GetBlockNum() * Simt::GetThreadNum()) {
        T2 indexDim[NOCON_IDX_MAX_DIM_NUM] = {0, 0, 0, 0};
        T2 offset = i;
        T2 inputIndex = 0;
        for (uint32_t j = 0; j < IdxDim; j++) {
            T2 stride = 1;
            for (uint32_t k = j + 1; k < IdxDim; k++) {
                stride *= simtParamsPtr_->indexShape[k];
            }
            indexDim[j] = offset / stride;
            offset = offset - indexDim[j] * stride;
        }

        T2 idx1 = 0;
        T2 idx2 = 0;
        T2 idx3 = 0;
        T2 idx4 = 0;
        for (uint32_t j = 0; j < IdxDim; j++) {
            idx1 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_ZERO][j];
        }
        int64_t indexValue1 = indexTensor0[idx1];
        if (indexValue1 < 0) {
            indexValue1 += simtParamsPtr_->xShape[LIST_INDEX_ZERO];
        }

        for (uint32_t j = 0; j < IdxDim; j++) {
            idx2 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_ONE][j];
        }
        int64_t indexValue2 = indexTensor1[idx2];
        if (indexValue2 < 0) {
            indexValue2 += simtParamsPtr_->xShape[LIST_INDEX_ONE];
        }

        for (uint32_t j = 0; j < IdxDim; j++) {
            idx3 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_TWO][j];
        }
        int64_t indexValue3 = indexTensor2[idx3];
        if (indexValue3 < 0) {
            indexValue3 += simtParamsPtr_->xShape[LIST_INDEX_TWO];
        }

        for (uint32_t j = 0; j < IdxDim; j++) {
            idx4 += indexDim[j] * simtParamsPtr_->indexedStride[LIST_INDEX_THREE][j];
        }
        int64_t indexValue4 = indexTensor3[idx4];
        if (indexValue4 < 0) {
            indexValue4 += simtParamsPtr_->xShape[LIST_INDEX_THREE];
        }

        inputIndex = indexValue1 * simtParamsPtr_->xStride[LIST_INDEX_ZERO] + indexValue2 * simtParamsPtr_->xStride[LIST_INDEX_ONE] +
                     indexValue3 * simtParamsPtr_->xStride[LIST_INDEX_TWO] + indexValue4 * simtParamsPtr_->xStride[LIST_INDEX_THREE];
        outputGm_[i] = inputXGm_[inputIndex];
    }
}

template <typename T, typename P, typename T2>
class KernelIndexNoContiguousPerf {
public:
    __aicore__ inline KernelIndexNoContiguousPerf(TPipe& pipe)
    : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR output, GM_ADDR inputX, GM_ADDR indices, IndexNonContinuousTilingData tilingData);
    __aicore__ inline void Process();
    __aicore__ inline __gm__ P* GetInputTensorAddr(GM_ADDR indices, uint16_t index);
    __aicore__ inline void ComputeParameter(IndexNonContinuousTilingData tilingData);

private:
    TPipe pipe_;
    AscendC::GlobalTensor<T> outputGm_;
    AscendC::GlobalTensor<T> inputXGm_;
    TBuf<TPosition::VECCALC> Buf_;

    GM_ADDR indices_ = nullptr;
    __ubuf__ SimtParams *simtParamsPtr_;

    uint64_t outputLength_ = 0;
    uint64_t indexedDimNum_ = 0;
    uint64_t inputDimNum_ = 0;
};

template <typename T, typename P, typename T2>
__aicore__ inline void KernelIndexNoContiguousPerf<T, P, T2>::Init(
    GM_ADDR output, GM_ADDR inputX, GM_ADDR indices, IndexNonContinuousTilingData tilingData)
{
    uint32_t blockSize = Ops::Base::GetUbBlockSize();
    inputXGm_.SetGlobalBuffer((__gm__ T*)(inputX));
    outputGm_.SetGlobalBuffer((__gm__ T*)(output));
    pipe_.InitBuffer(Buf_, Ops::Base::CeilAlign(sizeof(SimtParams), static_cast<uint64_t>(blockSize)));
    LocalTensor<int64_t> simtParamsUb = Buf_.Get<int64_t>();
    simtParamsPtr_ = (__ubuf__ SimtParams*)simtParamsUb.GetPhyAddr();
    indices_ = indices;
    outputLength_ = tilingData.outputLength;
    indexedDimNum_ = tilingData.indexedDimNum;
    inputDimNum_ = tilingData.inputDimNum;
    ComputeParameter(tilingData);
}

template <typename T, typename P, typename T2>
__aicore__ inline void KernelIndexNoContiguousPerf<T, P, T2>::ComputeParameter(IndexNonContinuousTilingData tilingData)
{
    for (uint16_t i = 0; i < indexedDimNum_; i++) {
        simtParamsPtr_->indexedStride[LIST_INDEX_ZERO][i] = tilingData.indexStride1[i];
        simtParamsPtr_->indexedStride[LIST_INDEX_ONE][i] = tilingData.indexStride2[i];
        simtParamsPtr_->indexedStride[LIST_INDEX_TWO][i] = tilingData.indexStride3[i];
        simtParamsPtr_->indexedStride[LIST_INDEX_THREE][i] = tilingData.indexStride4[i];
    }
    for (uint16_t i = 0; i < NOCON_IDX_MAX_DIM_NUM; i++) {
        simtParamsPtr_->indexShape[i] = tilingData.indexShape[i];
        simtParamsPtr_->xShape[i] = tilingData.xShape[i];
        simtParamsPtr_->xStride[i] = tilingData.xStride[i];
    }
}

template <typename T, typename P, typename T2>
__aicore__ inline __gm__ P* KernelIndexNoContiguousPerf<T, P, T2>::GetInputTensorAddr(GM_ADDR indices, uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(indices);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
                                          // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> OFFSET);
    return reinterpret_cast<__gm__ P*>(*(tensorPtr + index));
}

template <typename T, typename P, typename T2>
__aicore__ inline void KernelIndexNoContiguousPerf<T, P, T2>::Process()
{
    // get dynamci input tensor
    __gm__ P* indexTensor0 = GetInputTensorAddr(indices_, LIST_INDEX_ZERO);
    __gm__ P* indexTensor1 = GetInputTensorAddr(indices_, LIST_INDEX_ONE);
    if (inputDimNum_ == INPUTDIMNUMTWO) {
        if (indexedDimNum_ == INDEX_DIM_NUM_ONE) {
            AscendC::Simt::VF_CALL<SimtDim2Compute<T, P, T2, INDEX_DIM_NUM_ONE>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_TWO) {
            AscendC::Simt::VF_CALL<SimtDim2Compute<T, P, T2, INDEX_DIM_NUM_TWO>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_THREE) {
            AscendC::Simt::VF_CALL<SimtDim2Compute<T, P, T2, INDEX_DIM_NUM_THREE>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_FOUR) {
            AscendC::Simt::VF_CALL<SimtDim2Compute<T, P, T2, INDEX_DIM_NUM_FOUR>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, simtParamsPtr_, outputLength_);
        }
    } else if (inputDimNum_ == INPUTDIMNUMTHREE) {
        if (indexedDimNum_ == INDEX_DIM_NUM_ONE) {
            __gm__ P* indexTensor2 = GetInputTensorAddr(indices_, LIST_INDEX_TWO);
            AscendC::Simt::VF_CALL<SimtDim3Compute<T, P, T2, INDEX_DIM_NUM_ONE>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, indexTensor2, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_TWO) {
            __gm__ P* indexTensor2 = GetInputTensorAddr(indices_, LIST_INDEX_TWO);
            AscendC::Simt::VF_CALL<SimtDim3Compute<T, P, T2, INDEX_DIM_NUM_TWO>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, indexTensor2, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_THREE) {
            __gm__ P* indexTensor2 = GetInputTensorAddr(indices_, LIST_INDEX_TWO);
            AscendC::Simt::VF_CALL<SimtDim3Compute<T, P, T2, INDEX_DIM_NUM_THREE>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, indexTensor2, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_FOUR) {
            __gm__ P* indexTensor2 = GetInputTensorAddr(indices_, LIST_INDEX_TWO);
            AscendC::Simt::VF_CALL<SimtDim3Compute<T, P, T2, INDEX_DIM_NUM_FOUR>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, indexTensor2, simtParamsPtr_, outputLength_);
        }
    } else if (inputDimNum_ == INPUTDIMNUMFOUR) {
        if (indexedDimNum_ == INDEX_DIM_NUM_ONE) {
            __gm__ P* indexTensor2 = GetInputTensorAddr(indices_, LIST_INDEX_TWO);
            __gm__ P* indexTensor3 = GetInputTensorAddr(indices_, LIST_INDEX_THREE);
            AscendC::Simt::VF_CALL<SimtDim4Compute<T, P, T2, INDEX_DIM_NUM_ONE>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, indexTensor2, indexTensor3, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_TWO) {
            __gm__ P* indexTensor2 = GetInputTensorAddr(indices_, LIST_INDEX_TWO);
            __gm__ P* indexTensor3 = GetInputTensorAddr(indices_, LIST_INDEX_THREE);
            AscendC::Simt::VF_CALL<SimtDim4Compute<T, P, T2, INDEX_DIM_NUM_TWO>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, indexTensor2, indexTensor3, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_THREE) {
            __gm__ P* indexTensor2 = GetInputTensorAddr(indices_, LIST_INDEX_TWO);
            __gm__ P* indexTensor3 = GetInputTensorAddr(indices_, LIST_INDEX_THREE);
            AscendC::Simt::VF_CALL<SimtDim4Compute<T, P, T2, INDEX_DIM_NUM_THREE>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, indexTensor2, indexTensor3, simtParamsPtr_, outputLength_);
        } else if (indexedDimNum_ == INDEX_DIM_NUM_FOUR) {
            __gm__ P* indexTensor2 = GetInputTensorAddr(indices_, LIST_INDEX_TWO);
            __gm__ P* indexTensor3 = GetInputTensorAddr(indices_, LIST_INDEX_THREE);
            AscendC::Simt::VF_CALL<SimtDim4Compute<T, P, T2, INDEX_DIM_NUM_FOUR>>(
                AscendC::Simt::Dim3{USED_THREAD}, (__gm__ T*)outputGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
                indexTensor0, indexTensor1, indexTensor2, indexTensor3, simtParamsPtr_, outputLength_);
        }
    }
}
} // namespace Index

#endif // INDEX_PERF_NO_CONTINUOUS_H_