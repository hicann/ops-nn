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
 * \file embedding_simt_no_contiguous.h
 * \brief
 */
#ifndef EMBEDDING_NO_CONTIGUOUS_H
#define EMBEDDING_NO_CONTIGUOUS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
 #include "../inc/platform.h"

namespace Embedding {
using namespace AscendC;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 2048;
constexpr int64_t ONE_BLOCK_SIZE = platform::GetUbBlockSize();

template <typename X_T, typename INDICES_T, typename COM_T>
class EmbeddingKernelNoContiguous
{
public:
    __aicore__ inline EmbeddingKernelNoContiguous(const EmbeddingNoContiguousTilingData* tiling, TPipe* pipe)
        : tilingData_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y);
    __aicore__ inline void Process();

private:
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<X_T> yGm_;
    TBuf<QuePosition::VECCALC> idsBuf_;

    const EmbeddingNoContiguousTilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    COM_T currentCoreElements_ = 0;
    COM_T curIndicesNum_ = 0;
    COM_T indicesStart_ = 0;
    COM_T yIndexStart_ = 0;
};

template <typename X_T, typename INDICES_T, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void ComputeSimt(
    __gm__ X_T* x, __gm__ INDICES_T* indicesGm, __ubuf__ INDICES_T* indicesUb, __gm__ X_T* y, COM_T yIndexBase,
    COM_T currentCoreElements, COM_T indicesBase, COM_T curIndicesNum, COM_T m0, COM_T shift0, COM_T m1, COM_T shift1,
    COM_T indicesDim1Size, COM_T indicesDim0Stride, COM_T indicesDim1Stride, COM_T gatherSize, COM_T innerSize,
    COM_T xDim0Stride, COM_T xDim1Stride)
{
    for (COM_T index = static_cast<COM_T>(Simt::GetThreadIdx()); index < curIndicesNum;
         index += static_cast<COM_T>(Simt::GetThreadNum())) {
        COM_T idsIndex = indicesBase + index;
        // get indices dim0 dim1 index
        COM_T dim0Idx = Simt::UintDiv(idsIndex, m1, shift1);
        COM_T dim1Idx = idsIndex - dim0Idx * indicesDim1Size;

        COM_T indicesIdx = dim0Idx * indicesDim0Stride + dim1Idx * indicesDim1Stride;
        indicesUb[index] = indicesGm[indicesIdx];
    }

    AscendC::Simt::ThreadBarrier();

    for (COM_T index = static_cast<COM_T>(Simt::GetThreadIdx()); index < currentCoreElements;
         index += static_cast<COM_T>(Simt::GetThreadNum())) {
        COM_T yIndex = yIndexBase + index;
        // get x dim0 dim1 index
        COM_T gatherIdx = Simt::UintDiv(yIndex, m0, shift0);
        COM_T innerIdx = yIndex - gatherIdx * innerSize;

        COM_T indicesVal = static_cast<COM_T>(indicesUb[gatherIdx - indicesBase]);
        COM_T xIndex = indicesVal * xDim0Stride + innerIdx * xDim1Stride;
        y[yIndex] = x[xIndex];
    }
}

template <typename X_T, typename INDICES_T, typename COM_T>
__aicore__ inline void EmbeddingKernelNoContiguous<X_T, INDICES_T, COM_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y)
{
    xGm_.SetGlobalBuffer((__gm__ X_T*)x);
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    yGm_.SetGlobalBuffer((__gm__ X_T*)y);

    currentCoreElements_ =
        GetBlockIdx() != (tilingData_->needCoreNum - 1) ? tilingData_->perCoreElements : tilingData_->lastCoreElements;
    yIndexStart_ = GetBlockIdx() * tilingData_->perCoreElements;
    COM_T yIndexEnd = yIndexStart_ + currentCoreElements_ - 1;
    indicesStart_ = yIndexStart_ / tilingData_->innerSize;
    curIndicesNum_ = yIndexEnd / tilingData_->innerSize - indicesStart_ + 1;
    int64_t idsAlign = ops::Aligned(static_cast<int64_t>(curIndicesNum_ * sizeof(INDICES_T)), ONE_BLOCK_SIZE);
    pipe_->InitBuffer(idsBuf_, idsAlign);
}

template <typename X_T, typename INDICES_T, typename COM_T>
__aicore__ inline void EmbeddingKernelNoContiguous<X_T, INDICES_T, COM_T>::Process()
{
    if (GetBlockIdx() >= tilingData_->needCoreNum) {
        return;
    }
    COM_T m0{1};
    COM_T shift0{1};
    COM_T m1{1};
    COM_T shift1{1};
    // fast division
    GetUintDivMagicAndShift(m0, shift0, static_cast<COM_T>(tilingData_->innerSize));
    GetUintDivMagicAndShift(m1, shift1, static_cast<COM_T>(tilingData_->indicesDim1Size));
    LocalTensor<INDICES_T> idsLocal = idsBuf_.Get<INDICES_T>();

    AscendC::Simt::VF_CALL<ComputeSimt<X_T, INDICES_T, COM_T>>(
        Simt::Dim3(tilingData_->threadNum), (__gm__ X_T*)(xGm_.GetPhyAddr()),
        (__gm__ INDICES_T*)(indicesGm_.GetPhyAddr()), (__ubuf__ INDICES_T*)(idsLocal.GetPhyAddr()),
        (__gm__ X_T*)(yGm_.GetPhyAddr()), yIndexStart_, currentCoreElements_, indicesStart_, curIndicesNum_,
        static_cast<COM_T>(m0), static_cast<COM_T>(shift0), static_cast<COM_T>(m1), static_cast<COM_T>(shift1),
        static_cast<COM_T>(tilingData_->indicesDim1Size), static_cast<COM_T>(tilingData_->indicesDim0Stride),
        static_cast<COM_T>(tilingData_->indicesDim1Stride), static_cast<COM_T>(tilingData_->gatherSize),
        static_cast<COM_T>(tilingData_->innerSize), static_cast<COM_T>(tilingData_->xDim0Stride),
        static_cast<COM_T>(tilingData_->xDim1Stride));
}
} // namespace Embedding
#endif // EMBEDDING_NO_CONTIGUOUS_H
