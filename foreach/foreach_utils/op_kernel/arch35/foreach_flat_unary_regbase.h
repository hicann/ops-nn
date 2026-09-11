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
 * \file foreach_flat_unary_regbase.h
 * \brief Common RegBase transport for same-position unary foreach operators.
 */

#ifndef FOREACH_FLAT_UNARY_REGBASE_H
#define FOREACH_FLAT_UNARY_REGBASE_H

#include "foreach_regbase_common.h"

namespace ForeachFlatRegbase {
using namespace AscendC;

/**
 * Dtype-independent tensor-list dispatcher. It converts the flat core range in
 * ForeachSoloTilingDataRegbase into per-tensor segments and delegates storage
 * and compute work to a typed mover.
 */
template <typename Tiling>
class TensorListUnaryDispatcher {
public:
    __aicore__ inline void Init(const Tiling* tilingData)
    {
        const uint32_t blockIdx = GetBlockIdx();
        tensorDataCountList_ = const_cast<int64_t*>(tilingData->tensorDataCountList);
        tensorStart_ = tilingData->tensorStartList[blockIdx];
        tensorEnd_ = tilingData->tensorEndList[blockIdx];
        tensorStartOffset_ = tilingData->tensorStartOffsetList[blockIdx];
        tensorEndOffset_ = tilingData->tensorEndOffsetList[blockIdx];
    }

    template <typename TypedMover>
    __aicore__ inline void Process(GM_ADDR inputs, GM_ADDR outputs, TypedMover& mover)
    {
        ListTensorDesc inputDesc(reinterpret_cast<__gm__ void*>(inputs));
        ListTensorDesc outputDesc(reinterpret_cast<__gm__ void*>(outputs));
        for (uint16_t tensorIdx = tensorStart_; tensorIdx <= tensorEnd_; ++tensorIdx) {
            int64_t cursorStart = tensorIdx == tensorStart_ ? tensorStartOffset_ : 0;
            int64_t cursorEnd = tensorIdx == tensorEnd_ ? tensorEndOffset_ : tensorDataCountList_[tensorIdx] - 1;
            int64_t dataCount = cursorEnd - cursorStart + 1;
            // Empty tensors are valid. Do not resolve their data pointers or issue a zero/negative DMA.
            if (dataCount <= 0) {
                continue;
            }
            mover.ProcessTensor(inputDesc, outputDesc, tensorIdx, cursorStart, dataCount);
        }
    }

private:
    int64_t* tensorDataCountList_ = nullptr;
    uint16_t tensorStart_ = 0;
    uint16_t tensorEnd_ = 0;
    int64_t tensorStartOffset_ = 0;
    int64_t tensorEndOffset_ = -1;
};

/**
 * Typed storage mover. This is the only layer that owns TQue resources and it
 * allocates exactly one double-buffered queue per input/output flow. ComputePolicy
 * receives LocalTensor handles only and therefore cannot add TBuf/UB allocations.
 */
template <typename T, typename ComputePolicy>
class TypedUnaryMover {
public:
    static_assert(!ComputePolicy::kUsesExtraUb, "flat RegBase compute policies must not allocate extra UB");

    __aicore__ inline void Init(uint32_t tileElements, TPipe* pipe)
    {
        tileElements_ = tileElements;
        pipe->InitBuffer(inputQueue_, BUFFER_NUM, tileElements_ * sizeof(T));
        pipe->InitBuffer(outputQueue_, BUFFER_NUM, tileElements_ * sizeof(T));
    }

    __aicore__ inline void ProcessTensor(ListTensorDesc& inputDesc, ListTensorDesc& outputDesc, uint16_t tensorIdx,
                                         int64_t cursorStart, int64_t dataCount)
    {
        inputGlobal_.SetGlobalBuffer(inputDesc.GetDataPtr<__gm__ T>(tensorIdx) + cursorStart);
        outputGlobal_.SetGlobalBuffer(outputDesc.GetDataPtr<__gm__ T>(tensorIdx) + cursorStart);
        int64_t tileOffset = 0;
        while (tileOffset < dataCount) {
            uint32_t currentCount = static_cast<uint32_t>(
                dataCount - tileOffset < tileElements_ ? dataCount - tileOffset : tileElements_);
            CopyIn(tileOffset, currentCount);
            Compute(currentCount);
            CopyOut(tileOffset, currentCount);
            tileOffset += currentCount;
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t tileOffset, uint32_t dataCount)
    {
        LocalTensor<T> inputLocal = inputQueue_.template AllocTensor<T>();
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataCount * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(inputLocal, inputGlobal_[tileOffset], copyParams, padParams);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t dataCount)
    {
        LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
        LocalTensor<T> outputLocal = outputQueue_.template AllocTensor<T>();
        ComputePolicy::Run(inputLocal, outputLocal, dataCount);
        inputQueue_.FreeTensor(inputLocal);
        outputQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(int64_t tileOffset, uint32_t dataCount)
    {
        LocalTensor<T> outputLocal = outputQueue_.template DeQue<T>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataCount * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(outputGlobal_[tileOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    static constexpr int32_t BUFFER_NUM = 2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    GlobalTensor<T> inputGlobal_;
    GlobalTensor<T> outputGlobal_;
    uint32_t tileElements_ = 0;
};

/**
 * Reusable Flow<1> kernel facade. The Host template may be shared by Flow<M>;
 * this facade deliberately specializes only the unary transport implemented here.
 */
template <typename T, typename Tiling, typename ComputePolicy>
class FlatUnaryKernel {
public:
    __aicore__ inline void Init(const Tiling* tilingData, TPipe* pipe)
    {
        dispatcher_.Init(tilingData);
        mover_.Init(tilingData->tileElems, pipe);
    }

    __aicore__ inline void Process(GM_ADDR inputs, GM_ADDR outputs) { dispatcher_.Process(inputs, outputs, mover_); }

private:
    TensorListUnaryDispatcher<Tiling> dispatcher_;
    TypedUnaryMover<T, ComputePolicy> mover_;
};
} // namespace ForeachFlatRegbase

#endif // FOREACH_FLAT_UNARY_REGBASE_H
