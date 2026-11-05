/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FOREACH_ZERO_INPLACE_SIMD_H
#define FOREACH_ZERO_INPLACE_SIMD_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_zero_inplace_tiling_data.h"
#include "foreach_zero_inplace_tiling_key.h"

namespace ForeachZeroInplaceA5 {

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <typename T>
class ForeachZeroInplaceSIMD {
public:
    __aicore__ inline void Init(GM_ADDR x, const ForeachZeroInplaceTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        x_ = x;
        tensorElements_ = tilingData->tensorElements;
        tensorStart_ = tilingData->tensorStart[blockIdx_];
        tensorEnd_ = tilingData->tensorEnd[blockIdx_];
        tensorStartOffset_ = tilingData->tensorStartOffset[blockIdx_];
        tensorEndOffset_ = tilingData->tensorEndOffset[blockIdx_];
        if (tensorStart_ <= tensorEnd_) {
            pipe_.InitBuffer(zeroQueue_, BUFFER_NUM, tilingData->ubChunkBytes);
            maxDataCount_ = tilingData->ubChunkBytes / sizeof(T);
        }
    }

    __aicore__ inline void Process()
    {
        if (tensorStart_ > tensorEnd_) {
            return;
        }

        ListTensorDesc xList(reinterpret_cast<__gm__ void*>(x_));
        for (uint16_t tensorIndex = tensorStart_; tensorIndex <= tensorEnd_; ++tensorIndex) {
            const int64_t cursorStart = tensorIndex == tensorStart_ ? tensorStartOffset_ : 0;
            const int64_t cursorEnd = tensorIndex == tensorEnd_ ? tensorEndOffset_ : tensorElements_[tensorIndex] - 1;
            if (cursorEnd < cursorStart) {
                continue;
            }
            outputGm_.SetGlobalBuffer(xList.GetDataPtr<T>(tensorIndex) + cursorStart);
            ZeroTensor(cursorEnd - cursorStart + 1);
        }
    }

private:
    __aicore__ inline void ZeroTensor(int64_t dataCount)
    {
        const uint64_t fullChunks = static_cast<uint64_t>(dataCount) / maxDataCount_;
        const uint32_t tailCount = static_cast<uint32_t>(dataCount % maxDataCount_);
        for (uint64_t chunk = 0; chunk < fullChunks; ++chunk) {
            ZeroChunk(chunk, maxDataCount_);
        }
        if (tailCount > 0) {
            ZeroChunk(fullChunks, tailCount);
        }
    }

    __aicore__ inline void ZeroChunk(uint64_t chunkIndex, uint32_t dataCount)
    {
        LocalTensor<T> zeroLocal = zeroQueue_.AllocTensor<T>();
        Duplicate(zeroLocal, static_cast<T>(0), dataCount);
        zeroQueue_.EnQue(zeroLocal);

        LocalTensor<T> zeroOut = zeroQueue_.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[chunkIndex * maxDataCount_], zeroOut, copyParams);
        zeroQueue_.FreeTensor(zeroOut);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECOUT, 1> zeroQueue_;
    GlobalTensor<T> outputGm_;
    GM_ADDR x_ = nullptr;
    const int64_t* tensorElements_ = nullptr;
    uint32_t blockIdx_ = 0;
    uint32_t maxDataCount_ = 0;
    uint16_t tensorStart_ = 1;
    uint16_t tensorEnd_ = 0;
    int64_t tensorStartOffset_ = 0;
    int64_t tensorEndOffset_ = -1;
};

template <typename T>
__aicore__ inline void Process(GM_ADDR x, const ForeachZeroInplaceTilingData* tilingData)
{
    ForeachZeroInplaceSIMD<T> op;
    op.Init(x, tilingData);
    op.Process();
}

} // namespace ForeachZeroInplaceA5

#endif // FOREACH_ZERO_INPLACE_SIMD_H
