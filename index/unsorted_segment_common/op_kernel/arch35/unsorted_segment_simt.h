/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unsorted_segment.h
 * \brief unsorted_segment kernel
 */

#ifndef UNSORTED_SEGMENT_H
#define UNSORTED_SEGMENT_H

#include "unsorted_segment_base.h"

namespace UnsortedSegment
{
using namespace AscendC;
constexpr uint64_t MAX_INT32_NUM = 2147483647;

template <typename T, typename Index, uint8_t Mode>
class KernelUnsortedSegment
{
public:
    __aicore__ inline KernelUnsortedSegment(const UnsortedSegmentSimtTilingData* tiling, TPipe* pipe)
        : td_(tiling), pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
                                
    {
        InitGm<T, Mode>(output, td_->outputOuterDim * td_->innerDim);

        xGm.SetGlobalBuffer((__gm__ T*)(x));
        segmentIdsGm.SetGlobalBuffer((__gm__ Index*)(segmentIds));
        outputGm.SetGlobalBuffer((__gm__ T*)(output));
    }

    template <typename COM_T>
    __aicore__ inline void SimtShell(COM_T inputLength)
    {
        __gm__ T* input = (__gm__ T*)xGm.GetPhyAddr();
        __gm__ Index* segmentIds = (__gm__ Index*)segmentIdsGm.GetPhyAddr();
        __gm__ T* output = (__gm__ T*)outputGm.GetPhyAddr();

        uint32_t blockNums = GetBlockNum();
        uint64_t outputOuterDimSizeTmp = td_->outputOuterDim;
        COM_T innerDimSizeTmp = td_->innerDim;
        COM_T magic = 1;
        COM_T shift = 1;
        GetUintDivMagicAndShift(magic, shift, static_cast<COM_T>(innerDimSizeTmp));
        AscendC::Simt::VF_CALL<SimtComputeSegment<T, Index, COM_T, Mode>>(
            Simt::Dim3(static_cast<uint32_t>(td_->maxThread)), input, segmentIds, output, blockNums, inputLength,
            innerDimSizeTmp, outputOuterDimSizeTmp, magic, shift);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= GetBlockNum()) {
            return;
        }

        uint64_t inputLength = td_->inputOuterDim * td_->innerDim;
        if (inputLength > MAX_INT32_NUM) {
            SimtShell<uint64_t>(inputLength);
        } else {
            SimtShell<uint32_t>(inputLength);
        }
    }

private:
    TPipe* pipe_ = nullptr;
    const UnsortedSegmentSimtTilingData* td_;
    AscendC::GlobalTensor<T> xGm, outputGm;
    AscendC::GlobalTensor<Index> segmentIdsGm;
};
}  // namespace UnsortedSegment
#endif
