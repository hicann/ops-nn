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
 * \file unsorted_segment_prod.h
 * \brief unsorted_segment_prod.h
 */

#ifndef UNSORTED_SEGMENT_PROD_H
#define UNSORTED_SEGMENT_PROD_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "./unsorted_segment_prod_base.h"

namespace UnsortedSegmentProd {
// ======================== Kernel class ========================
template <typename T, typename Index>
class KernelUnsortedSegmentProd {
public:
    __aicore__ inline KernelUnsortedSegmentProd(const UnsortedSegment::UnsortedSegmentSimtTilingData* tiling,
                                                TPipe* pipe)
        : td_(tiling), pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
    {
        InitGmProd<T, InitGmOneValue<T>>(output, td_->outputOuterDim * td_->innerDim);

        xGm.SetGlobalBuffer((__gm__ T*)(x));
        segmentIdsGm.SetGlobalBuffer((__gm__ Index*)(segmentIds));
        outputGm.SetGlobalBuffer((__gm__ T*)(output));
    }

    template <typename COM_T>
    __aicore__ inline void SimtShellNormal(COM_T inputLength)
    {
        __gm__ T* input = (__gm__ T*)xGm.GetPhyAddr();
        __gm__ Index* segmentIds = (__gm__ Index*)segmentIdsGm.GetPhyAddr();
        __gm__ T* output = (__gm__ T*)outputGm.GetPhyAddr();

        uint32_t blockNums = GetBlockNum();
        COM_T innerDimSizeTmp = td_->innerDim;
        COM_T magic = 1;
        COM_T shift = 1;
        GetUintDivMagicAndShift(magic, shift, static_cast<COM_T>(innerDimSizeTmp));
        AscendC::Simt::VF_CALL<SimtComputeSegmentProdNormal<T, Index, COM_T>>(
            Simt::Dim3(static_cast<uint32_t>(td_->maxThread)), input, segmentIds, output, blockNums, inputLength,
            innerDimSizeTmp, td_->outputOuterDim, magic, shift);
    }

    template <typename COM_T>
    __aicore__ inline void SimtShellHalf2(COM_T inputLength)
    {
        __gm__ T* input = (__gm__ T*)xGm.GetPhyAddr();
        __gm__ Index* segmentIds = (__gm__ Index*)segmentIdsGm.GetPhyAddr();
        __gm__ T* output = (__gm__ T*)outputGm.GetPhyAddr();

        uint32_t blockNums = GetBlockNum();
        uint64_t innerDimSizeTmp = td_->innerDim;
        uint64_t inputOuterDim = td_->inputOuterDim;
        uint64_t totalOutputSize = td_->outputOuterDim * innerDimSizeTmp;

        // Check if output row start is even-aligned and innerDim is even
        bool innerDimEven = (innerDimSizeTmp % 2 == 0);
        bool outputStartEven = ((reinterpret_cast<uint64_t>(output) / sizeof(T)) % 2 == 0);

        if (innerDimEven && outputStartEven) {
            COM_T usedThread = static_cast<COM_T>(inputOuterDim * innerDimSizeTmp / 2);
            AscendC::Simt::VF_CALL<SimtComputeSegmentProdHalf2Even<T, Index, COM_T>>(
                Simt::Dim3(static_cast<uint32_t>(td_->maxThread)), input, segmentIds, output, blockNums, usedThread,
                static_cast<COM_T>(innerDimSizeTmp), td_->outputOuterDim);
        } else {
            COM_T eachRowThread = static_cast<COM_T>((innerDimSizeTmp + 2) / 2);
            COM_T usedThread = static_cast<COM_T>(inputOuterDim) * eachRowThread;
            AscendC::Simt::VF_CALL<SimtComputeSegmentProdHalf2Odd<T, Index, COM_T>>(
                Simt::Dim3(static_cast<uint32_t>(td_->maxThread)), input, segmentIds, output, blockNums, usedThread,
                static_cast<COM_T>(innerDimSizeTmp), td_->outputOuterDim, eachRowThread, totalOutputSize);
        }
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= GetBlockNum()) {
            return;
        }

        uint64_t inputLength = td_->inputOuterDim * td_->innerDim;

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            // half/bfloat16: use half2/bfloat16x2_t atomCas
            if (inputLength > MAX_INT32_NUM) {
                SimtShellHalf2<uint64_t>(inputLength);
            } else {
                SimtShellHalf2<uint32_t>(inputLength);
            }
        } else {
            // int32/int64/uint32/uint64/float: use normal atomCas
            if (inputLength > MAX_INT32_NUM) {
                SimtShellNormal<uint64_t>(inputLength);
            } else {
                SimtShellNormal<uint32_t>(inputLength);
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    const UnsortedSegment::UnsortedSegmentSimtTilingData* td_;
    AscendC::GlobalTensor<T> xGm, outputGm;
    AscendC::GlobalTensor<Index> segmentIdsGm;
};
} // namespace UnsortedSegmentProd

#endif // UNSORTED_SEGMENT_PROD_H
