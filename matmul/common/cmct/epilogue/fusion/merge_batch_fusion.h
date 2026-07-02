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
 * \file merge_batch_fusion.h
 * \brief MergeBatch fusion ops for x3 copy and Add/Mul.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../tile/x3_copy_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_, bool IS_ADD>
class MergeBatchFusionBase {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
        bool x3BatchBroadcast{false};
        int64_t x3M{0};
    };

    using Params = Arguments;

    AscendC::GlobalTensor<DataTypeIn> inputGlobal_;
    bool x3BatchBroadcast_{false};
    int64_t x3M_{0};

    __aicore__ inline void Init(Params const& params)
    {
        inputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn*>(params.inputGmAddr));
        x3BatchBroadcast_ = params.x3BatchBroadcast;
        x3M_ = params.x3M;
    }

    template <class LocalTensor>
    __aicore__ inline bool CopyX3(LocalTensor x3Local, int64_t curRows, int64_t curOffset, int64_t localRowOffset,
        int64_t nAlign, int64_t n)
    {
        if (n <= 0 || nAlign < n || curRows <= 0) {
            return false;
        }
        int64_t x3Offset = x3BatchBroadcast_ ? localRowOffset * n + curOffset % n : curOffset;
        uint32_t dstStride = static_cast<uint32_t>((nAlign - n) * sizeof(DataTypeOut) / UB_ALIGN_SIZE);
        return Detail::CopyFusionX3Aligned<DataTypeIn, DataTypeOut>(
            x3Local, inputGlobal_, x3Offset, curRows, n, n, curRows * nAlign, false, nAlign, dstStride,
            x3BatchBroadcast_, x3M_, false, false);
    }

    template <class SrcTensor, class X3Tensor, class OutTensor>
    __aicore__ inline void Apply(SrcTensor yLocal, X3Tensor x3Local, OutTensor outputLocal, int64_t elemCount)
    {
        if constexpr (IS_ADD) {
            AscendC::Add(outputLocal, yLocal, x3Local, elemCount);
        } else {
            AscendC::Mul(outputLocal, yLocal, x3Local, elemCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <typename DataTypeOut_, typename DataTypeIn_>
using MergeBatchFusionAdd = MergeBatchFusionBase<DataTypeOut_, DataTypeIn_, true>;

template <typename DataTypeOut_, typename DataTypeIn_>
using MergeBatchFusionMul = MergeBatchFusionBase<DataTypeOut_, DataTypeIn_, false>;

} // namespace Block
} // namespace Gemm
} // namespace Cmct