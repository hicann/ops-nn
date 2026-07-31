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
 * \file block_epilogue_mergebatch.h
 * \brief MergeBatch epilogue for fused Add/Mul.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../inc/kernel_utils.h"
#include "../utils/common_utils.h"
#include "../utils/tuple_utils.h"
#include "fusion/merge_batch_fusion.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_, typename FusionOp_>
class BlockEpilogueMergeBatch {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    using X3Type = typename FusionOp::X3Type;
    using FusionParams = typename FusionOp::Params;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool highPrecision = !AscendC::IsSameType<DataTypeIn, X3Type>::value;

    struct Arguments {
        GM_ADDR cGmAddr{nullptr};
        typename FusionOp::Arguments fusionArgs{};
    };

    struct Params {
        GM_ADDR cGmAddr{nullptr};
        FusionParams fusionParams{};
    };

    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    AscendC::LocalTensor<DataTypeIn> fusionUbLocal_{AscendC::TPosition::VECIN, 0,
                                                    AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
    FusionOp fusionOp_;
    uint64_t m_{0};
    uint64_t n_{0};

    __aicore__ inline void Init(Params const& params, const TupleShape& problemShape)
    {
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.cGmAddr));
        m_ = Get<MNK_M>(problemShape);
        n_ = Get<MNK_N>(problemShape);
        fusionOp_.Init(params.fusionParams);
    }

    __aicore__ inline static int64_t GetMinX3BufferElems(int64_t nAlign)
    {
        constexpr int64_t x3BytesPerElem = sizeof(X3Type) + (highPrecision ? sizeof(DataTypeIn) : 0);
        int64_t reserveBytes = nAlign * x3BytesPerElem;
        return CeilDiv(reserveBytes, static_cast<int64_t>(sizeof(DataTypeIn)));
    }

    __aicore__ inline void ApplyFusionAndCopyOut(AscendC::LocalTensor<DataTypeIn> yLocal,
                                                 AscendC::LocalTensor<X3Type> x3Local,
                                                 AscendC::LocalTensor<DataTypeIn> castX3Local, int64_t curCount,
                                                 int64_t curRows, int64_t curOffset, uint32_t outputSrcStride)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        if constexpr (highPrecision) {
            AscendC::Cast(castX3Local, x3Local, AscendC::RoundMode::CAST_NONE, curCount);
            AscendC::PipeBarrier<PIPE_V>();
            fusionOp_.Apply(yLocal, castX3Local, yLocal, curCount);
        } else {
            fusionOp_.Apply(yLocal, x3Local, yLocal, curCount);
        }
        AscendC::LocalTensor<DataTypeOut> outputLocal = yLocal.template ReinterpretCast<DataTypeOut>();
        if constexpr (highPrecision) {
            outputLocal = x3Local.template ReinterpretCast<DataTypeOut>();
            AscendC::Cast(outputLocal, yLocal, AscendC::RoundMode::CAST_RINT, curCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams outParams{static_cast<uint16_t>(curRows),
                                             static_cast<uint32_t>(n_ * sizeof(DataTypeOut)), outputSrcStride, 0, 0};
        AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[curOffset], outputLocal, outParams);
    }

    __aicore__ inline auto GetFusionUbTensor() { return fusionUbLocal_; }

    __aicore__ inline void RunUbFusion(int64_t offsetC, int64_t curBatchCount)
    {
        // Phase 1: place the complete MMAD result first and calculate the remaining UB for x3 processing.
        int64_t nAlign = Align(static_cast<uint64_t>(n_), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        int64_t resultRows = curBatchCount * m_;
        int64_t resultElems = resultRows * nAlign;
        int64_t freeUbBytes = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE) -
                              resultElems * static_cast<int64_t>(sizeof(DataTypeIn));
        constexpr int64_t x3BytesPerElem = sizeof(X3Type) + (highPrecision ? sizeof(DataTypeIn) : 0);
        int64_t rowsPerLoop = ops::FloorDiv(freeUbBytes, nAlign * x3BytesPerElem);

        // Phase 2: map the remaining UB to the raw x3 buffer and the optional high-precision cast buffer.
        int64_t x3BufferElems = CeilDiv(rowsPerLoop * nAlign * static_cast<int64_t>(sizeof(X3Type)),
                                        static_cast<int64_t>(sizeof(DataTypeIn)));
        AscendC::LocalTensor<X3Type> x3Local = fusionUbLocal_[resultElems].template ReinterpretCast<X3Type>();
        int64_t castX3Offset = resultElems + (highPrecision ? x3BufferElems : 0);
        AscendC::LocalTensor<DataTypeIn> castX3Local = fusionUbLocal_[castX3Offset];
        uint32_t outputSrcStride = static_cast<uint32_t>((nAlign - n_) * sizeof(DataTypeOut) / UB_ALIGN_SIZE);

        // Phase 3: repeatedly load x3, apply add/mul, cast if needed, and copy each row chunk to GM.
        for (int64_t rowOffset = 0; rowOffset < resultRows; rowOffset += rowsPerLoop) {
            int64_t curRows = AscendC::Std::min(rowsPerLoop, resultRows - rowOffset);
            int64_t curOffset = offsetC + rowOffset * n_;
            AscendC::LocalTensor<DataTypeIn> yLocal = fusionUbLocal_[rowOffset * nAlign];
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            fusionOp_.CopyX3(x3Local, curRows, curOffset, rowOffset, nAlign, static_cast<int64_t>(n_));
            ApplyFusionAndCopyOut(yLocal, x3Local, castX3Local, curRows * nAlign, curRows, curOffset, outputSrcStride);
        }
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
