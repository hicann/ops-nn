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
    using FusionParams = typename FusionOp::Params;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;

    struct Arguments {
        GM_ADDR cGmAddr{nullptr};
        typename FusionOp::Arguments fusionArgs{};
    };

    struct Params {
        GM_ADDR cGmAddr{nullptr};
        FusionParams fusionParams{};
    };

    struct FusionRange {
        int64_t rowBegin{0};
        int64_t rowCount{0};
    };

    struct PrefetchState {
        bool valid{false};
        int64_t curRows{0};
        int64_t curOffset{0};
        int64_t localRowOffset{0};
        int64_t nAlign{0};
        int64_t stageRows{0};
        uint32_t dstStride{0};
    };

    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    AscendC::LocalTensor<DataTypeIn> fusionUbLocal_{
        AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
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

    __aicore__ inline FusionRange GetFusionRange(int64_t batchRows) const
    {
        int64_t rowCountAll = batchRows * m_;
        int64_t halfRows = CeilDiv(rowCountAll, AscendC::GetTaskRation());
        int64_t rowBegin = halfRows * AscendC::GetSubBlockIdx();
        int64_t rowCount = rowCountAll > rowBegin ? AscendC::Std::min(halfRows, rowCountAll - rowBegin) : 0;
        return {rowBegin, rowCount};
    }

    __aicore__ inline int64_t GetStageRows(int64_t& nAlign) const
    {
        int64_t alignUnit = static_cast<int64_t>(AscendC::AuxGetC0Size<DataTypeIn>());
        if (alignUnit <= 0) {
            return 0;
        }
        nAlign = Align(static_cast<uint64_t>(n_), static_cast<uint64_t>(alignUnit));
        if (nAlign <= 0) {
            return 0;
        }
        int64_t ubElems = AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn);
        return AscendC::Std::min(ubElems / NUM_TWO / nAlign, static_cast<int64_t>(UINT16_MAX));
    }

    __aicore__ inline void CopyY(AscendC::LocalTensor<DataTypeIn> yLocal, int64_t curRows, int64_t curOffset,
        int64_t nAlign, uint32_t dstStride, const AscendC::DataCopyPadExtParams<DataTypeIn>& padParams)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows), static_cast<uint32_t>(n_ * sizeof(DataTypeIn)), 0, dstStride, 0};
        AscendC::DataCopyPad(yLocal, outputGlobal_[curOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyInputs(AscendC::LocalTensor<DataTypeIn> yLocal,
        AscendC::LocalTensor<DataTypeIn> x3Local, int64_t curRows, int64_t curOffset, int64_t localRowOffset,
        int64_t nAlign, uint32_t dstStride, const AscendC::DataCopyPadExtParams<DataTypeIn>& padParams)
    {
        CopyY(yLocal, curRows, curOffset, nAlign, dstStride, padParams);
        fusionOp_.CopyX3(x3Local, curRows, curOffset, localRowOffset, nAlign, static_cast<int64_t>(n_));
    }

    __aicore__ inline PrefetchState PrefetchX3(int64_t offsetC, int64_t batchRows)
    {
        PrefetchState state{};
        if (fusionOp_.x3BatchBroadcast_) {
            return state;
        }
        FusionRange range = GetFusionRange(batchRows);
        if (range.rowCount <= 0) {
            return state;
        }
        int64_t nAlign = 0;
        int64_t stageRows = GetStageRows(nAlign);
        if (stageRows <= 0) {
            return state;
        }

        AscendC::LocalTensor<DataTypeIn> x3Local = fusionUbLocal_[stageRows * nAlign];
        int64_t curRows = AscendC::Std::min(stageRows, range.rowCount);
        int64_t curOffset = offsetC + range.rowBegin * n_;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        fusionOp_.CopyX3(x3Local, curRows, curOffset, range.rowBegin, nAlign, static_cast<int64_t>(n_));

        state.valid = true;
        state.curRows = curRows;
        state.curOffset = curOffset;
        state.localRowOffset = range.rowBegin;
        state.nAlign = nAlign;
        state.stageRows = stageRows;
        state.dstStride = static_cast<uint32_t>((nAlign - n_) * sizeof(DataTypeIn) / UB_ALIGN_SIZE);
        return state;
    }

    __aicore__ inline void ApplyAndCopyOut(
        AscendC::LocalTensor<DataTypeIn> yLocal, AscendC::LocalTensor<DataTypeIn> x3Local, int64_t curCount,
        int64_t curRows, int64_t curOffset, uint32_t dstStride)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        fusionOp_.Apply(yLocal, x3Local, yLocal, curCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams outParams{
            static_cast<uint16_t>(curRows), static_cast<uint32_t>(n_ * sizeof(DataTypeIn)), dstStride, 0, 0};
        AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[curOffset], yLocal, outParams);
    }

    __aicore__ inline void Run(int64_t offsetC, int64_t batchRows, PrefetchState const& prefetchState)
    {
        FusionRange range = GetFusionRange(batchRows);
        if (range.rowCount <= 0) {
            return;
        }
        int64_t nAlign = 0;
        int64_t stageRows = GetStageRows(nAlign);
        if (stageRows <= 0) {
            return;
        }

        AscendC::LocalTensor<DataTypeIn> yLocal = fusionUbLocal_;
        AscendC::LocalTensor<DataTypeIn> x3Local = fusionUbLocal_[stageRows * nAlign];
        AscendC::DataCopyPadExtParams<DataTypeIn> padParams{false, 0, 0, 0};
        uint32_t dstStride = static_cast<uint32_t>((nAlign - n_) * sizeof(DataTypeIn) / UB_ALIGN_SIZE);
        int64_t offset = offsetC + range.rowBegin * n_;
        for (int64_t rowOffset = 0; rowOffset < range.rowCount; rowOffset += stageRows) {
            int64_t curRows = AscendC::Std::min(stageRows, range.rowCount - rowOffset);
            int64_t curOffset = offset + rowOffset * n_;
            if (rowOffset > 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            }
            bool usePrefetch = prefetchState.valid && rowOffset == 0 && prefetchState.curRows == curRows &&
                prefetchState.curOffset == curOffset && prefetchState.localRowOffset == range.rowBegin &&
                prefetchState.nAlign == nAlign && prefetchState.stageRows == stageRows &&
                prefetchState.dstStride == dstStride;
            if (usePrefetch) {
                CopyY(yLocal, curRows, curOffset, nAlign, dstStride, padParams);
            } else {
                CopyInputs(yLocal, x3Local, curRows, curOffset, range.rowBegin + rowOffset, nAlign, dstStride,
                    padParams);
            }
            ApplyAndCopyOut(yLocal, x3Local, curRows * nAlign, curRows, curOffset, dstStride);
        }
    }

private:
    constexpr static int64_t NUM_TWO = 2;
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct