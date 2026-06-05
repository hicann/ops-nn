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
 * \file block_epilogue_fixpipe_fusion.h
 * \brief BlockEpilogue for fixpipe path with FusionOp (Add/Mul).
 *        After fixpipe writes matmul result to UB, this epilogue reads x3 from GM,
 *        does Add/Mul in UB, then writes result to GM.
 *        Interface is compatible with BlockEpilogueFixpipe.
 */

#pragma once
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "fusion/default_fusion_op.h"
#include "fusion/fusion_add.h"
#include "fusion/fusion_mul.h"
#include "../utils/common_utils.h"
#include "../utils/status_utils.h"
namespace Cmct {
namespace Gemm {
namespace Block {

template <
    typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_,
    typename FusionOp_>
class BlockEpilogueFixpipeFusion {
public:
    using FusionParams = typename FusionOp_::Params;

    __aicore__ inline BlockEpilogueFixpipeFusion()
    {}

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        typename FusionOp_::Arguments fusionArgs{};
    };

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        FusionParams fusionParams{};
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    using DispatchPolicy = DispatchPolicy_;

    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    // block shape: mL1, nL1, k, batch, mL0, nL0
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

    // input ub tensor and output global tensor
    AscendC::LocalTensor<DataTypeIn> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::LocalTensor<DataTypeIn> ubLocalTmp_;
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    AscendC::GlobalTensor<DataTypeIn> x3Global_;

    // attribute
    ProblemShape problemShape_;
    int64_t ubBaseOffset_{0};

    __aicore__ inline void Init(Params const &params, ProblemShape &problemShape)
    {
        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut *>(params.outGmAddr));
        x3Global_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn *>(params.fusionParams.inputGmAddr));
        ASCENDC_ASSERT(sizeof(DataTypeIn) >= sizeof(DataTypeOut),
            { KERNEL_LOG(KERNEL_EORROR, "Unsupport dtype size %zu, %zu!", sizeof(DataTypeIn), sizeof(DataTypeOut)); });
    }

    __aicore__ inline void Run(BlockShape const &blockShape, int64_t dstOffset, bool splitM)
    {
        int64_t blockShapeM = Get<MNK_M0>(blockShape);
        int64_t halfBlockShapeM = Cmct::Gemm::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        if (splitM) {
            blockShapeM = (static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL
                              ? (halfBlockShapeM - AscendC::GetSubBlockIdx()) : halfBlockShapeM;
        }
        int64_t blockShapeN = Get<MNK_N0>(blockShape);
        int64_t N = Get<MNK_N>(problemShape_);
        int64_t blockShapeNAlign = static_cast<int64_t>(
            Cmct::Gemm::Align(static_cast<uint64_t>(blockShapeN),
                              static_cast<uint64_t>(AscendC::AuxGetC0Size<DataTypeOut>())));
        int64_t resultCount = blockShapeM * blockShapeNAlign;
        if (resultCount <= 0) {
            return;
        }
        int64_t offset = dstOffset + halfBlockShapeM * N * (AscendC::GetSubBlockIdx() & 0x1);
        int64_t ubTotalElems = AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn);
        int64_t x3StageElems = ubTotalElems - ubBaseOffset_ - resultCount;
        int64_t stageRows = x3StageElems / blockShapeNAlign;
        ASCENDC_ASSERT(stageRows > 0,
            { KERNEL_LOG(KERNEL_EORROR, "Unsupport fixpipe fusion UB size %ld, %ld, %ld!",
                  ubTotalElems - ubBaseOffset_, resultCount, blockShapeNAlign); });
        if (stageRows <= 0) {
            return;
        }
        stageRows = AscendC::Std::min(stageRows, blockShapeM);
        ProcessFusionStages(blockShapeM, blockShapeN, blockShapeNAlign, N, offset, resultCount, stageRows);
    }

    __aicore__ inline void ProcessFusionStages(int64_t blockShapeM, int64_t blockShapeN,
        int64_t blockShapeNAlign, int64_t N, int64_t offset, int64_t resultCount, int64_t stageRows)
    {
        AscendC::LocalTensor<DataTypeIn> x3Local = ubLocalTmp_[resultCount];
        AscendC::DataCopyPadExtParams<DataTypeIn> padParams{false, 0, 0, 0};
        uint32_t x3DstStride =
            static_cast<uint32_t>((blockShapeNAlign - blockShapeN) * sizeof(DataTypeIn) / UB_ALIGN_SIZE);
        uint32_t outSrcStride =
            static_cast<uint32_t>((blockShapeNAlign - blockShapeN) * sizeof(DataTypeOut) / UB_ALIGN_SIZE);

        for (int64_t rowOffset = 0; rowOffset < blockShapeM; rowOffset += stageRows) {
            int64_t curRows = AscendC::Std::min(stageRows, blockShapeM - rowOffset);
            int64_t curCount = curRows * blockShapeNAlign;
            int64_t curOffset = offset + rowOffset * N;
            AscendC::LocalTensor<DataTypeIn> mmLocal = ubLocalTmp_[rowOffset * blockShapeNAlign];

            AscendC::DataCopyExtParams x3CopyParams{
                static_cast<uint16_t>(curRows),
                static_cast<uint32_t>(blockShapeN * sizeof(DataTypeIn)),
                static_cast<uint32_t>((N - blockShapeN) * sizeof(DataTypeIn)),
                x3DstStride, 0};
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
            AscendC::DataCopyPad(x3Local, x3Global_[curOffset], x3CopyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

            if constexpr (AscendC::IsSameType<FusionOp_, FusionAdd<DataTypeOut_, DataTypeIn_>>::value) {
                AscendC::Add(mmLocal, mmLocal, x3Local, curCount);
            } else {
                AscendC::Mul(mmLocal, mmLocal, x3Local, curCount);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

            AscendC::DataCopyExtParams outCopyParams{
                static_cast<uint16_t>(curRows),
                static_cast<uint32_t>(blockShapeN * sizeof(DataTypeOut)),
                outSrcStride,
                static_cast<uint32_t>((N - blockShapeN) * sizeof(DataTypeOut)), 0};
            AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[curOffset], mmLocal, outCopyParams);
        }
    }

    __aicore__ inline auto GetTensor(uint64_t uBPingPong)
    {
        // GetTensor from ub — same layout as BlockEpilogueFixpipe
        int64_t ubOffset = (uBPingPong * AscendC::TOTAL_UB_SIZE / sizeof(DataTypeOut)) >> 1;
        ubBaseOffset_ = ubOffset;
        ubLocalTmp_ = ubLocal_[ubOffset];
        return ubLocalTmp_;
    }

    __host_aicore__ static Status CanImplement(Arguments const &args)
    {
        if (l0M * l0N * sizeof(DataTypeIn_) > TOTAL_UB_SIZE) {
            return Status::l1L0ErrorExceedsLimit;
        }
        return Status::success;
    }

    __aicore__ inline void operator()(BlockShape const &blockShape, int64_t dstOffset = 0, bool splitM = false)
    {
        Run(blockShape, dstOffset, splitM);
        return;
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // BLOCK_EPILOGUE_FIXPIPE_FUSION_H
