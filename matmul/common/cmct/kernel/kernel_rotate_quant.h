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
 * \file kernel_rotate_quant.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"
#include "../block/block_rotate_quant.h"
#include "../block/block_mmad_builder.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"
#include "../epilogue/block_epilogue_rotate_quant.h"
#include "../epilogue/block_epilogue_empty.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

__aicore__ inline uint64_t GetCurrentBlockIdx()
{
    if ASCEND_IS_AIV {
        return AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    }
    return AscendC::GetBlockIdx();
}
template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelRotateQuant {
public:
    __aicore__ inline KernelRotateQuant() {}
    __aicore__ inline ~KernelRotateQuant() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool transA = false;
    static constexpr bool transB = false;

    // schedulerOp
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmadBuilder::L1TileShape, typename BlockMmadBuilder::L0TileShape, BlockScheduler,
        transA, transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;

    // come from cann
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    // no bias in rotate_quant, task BiasType as ScaleType
    using ScaleType = typename BlockMmadBuilder::BiasType;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ParamsShape = Shape<uint64_t, uint64_t, uint64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> rotGlobal_;
    AscendC::GlobalTensor<bfloat16_t> alphaGlobal_;
    AscendC::GlobalTensor<uint16_t> alphaRawGlobal_;

    TupleShape problemShape_{};
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t b_{0};
    bfloat16_t alpha_{0.0f};
    bool needClamp_{false};

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline static TupleShape ToShapeTuple(const ProblemShape& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams_ = params.mmadParams;
        m_ = Get<MNK_M>(problemShape_);
        n_ = Get<MNK_N>(problemShape_);
        k_ = Get<MNK_K>(problemShape_);
        b_ = Get<MNK_B>(problemShape_);
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr));
        rotGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr));
        if (params.schParams.tilingData->hasAlpha == 1) {
            alphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(blockMmadParams_.cGmAddr));
            alphaRawGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(blockMmadParams_.cGmAddr));
            alpha_ = alphaGlobal_.GetValue(0);
            uint16_t alphaRaw = alphaRawGlobal_.GetValue(0);
            // alpha大于0并且小于1时，需要做截断
            needClamp_ = (alphaRaw != 0x0000 && (alphaRaw & 0x7f80) < 0x3f80 && (alphaRaw & 0x8000) == 0);
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        int64_t curBlockIdx = GetCurrentBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        if (blockNum <= 0) {
            return;
        }
        Init(params);

        BlockSchedulerOp bs(params.problemShape, blockNum, params.schParams);
        int64_t tileNum = bs.GetTileNum();
        int64_t realBlockNum = bs.GetBlockNum(blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }

        BlockMmadOp blockMmadOp;
        BlockEpilogue epilogueOp;
        if ASCEND_IS_AIC {
            blockMmadOp.Init(problemShape_, bs.GetTileL1Shape(), bs.GetTileL0Shape());
        }
        if ASCEND_IS_AIV {
            epilogueOp.Init(params.epilogueParams, problemShape_, alpha_, needClamp_, bs.GetAxis(), bs.GetRoundMode(),
                            bs.GetScaleAlg(), bs.GetDstTypeMax(), bs.GetInvDstTypeMax());
        }

        auto ubLocal = epilogueOp.GetTensor();
        for (int64_t tileIdx = curBlockIdx, roundIdx = 0; tileIdx < tileNum; tileIdx += blockNum, roundIdx++) {
            TupleL1L0Shape blockShape = bs.GetBlockShape(tileIdx);
            int64_t offset = bs.GetOffset(tileIdx);
            int64_t rotOffset = bs.GetRotOffset(tileIdx);
            int64_t blkM = Get<MNK_M>(blockShape);
            if ASCEND_IS_AIC {
                if (roundIdx > 0) {
                    CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        (roundIdx & 1) == 1 ? SYNC_AIV_AIC_FLAG : SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                }
                blockMmadOp.template operator()<AscendC::LocalTensor<CType>>(
                    ubLocal, aGlobal_[offset * k_], rotGlobal_[rotOffset], blockShape, tileIdx < blockNum);
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>((roundIdx & 1) == 0 ? SYNC_AIC_AIV_FLAG :
                                                                                      SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            }
            if ASCEND_IS_AIV {
                if ((roundIdx & 1) == AscendC::GetSubBlockIdx()) {
                    CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(SYNC_AIC_AIV_FLAG);
                    epilogueOp(offset, blkM, k_); // 给ub的是blkM * k的数据量
                    if (tileIdx + blockNum < tileNum) {
                        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                    }
                }
            }
        }
    }

    __host_aicore__ static Status CheckShape(const ProblemShape& shape)
    {
        int64_t m = shape.m;
        int64_t n = shape.n;
        int64_t k = shape.k;
        if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
            return Status::mnkErrorExceedsLimit;
        }
        if (!transA && n > MATRIX_INNER_DIM_LIMIT_SIZE) {
            return Status::mkErrorMatrixExceedsLimit;
        }
        if (transA && m > MATRIX_INNER_DIM_LIMIT_SIZE) {
            return Status::kmErrorMatrixExceedsLimit;
        }
        if (!transB && n > MATRIX_INNER_DIM_LIMIT_SIZE) {
            return Status::knErrorMatrixExceedsLimit;
        }
        if (transB && k > MATRIX_INNER_DIM_LIMIT_SIZE) {
            return Status::nkErrorMatrixExceedsLimit;
        }
        return Status::success;
    }

    __host_aicore__ static Status CheckArgs(const Arguments& args)
    {
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        CHECK_AND_RETURN(BlockMmadBuilder::CheckArgs(args.mmadArgs));
        return Status::success;
    }

    __host_aicore__ static size_t GetWorkSpaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        workSpaceSize += BlockMmadBuilder::GetWorkSpaceSize();
        return workSpaceSize;
    }

    __host_aicore__ static Params InitParams(const Arguments& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        Params params = {args.problemShape, mmadParams, {}, {}};
        return params;
    }

    __aicore__ inline void operator()(const Params& params) { Run(params); }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct