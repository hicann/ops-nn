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
 * \file kernel_matmul_merge_batch.h
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
#include "../block/block_mmad_mergebatch.h"
#include "../block/block_mmad_builder.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"
#include "../epilogue/block_epilogue_empty.h"
namespace Cmct {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatMulMergeBatch {
public:
    __aicore__ inline KernelMatMulMergeBatch() {}
    __aicore__ inline ~KernelMatMulMergeBatch() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    const static int16_t AIV_SYNC_AIC_FLAG = 5;
    const static int16_t AIC_SYNC_AIV_FLAG = 8;
    const static int16_t FLAG_ID_MAX = 16;
    const static int16_t COUNT_ID_MAX = 15;
    const static int16_t COUNT_FLAG = 3;
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
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    // GM Tensor
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    // Shape
    TupleShape problemShape_{};
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t b_{0};

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

    __aicore__ inline static TupleShape ToShapeTuple(ProblemShape const& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams = params.mmadParams;
        m_ = Get<MNK_M>(problemShape_);
        n_ = Get<MNK_N>(problemShape_);
        k_ = Get<MNK_K>(problemShape_);
        b_ = Get<MNK_B>(problemShape_);
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr));
    }

    __host_aicore__ static Status CheckShape(ProblemShape const& shape)
    {
        int64_t m = shape.m;
        int64_t n = shape.n;
        int64_t k = shape.k;
        int64_t b = shape.b;
        if (b > INT32_MAX) {
            return Status::batchErrorExcceedsLimit;
        }
        // Check m, n, k overlimit data type
        if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
            return Status::mnkErrorExceedsLimit;
        }
        // Check matrix size exceeds limit
        if (!transA && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // mk matrix k limit
            return Status::mkErrorMatrixExceedsLimit;
        }

        if (transA && m > MATRIX_INNER_DIM_LIMIT_SIZE) { // km matrix m limit
            return Status::kmErrorMatrixExceedsLimit;
        }
        if (!transB && n > MATRIX_INNER_DIM_LIMIT_SIZE) { // kn matrix n limit
            return Status::knErrorMatrixExceedsLimit;
        }

        if (transB && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // nk matrix k limit
            return Status::nkErrorMatrixExceedsLimit;
        }
        return Status::success;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        CHECK_AND_RETURN(BlockMmadBuilder::CanImplement(args.mmadArgs));

        return Status::success;
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkspaceSize();

        return workSpaceSize;
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        // mmad params with epiligue takes workspaceGm as output
        Params params = {args.problemShape, mmadParams, {}};
        return params;
    }

    __aicore__ inline void ApplyCacheHint(BlockSchedulerOp& bs)
    {
        if (bs.GetBL2CacheDisable()) {
            bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (bs.GetAL2CacheDisable()) {
            aGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
    }

    __aicore__ inline void WaitAivDone(int64_t count, bool enableCVSync)
    {
        if (enableCVSync) {
            int64_t countId = count / COUNT_ID_MAX % COUNT_FLAG;
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId);
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void SetHf32Mode(BlockSchedulerOp& bs, bool enable)
    {
        if ASCEND_IS_AIC {
            if (bs.GetHf32Flag()) {
                AscendC::SetHF32Mode(enable ? 1 : 0);
                if (enable) {
                    AscendC::SetHF32TransMode(1);
                }
            }
        }
    }

    template <class BlockMmadOp_, class BlockEpilogueOp_>
    __aicore__ inline void RunTiles(BlockMmadOp_& blockMmadOp, BlockEpilogueOp_& epilogueOp, BlockSchedulerOp& bs,
                                    int64_t curBlockIdx, int64_t blockNum, int64_t tileNum)
    {
        constexpr bool enableFusion = BlockMmadOp::DispatchPolicy::enableAdd || BlockMmadOp::DispatchPolicy::enableMul;
        int64_t count = 0;
        bool enableCVSync = false;
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockOffset = GetOffsetIterBatch(blockCoord, problemShape_, aGlobal_, bGlobal_, cGlobal_);
            int64_t offsetA = Get<0>(blockOffset);
            int64_t offsetB = Get<1>(blockOffset);
            int64_t offsetC = Get<2>(blockOffset);
            if ASCEND_IS_AIC {
                if constexpr (enableFusion) {
                    if (enableCVSync) {
                        int64_t countId = count / COUNT_ID_MAX % COUNT_FLAG;
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId);
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId +
                                                                                  FLAG_ID_MAX);
                    }
                }
                blockMmadOp(cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB], Get<3>(blockShape));
                if constexpr (enableFusion) {
                    enableCVSync = true;
                    count++;
                    int64_t countId = count / COUNT_ID_MAX % COUNT_FLAG;
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + countId);
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + countId + FLAG_ID_MAX);
                }
            }
            if constexpr (enableFusion) {
                if ASCEND_IS_AIV {
                    count++;
                    int64_t countId = count / COUNT_ID_MAX % COUNT_FLAG;
                    auto prefetchState = epilogueOp.PrefetchX3(offsetC, Get<3>(blockShape));
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE2>(AIC_SYNC_AIV_FLAG + countId);
                    epilogueOp.Run(offsetC, Get<3>(blockShape), prefetchState);
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + countId);
                }
            }
        }
        if ASCEND_IS_AIC {
            if constexpr (enableFusion) {
                WaitAivDone(count, enableCVSync);
            }
        }
    }

    __aicore__ inline void operator()(Params const& params)
    {
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        BlockEpilogue epilogueOp;
        // Get hardware block index.
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        // Get runtime block count.
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);
        constexpr bool enableFusion = BlockMmadOp::DispatchPolicy::enableAdd || BlockMmadOp::DispatchPolicy::enableMul;
        if constexpr (!enableFusion) {
            if ASCEND_IS_AIV {
                return;
            }
        }
        if ASCEND_IS_AIV {
            curBlockIdx /= AscendC::GetTaskRation();
        }
        BlockSchedulerOp bs(params.problemShape, blockNum, params.schParams);
        ApplyCacheHint(bs);
        // Split batch axis.
        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        TupleShape iterBatchTuple = bs.GetIterBatchTuple();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        blockMmadOp.Init(problemShape_, iterBatchTuple, tileL1, tileL0, static_cast<uint8_t>(bs.GetShiftValue()));
        if constexpr (enableFusion) {
            if ASCEND_IS_AIV {
                epilogueOp.Init(params.epilogueParams, problemShape_);
            }
        }
#if __NPU_ARCH__ != 5102
        SetHf32Mode(bs, true);
#endif
        RunTiles(blockMmadOp, epilogueOp, bs, curBlockIdx, blockNum, tileNum);
#if __NPU_ARCH__ != 5102
        SetHf32Mode(bs, false);
#endif
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
