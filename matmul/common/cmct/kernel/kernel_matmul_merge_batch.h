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

    __aicore__ inline void WaitAivDone(int64_t l0CEventID)
    {
        if (l0CEventID <= 0) {
            return;
        }
        int64_t lastPingPongId = (l0CEventID - 1) & 0x1;
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + lastPingPongId * FLAG_ID_MAX);
        if (l0CEventID > 1) {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG +
                                                                      (lastPingPongId ^ 0x1) * FLAG_ID_MAX);
        }
    }

    template <class BlockMmadOp_, class BlockEpilogueOp_, class CTensor>
    __aicore__ inline void RunUbFusionChunk(BlockMmadOp_& blockMmadOp, BlockEpilogueOp_& epilogueOp, CTensor cLocal,
                                            int64_t offsetA, int64_t offsetB, int64_t offsetC, int64_t curBatchCount,
                                            int64_t& l0CEventID)
    {
        int64_t pingPongId = l0CEventID & 0x1;
        if ASCEND_IS_AIC {
            if (l0CEventID > 1) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + pingPongId * FLAG_ID_MAX);
            }
            blockMmadOp(cLocal, aGlobal_[offsetA], bGlobal_[offsetB], curBatchCount);
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingPongId * FLAG_ID_MAX);
        }
        if ASCEND_IS_AIV {
            if (pingPongId == static_cast<int64_t>(AscendC::GetSubBlockIdx())) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE2>(AIC_SYNC_AIV_FLAG);
                // Apply add/mul to the complete UB result on the AIV selected by Ping/Pong parity.
                epilogueOp.RunUbFusion(offsetC, curBatchCount);
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            }
        }
        l0CEventID++;
    }

    template <class BlockMmadOp_, class BlockEpilogueOp_>
    __aicore__ inline void RunUbFusionTiles(BlockMmadOp_& blockMmadOp, BlockEpilogueOp_& epilogueOp,
                                            BlockSchedulerOp& bs, int64_t curBlockIdx, int64_t blockNum,
                                            int64_t tileNum)
    {
        int64_t nAlign = Align(static_cast<uint64_t>(n_), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        int64_t elemsPerBatch = static_cast<int64_t>(m_) * nAlign;
        int64_t ubElems = AscendC::TOTAL_UB_SIZE / sizeof(CType);
        // Reserve one x3 buffer row for the AIV selected by the current Ping/Pong buffer.
        int64_t reserveElems = BlockEpilogue::GetMinX3BufferElems(nAlign);
        int64_t maxBatchPerUb = (ubElems - reserveElems) / elemsPerBatch;
        int64_t l0CEventID = 0;
        // Use the epilogue-owned UB as the Fixpipe destination shared by the AIC/AIV fusion pipeline.
        AscendC::LocalTensor<CType> cLocal = epilogueOp.GetFusionUbTensor();
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockOffset = GetOffsetIterBatch(blockCoord, problemShape_, aGlobal_, bGlobal_, cGlobal_);
            int64_t tileBatchCount = Get<3>(blockShape);
            for (int64_t batchOffset = 0; batchOffset < tileBatchCount; batchOffset += maxBatchPerUb) {
                int64_t curBatchCount = AscendC::Std::min(maxBatchPerUb, tileBatchCount - batchOffset);
                int64_t offsetA = Get<0>(blockOffset) +
                                  batchOffset * static_cast<int64_t>(m_) * static_cast<int64_t>(k_);
                int64_t offsetB = Get<1>(blockOffset) +
                                  batchOffset * static_cast<int64_t>(k_) * static_cast<int64_t>(n_);
                int64_t offsetC = Get<2>(blockOffset) +
                                  batchOffset * static_cast<int64_t>(m_) * static_cast<int64_t>(n_);
                // Run one MMAD-to-UB chunk on the AIV selected by the Ping/Pong parity.
                RunUbFusionChunk(blockMmadOp, epilogueOp, cLocal, offsetA, offsetB, offsetC, curBatchCount, l0CEventID);
            }
        }
        if ASCEND_IS_AIC {
            // Wait for the last outstanding Ping and Pong owners before the AIC exits.
            WaitAivDone(l0CEventID);
        }
    }

    template <class BlockMmadOp_, class BlockEpilogueOp_>
    __aicore__ inline void RunTiles(BlockMmadOp_& blockMmadOp, BlockEpilogueOp_& epilogueOp, BlockSchedulerOp& bs,
                                    int64_t curBlockIdx, int64_t blockNum, int64_t tileNum)
    {
        constexpr bool enableFusion = BlockMmadOp::DispatchPolicy::enableAdd || BlockMmadOp::DispatchPolicy::enableMul;
        if constexpr (enableFusion) {
            // Alternate each MergeBatch UB chunk between AIV0/Ping and AIV1/Pong.
            RunUbFusionTiles(blockMmadOp, epilogueOp, bs, curBlockIdx, blockNum, tileNum);
        } else {
            for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
                auto blockShape = bs.GetBlockShape(tileIdx);
                auto blockCoord = bs.GetBlockCoord(tileIdx);
                auto blockOffset = GetOffsetIterBatch(blockCoord, problemShape_, aGlobal_, bGlobal_, cGlobal_);
                if ASCEND_IS_AIC {
                    blockMmadOp(cGlobal_[Get<2>(blockOffset)], aGlobal_[Get<0>(blockOffset)],
                                bGlobal_[Get<1>(blockOffset)], Get<3>(blockShape));
                }
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
        SetHf32Mode(bs.GetHf32Flag(), true);
        RunTiles(blockMmadOp, epilogueOp, bs, curBlockIdx, blockNum, tileNum);
        SetHf32Mode(bs.GetHf32Flag(), false);
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
