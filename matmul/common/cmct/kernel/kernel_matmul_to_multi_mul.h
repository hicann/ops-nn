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
 * \file kernel_matmul_to_multi_mul.h
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
#include "../block/block_mmad_to_multi_mul.h"
#include "../block/block_mmad_builder.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"
#include "../epilogue/block_epilogue_empty.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelMatmulToVector {
    static_assert(AscendC::Std::always_false_v<BlockEpilogue_>,
                  "KernelMatmulToVector is not implemented for this BlockEpilogue");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulToVector<ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
                           std::enable_if_t<std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    __aicore__ inline KernelMatmulToVector() {}
    __aicore__ inline ~KernelMatmulToVector() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;

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
    using ParamsShape = Shape<uint64_t, uint64_t, uint64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    // GM Tensor
    AscendC::GlobalTensor<float> aGlobal_;
    AscendC::GlobalTensor<float> bGlobal_;
    AscendC::GlobalTensor<float> cGlobal_;
    AscendC::GlobalTensor<float> biasGlobal_;

    // Shape
    TupleShape problemShape_{};
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t b_{0};
    bool hasBias_ = false;

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

    __aicore__ inline void Init(const Params& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams_ = params.mmadParams;
        m_ = Get<MNK_M>(problemShape_);
        n_ = Get<MNK_N>(problemShape_);
        k_ = Get<MNK_K>(problemShape_);
        b_ = Get<MNK_B>(problemShape_);

        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.cGmAddr));
        if (blockMmadParams_.biasGmAddr != nullptr) {
            hasBias_ = true;
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.biasGmAddr));
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        if ASCEND_IS_AIC {
            return;
        }
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        // Get blockIdx 这里是硬件获得的blockidx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        // Get BlockNum 这里是rts获得的核数
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);

        BlockSchedulerOp bs(params.problemShape, params.schParams);
        int64_t realBlockNum = bs.GetRealBlockNum();
        TupleShape blockInfo = bs.GetBlockInfo();
        TupleShape tailInfo = bs.GetTailInfo();
        uint64_t baseM = Get<0>(blockInfo);
        uint64_t baseN = Get<1>(blockInfo);
        // 获取m分块数
        uint64_t mTileNum = Get<2>(blockInfo);
        // 获取n分块数
        uint64_t nTileNum = Get<3>(blockInfo);
        uint64_t tailM = Get<0>(tailInfo);
        uint64_t tailN = Get<1>(tailInfo);
        // 获取k方向尾块大小
        uint64_t tailK = Get<2>(tailInfo);
        // 获取k方向切分轮次
        uint64_t loopK = Get<3>(tailInfo);
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        blockMmadOp.Init(problemShape_, blockInfo, tailK, loopK, hasBias_);
        int64_t tileNum = bs.GetTileNum();
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            int64_t tileIdxN = tileIdx % nTileNum;
            int64_t tileIdxM = tileIdx / nTileNum;
            if (tileIdxM == mTileNum - 1) {
                blockMmadOp.SetTailM(tailM);
            } else {
                blockMmadOp.SetTailM(baseM);
            }
            if (tileIdxN == nTileNum - 1) {
                blockMmadOp.SetTailN(tailN);
            } else {
                blockMmadOp.SetTailN(baseN);
            }
            int64_t offsetA = tileIdxM * baseM * k_;
            int64_t offsetB = tileIdxN * baseN * k_;
            int64_t offsetC = tileIdxM * baseM * n_ + tileIdxN * baseN;
            blockMmadOp(cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB]);
        }
    }

    __aicore__ inline void operator()(const Params& params) { Run(params); }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
