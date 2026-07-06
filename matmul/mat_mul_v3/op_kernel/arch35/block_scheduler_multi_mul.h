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
 * \file block_scheduler_multi_mul.h
 * \brief
 */

#pragma once

#include "cmct/block/block_scheduler_utils.h"
#include "cmct/block/block_scheduler_policy.h"
#include "mat_mul_tiling_data.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class BlockSchedulerVectorBuiltIn {
public:
    int64_t usedCoreNum_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t baseM_{0};
    int64_t tailM_{0};
    int64_t baseN_{0};
    int64_t tailN_{0};
    int64_t baseK_{0};
    int64_t tailK_{0};
    int64_t loopK_{0};
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};

    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        const MatMulToVectorBasicTilingData* tilingData;
    };

public:
    __aicore__ inline BlockSchedulerVectorBuiltIn(const ProblemShape& shape, const Params& params)
    {
        usedCoreNum_ = params.tilingData->usedCoreNum;
        baseM_ = params.tilingData->baseM;
        baseN_ = params.tilingData->baseN;
        baseK_ = params.tilingData->baseK;
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        tailM_ = m_ % baseM_ == 0 ? baseM_ : m_ % baseM_;
        tailN_ = n_ % baseN_ == 0 ? baseN_ : n_ % baseN_;
        tailK_ = k_ % baseK_ == 0 ? baseK_ : k_ % baseK_;
        loopK_ = CeilDiv(k_, baseK_);
        mTileNum_ = CeilDiv(m_, baseM_);
        nTileNum_ = CeilDiv(n_, baseN_);
    }

    __aicore__ inline int64_t GetRealBlockNum() { return usedCoreNum_; }

    __aicore__ inline int64_t GetTileNum() { return mTileNum_ * nTileNum_; }

    __aicore__ inline TupleShape GetBlockInfo() { return {baseM_, baseN_, mTileNum_, nTileNum_}; }

    __aicore__ inline TupleShape GetTailInfo() { return {tailM_, tailN_, tailK_, loopK_}; }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Cmct::Gemm::BuiltInVectorScheduler, TransA_,
                              TransB_> {
    using SchedulerOp = BlockSchedulerVectorBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
