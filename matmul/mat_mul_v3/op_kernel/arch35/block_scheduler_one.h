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
 * \file block_scheduler_one.h
 * \brief
 */

#ifndef CMCT_BLOCK_SCHEDULER_ONE_BUILTIN_H
#define CMCT_BLOCK_SCHEDULER_ONE_BUILTIN_H

#include "cmct/block/block_scheduler_utils.h"
#include "cmct/block/block_scheduler_policy.h"
#include "mat_mul_tiling_data.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <
    class ProblemShape_,
    class L1TileShape_,
    class L0TileShape_
>
class BlockSchedulerOneBuiltIn {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t useAllCoreNum_{0};
    int64_t usedCoreNum_{0};
    int64_t loopK_{0};
    int64_t tailMN_{0};
    bool hasBias_{false};

    using ProblemShape = ProblemShape_;

    struct Params {
        const MatMulV3MNEqOneBasicTilingData* tilingData;
    };
public:
    __aicore__ inline BlockSchedulerOneBuiltIn(const ProblemShape& shape, const Params& params)
    {
        useAllCoreNum_ = params.tilingData->useAllCoreNum;
        usedCoreNum_ = params.tilingData->usedCoreNum;
        loopK_ = params.tilingData->loopK;
        tailMN_ = params.tilingData->tailMN;
        hasBias_ = params.tilingData->hasBias;
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
    }

    __aicore__ inline int64_t GetRealBlockNum()
    {
        return usedCoreNum_;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return useAllCoreNum_;
    }

    __aicore__ inline int64_t GetLoopK()
    {
        return loopK_;
    }

    __aicore__ inline int64_t GetTailMN()
    {
        return tailMN_;
    }

    __aicore__ inline bool GetBias()
    {
        return hasBias_;
    }
};

template <
    class ProblemShape_,
    class L1TileShape_,
    class L0TileShape_,
    bool TransA_,
    bool TransB_>
struct BlockSchedulerSelector<
    ProblemShape_,
    L1TileShape_,
    L0TileShape_,
    Cmct::Gemm::BuiltInOneScheduler,
    TransA_,
    TransB_
> {
using SchedulerOp = BlockSchedulerOneBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif