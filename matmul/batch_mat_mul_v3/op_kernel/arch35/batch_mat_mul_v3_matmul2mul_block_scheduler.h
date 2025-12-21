/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_matmul2mul_tiling.h
 * \brief
 */

#ifndef BATCH_MAT_MUL_V3_MATMUL2MUL_BLOCK_SCHEDULER_H
#define BATCH_MAT_MUL_V3_MATMUL2MUL_BLOCK_SCHEDULER_H

#include "matmul_act/matmul/block/block_scheduler_policy.h"
#include "matmul_act/matmul/block/block_scheduler_utils.h"
#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"

namespace Act {
namespace Gemm {
namespace Block {

template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class BlockSchedulerBatchMatMulToMulBuiltIn {
public:
    uint64_t batchNum_{0};
    uint64_t usedCoreNum_{0};
    uint64_t b_{0};

    using ProblemShape = ProblemShape_;
    using BatchShape = Shape<uint64_t, uint64_t, uint64_t, uint64_t>;
    using ParamsShape = Shape<uint64_t, uint64_t, uint64_t>;

    struct Params {
        const BatchMatMulToMulBasicTilingData* tilingData;
    };

    __aicore__ inline BlockSchedulerBatchMatMulToMulBuiltIn(
        const ProblemShape& shape, int64_t blockIdx, int64_t blockNum, const Params& params)
    {
        b_ = shape.b;
        usedCoreNum_ = params.tilingData->usedCoreNum;
        batchNum_ = params.tilingData->batchNum;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return usedCoreNum_ * MMV3DivCeil(b_, batchNum_ * usedCoreNum_);
    }

    __aicore__ inline ParamsShape GetParams(const Params& params)
    {
        return {usedCoreNum_, params.tilingData->singleCoreBatch, params.tilingData->alignNum};
    }

    __aicore__ inline BatchShape GetBtachShape(const Params& params)
    {
        return {
            batchNum_, params.tilingData->batchNumLastRound, params.tilingData->batchNumLastRoundTail,
            params.tilingData->lastCoreNum};
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<
    ProblemShape_, L1TileShape_, L0TileShape_, Act::Gemm::BuiltInBatchMatmulToMulScheduler, TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerBatchMatMulToMulBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Block
} // namespace Gemm
} // namespace Act

#endif
