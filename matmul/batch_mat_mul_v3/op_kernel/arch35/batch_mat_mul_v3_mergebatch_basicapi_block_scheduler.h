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
 * \file batch_mat_mul_v3_mergebatch_basicapi_block_scheduler.h
 * \brief
 */

#ifndef BATCH_MAT_MUL_V3_MERGEBATCH_BASICAPI_BLOCK_SCHEDULER_H
#define BATCH_MAT_MUL_V3_MERGEBATCH_BASICAPI_BLOCK_SCHEDULER_H

#include "matmul_act/matmul/block/block_scheduler_utils.h"
#include "matmul_act/matmul/block/block_scheduler_policy.h"
#include "matmul_act/utils/status_utils.h"
#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"

namespace Act {
namespace Gemm {
namespace Block {

template <
    class ProblemShape_,
    class L1TileShape_,
    class L0TileShape_
>
class BlockSchedulerMergeBatchBuiltIn {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t b_{0};
    int64_t k_{0};
    int64_t batchAL1_{1};
    int64_t batchBL1_{1};
    int64_t batchL0ab_{1};
    int64_t batchL0_{1};
    int64_t isHf32_{0};
    int64_t kL1_{1};
    int64_t baseK_{1};
    int64_t blockNum_{1};
    int64_t mainBatchLoop_{1};
    int64_t mainTailBatch_{1};
    int64_t mainTailBlock_{1};

    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        const BatchMatMulV3MergeBatchBasicTilingData* tilingData;
    };

public:
    __aicore__ inline BlockSchedulerMergeBatchBuiltIn(const ProblemShape& shape, int64_t blockNum,
                                                     const Params& params)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        b_ = shape.b;
        batchAL1_ = params.tilingData->batchAL1;
        batchBL1_ = params.tilingData->batchBL1;
        batchL0_ = params.tilingData->batchL0;
        kL1_ = params.tilingData->kL1;
        baseK_ = params.tilingData->baseK;
        isHf32_ = params.tilingData->isHf32;
        blockNum_ = blockNum;
        // 做负载均衡
        int64_t mainBatchNum = b_ / batchAL1_;
        mainBatchLoop_ = mainBatchNum / blockNum_;
        int64_t remainderBatch = b_ - mainBatchLoop_ * blockNum_ * batchAL1_;
        mainTailBatch_ = MMV3DivCeil(remainderBatch, blockNum_);
        mainTailBlock_ = remainderBatch % blockNum_;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return MMV3CeilAlign(MMV3DivCeil(b_, batchAL1_), blockNum_);
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetIterBatchTuple()
    {
        return {batchAL1_, batchBL1_, batchL0_, batchL0_};
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL1Shape()
    {
        return {0, 0, kL1_, batchAL1_};
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL0Shape()
    {
        return {0, 0, baseK_, batchL0_};
    }
    __aicore__ inline int64_t GetHf32Flag()
    {
        return isHf32_;
    }

    __aicore__ inline int64_t GetBlockNum(ProblemShape shape, int64_t blockNum)
    {
        int64_t tilingBlockNum = 0;
        if (GetTileNum() < blockNum) {
            tilingBlockNum = GetTileNum();
        } else {
            tilingBlockNum = blockNum;
        }
        return tilingBlockNum;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        int64_t curLoopIdx = tileIdx / blockNum_;
        // 如果是主块
        if (curLoopIdx < mainBatchLoop_) {
            return {0, 0, 0, batchAL1_};
        } else if (mainTailBatch_ > 0) {
            int64_t mainTailIdx = tileIdx % blockNum_;
            if (mainTailBlock_ > 0 && mainTailIdx >= mainTailBlock_) {
                // 如果是最终尾块
                return {0, 0, 0, mainTailBatch_ - 1};
            } else {
                // 如果是主尾块
                return {0, 0, 0, mainTailBatch_};
            }
        }
        return {0, 0, 0, batchAL1_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        int64_t curLoopIdx = tileIdx / blockNum_;
        // 如果是主块
        if (curLoopIdx < mainBatchLoop_) {
            return {0, 0, 0, tileIdx * batchAL1_};
        } else if (mainTailBatch_ > 0) {
            int64_t mainTailIdx = tileIdx % blockNum_;
            if (mainTailBlock_ > 0 && mainTailIdx >= mainTailBlock_) {
                // 如果是最终尾块
                return {0,
                    0,
                    0,
                    mainBatchLoop_ * batchAL1_ * blockNum_ + mainTailBlock_ * mainTailBatch_ +
                        (mainTailBatch_ - 1) * (mainTailIdx - mainTailBlock_)};
            } else {
                // 如果是主尾块
                return {0, 0, 0, mainBatchLoop_ * batchAL1_ * blockNum_ + mainTailIdx * mainTailBatch_};
            }
        }
        return {0, 0, 0, tileIdx * batchAL1_};
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
    Act::Gemm::BuiltInMergeBatchScheduler,
    TransA_,
    TransB_
> {
  using SchedulerOp = BlockSchedulerMergeBatchBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Block
} // namespace Gemm
} // namespace Act
#endif