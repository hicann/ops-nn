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
 * \file block_scheduler_mul.h
 * \brief
 */

#pragma once

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
class BlockSchedulerMulBuiltIn {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t tileNum_{0};
    int64_t usedCoreNum_{0};
    int64_t baseMN_{0};
    int64_t tailMN_{0};
    int64_t baseK_{0};
    int64_t tailK_{0};
    int64_t loopK_{0};
    int64_t loopM_{0};
    bool dataCopyMode_{false};

    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        const MatMulToMulBasicTilingData* tilingData;
    };
public:
    __aicore__ inline BlockSchedulerMulBuiltIn(const ProblemShape& shape, const Params& params)
    {
        tileNum_ = params.tilingData->tileNum;
        usedCoreNum_ = params.tilingData->usedCoreNum;
        baseMN_ = params.tilingData->baseMN;
        tailMN_ = params.tilingData->tailMN;
        baseK_ = params.tilingData->baseK;
        tailK_ = params.tilingData->tailK;
        loopK_ = params.tilingData->loopK;
        loopM_ = params.tilingData->loopM;
        dataCopyMode_ = params.tilingData->dataCopyMode;
        n_ = shape.n;
        k_ = shape.k;
    }

    __aicore__ inline int64_t GetRealBlockNum()
    {
        return usedCoreNum_;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return tileNum_;
    }

    __aicore__ inline int64_t GetLoopK()
    {
        return loopK_;
    }

    __aicore__ inline int64_t GetLoopM()
    {
        return loopM_;
    }

    __aicore__ inline bool GetDataCopyMode()
    {
        return dataCopyMode_;
    }

    __aicore__ inline TupleShape GetBlockInfo()
    {
        return {baseMN_, tailMN_, baseK_, tailK_};
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
    Cmct::Gemm::BuiltInMulScheduler,
    TransA_,
    TransB_
> {
using SchedulerOp = BlockSchedulerMulBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct

