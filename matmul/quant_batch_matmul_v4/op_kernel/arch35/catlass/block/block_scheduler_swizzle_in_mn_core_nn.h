/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_swizzle_in_mn_core_nn.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_ARCH35_CATLASS_BLOCK_BLOCK_SCHEDULER_SWIZZLE_IN_MN_CORE_NN_H
#define QUANT_BATCH_MATMUL_V4_ARCH35_CATLASS_BLOCK_BLOCK_SCHEDULER_SWIZZLE_IN_MN_CORE_NN_H
#include "act/matmul/block/block_scheduler_swizzle_in_mn_core.h"
#include "../../quant_batch_matmul_v4_tiling_data.h"
/*
iterateOrder = 0
scheduler diagram c:core b:block
| c0b0 | c0b1 | c0b2 | c2b0 | c2b1 |
------------------------------------
| c0b3 | c0b4 | c0b5 | c2b2 | c2b3 |
------------------------------------
| c1b0 | c1b1 | c1b2 | c3b0 | c3b1 |f

iterateOrder = 1
| c0b0 | c0b2 | c0b4 | c2b0 | c2b2 |
------------------------------------
| c0b1 | c0b3 | c0b5 | c2b1 | c2b3 |
------------------------------------
| c1b0 | c1b1 | c1b2 | c3b0 | c3b1 |
*/
namespace Act {
namespace Gemm {
namespace Block {

template <class ProblemShape_, class TileShape_, class BlockShape_>
class BlockSchedulerSwizzleInMnCoreNN : public BlockSchedulerSwizzleInMnCore<ProblemShape_, TileShape_, BlockShape_> {
public:
    using ProblemShape = ProblemShape_;
    using KernelAct = BlockSchedulerSwizzleInMnCore<ProblemShape_, TileShape_, BlockShape_>;
    using Arguments = typename KernelAct::Arguments;
    using Params = typename KernelAct::Params;
    using BlockSchedulerSwizzleInMnCoreAct = BlockSchedulerSwizzleInMnCore<ProblemShape_, TileShape_, BlockShape_>;
    __aicore__ inline BlockSchedulerSwizzleInMnCoreNN() = delete;
    __aicore__ inline BlockSchedulerSwizzleInMnCoreNN(const Params &params) : BlockSchedulerSwizzleInMnCoreAct(params)
    {
    }

    __aicore__ inline static Params ToUnderlyingArguments(
        ProblemShape const& problemShape, [[maybe_unused]] Arguments const& args,
        const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tiling)
    {
        auto baseM = static_cast<uint32_t>(tiling->matmulTiling.baseM);
        auto baseN = static_cast<uint32_t>(tiling->matmulTiling.baseN);
        return {.iterateOrder = tiling->matmulTiling.iterateOrder,
                .problemShape = problemShape,
                .tileShape = AscendC::MakeShape(baseM, baseN),
                .blockShape = AscendC::MakeShape(tiling->cubeBlockDimM, tiling->cubeBlockDimN)};
    }
};
}  // namespace Block
}  // namespace Gemm
}  // namespace Act
#endif