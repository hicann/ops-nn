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
 * \file block_prologue_b_cast_scsc_nn.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_ARCH35_CATLASS_PROLOGUE_BLOCK_PROLOGUE_B_CAST_SCSC_NN_H
#define QUANT_BATCH_MATMUL_V4_ARCH35_CATLASS_PROLOGUE_BLOCK_PROLOGUE_B_CAST_SCSC_NN_H
#include "act/prologue/block_prologue_b_cast_scsc.h"
#include "../../quant_batch_matmul_v4_tiling_data.h"

namespace Act {
namespace Prologue {
template <class DispatchPolicy, class InType, class OutType, class TileShapeL1>
class BlockPrologueNN : public BlockPrologue<DispatchPolicy, InType, OutType, TileShapeL1> {
public:
    using PrologueAct = BlockPrologue<DispatchPolicy, InType, OutType, TileShapeL1>;
    using Arguments = typename PrologueAct::Arguments;
    using Params = typename PrologueAct::Params;

    __aicore__ inline BlockPrologueNN() = default;
    __aicore__ inline BlockPrologueNN(const Params &params) : PrologueAct(params) {}

    template <class ProblemShape>
    __aicore__ inline static Params ToUnderlyingArguments(
        ProblemShape const& problemShape, Arguments const& args,
        const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tiling)
    {
        auto stepKa = static_cast<uint32_t>(tiling->matmulTiling.stepKa);
        auto stepKb = static_cast<uint32_t>(tiling->matmulTiling.stepKa);
        auto baseK = static_cast<uint32_t>(tiling->matmulTiling.baseK);
        auto baseM = static_cast<uint32_t>(tiling->matmulTiling.baseM);
        auto baseN = static_cast<uint32_t>(tiling->matmulTiling.baseN);
        return {.ptrB = args.ptrB,
                .tileShapeL1 = AscendC::MakeShape(baseM, baseN, stepKa * baseK, stepKb * baseK),
                .layoutB = args.layoutB,
                .l1BufNum = tiling->BL1Pingpong,
                .nUbSize = static_cast<int32_t>(tiling->nBubSize),
                .kUbSize = static_cast<int32_t>(tiling->kBubSize)};
    }
};
}  // namespace Prologue
}  // namespace Act

#endif