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
 * \file kernel_matmul_mix_with_weight_prologue_nn.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_ARCH35_CATLASS_KERNEL_KERNEL_MATMUL_MIX_WITH_WEIGHT_PROLOGUE_NN_H
#define QUANT_BATCH_MATMUL_V4_ARCH35_CATLASS_KERNEL_KERNEL_MATMUL_MIX_WITH_WEIGHT_PROLOGUE_NN_H

#include "act/matmul/kernel/kernel_matmul_mix_with_weight_prologue.h"
#include "../../quant_batch_matmul_v4_tiling_data.h"

namespace Act {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockScheduler_, class BlockPrologue_>
class KernelMatmulMixWeightPrologueNN
    : public KernelMatmulMixWeightPrologue<ProblemShape_, BlockMmad_, BlockScheduler_, BlockPrologue_> {
public:
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using BlockPrologue = BlockPrologue_;
    using KernelAct = KernelMatmulMixWeightPrologue<ProblemShape_, BlockMmad_, BlockScheduler_, BlockPrologue_>;
    using Arguments = typename KernelAct::Arguments;
    using Params = typename KernelAct::Params;

    __aicore__ inline KernelMatmulMixWeightPrologueNN() = default;
    __aicore__ inline KernelMatmulMixWeightPrologueNN(const Params &params) : KernelAct(params) {}
    __host_aicore__ static Params ToUnderlyingArguments(
        Arguments const& args, const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tiling)
    {
        return {.problemShape = args.problemShape,
                .mmad = BlockMmad::ToUnderlyingArguments(args.problemShape, args.mmad, tiling),
                .prologue = BlockPrologue::ToUnderlyingArguments(args.problemShape, args.prologue, tiling),
                .scheduler = BlockScheduler::ToUnderlyingArguments(args.problemShape, args.scheduler, tiling)};
    }
};
}  // namespace Kernel
}  // namespace Gemm
}  // namespace Act
#endif