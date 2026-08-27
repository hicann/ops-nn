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
 * \file flat_quant.h
 * \brief
 */

#pragma once

#include "blaze/attention/kernel/kernel_universal.h"
#include "blaze/attention/block/block_mmad.h"
#include "blaze/attention/block/block_scheduler_flat_quant.h"
#include "blaze/attention/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_flat_quant.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"
#include "tensor_api/tensor.h"

namespace FlatQuantNS {

template <class X_TYPE, class Y_TYPE, class SCALE_TYPE, class C_LAYOUT>
__aicore__ inline void FlatQuantBlazeKernel(GM_ADDR aGM, GM_ADDR p1GM, GM_ADDR p2GM, GM_ADDR cGM, GM_ADDR scaleGM,
                                            GM_ADDR workspaceGM, const FlatQuantTilingData& tilingData)
{
    using AType = X_TYPE;
    using BType = X_TYPE;
    using BiasType = SCALE_TYPE;
    using OutType = Y_TYPE;

    using LayoutA = C_LAYOUT;
    using LayoutB = C_LAYOUT;
    using LayoutC = C_LAYOUT;

    using AMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, false>;
    using BMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false>;
    using CMatmulType = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, OutType>;
    using BiasMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;

    using BlockScheduler = Blaze::Attention::Block::BlockSchedulerFlatQuant<
        AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>>;
    using DispatchPolicy = Blaze::Attention::BlockFlatQuant<Blaze::Attention::KernelFlatQuant>;
    using BlockMmad = Blaze::Attention::Block::BlockMmad<DispatchPolicy, AMatmulType, LayoutA, BMatmulType, LayoutB,
                                                         BiasMatmulType, LayoutC, CMatmulType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFlatQuant<AType, OutType, BiasType>;
    using FusionOp = Blaze::Epilogue::Fusion::DefaultFusion<OutType, AType>;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using MatmulKernel = Blaze::Attention::Kernel::AttentionUniversal<ProblemShape, BlockMmad, BlockEpilogue,
                                                                      BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    typename BlockScheduler::Params schParams;
    schParams.iterBatch = tilingData.iterBatch;
    schParams.dstTypeMax = tilingData.dstTypeMax;
    schParams.invDstTypeMax = tilingData.invDstTypeMax;

    constexpr int64_t BASE_K = 64;
    int64_t M = tilingData.M;
    int64_t N = tilingData.N;
    int64_t K = tilingData.K;
    int64_t iterBatch = tilingData.iterBatch;

    Params params = {{M, N, N, K},
                     {aGM,
                      p1GM,
                      p2GM,
                      {M, N, N, K},
                      {M * iterBatch, N, N, iterBatch},
                      {M * iterBatch, N, BASE_K, 1},
                      tilingData.hasP2 == 1},
                     {cGM, scaleGM, {M, N, N, K}, tilingData.dstTypeMax, tilingData.invDstTypeMax},
                     schParams};

    MatmulKernel mm;
    mm(params);
}
} // namespace FlatQuantNS
