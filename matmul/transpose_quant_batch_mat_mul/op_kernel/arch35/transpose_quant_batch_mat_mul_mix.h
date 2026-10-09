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
 * \file transpose_quant_batch_mat_mul_mix.h
 * \brief TQBMM MIX tensor API wrapper (FP8 per-token dequant, AIC+AIV)
 */
#pragma once
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_universal.h"

template <class A_TYPE, class B_TYPE, class X2_SCALE_TYPE, class X1_SCALE_TYPE, class C_TYPE, class BIAS_TYPE,
          class aLayout, class bLayout, class cLayout, uint64_t FULL_LOAD_MODE = 0, uint64_t PERM_X1 = 0,
          uint64_t NON_CONTIGUOUS_TYPE = 0>
__aicore__ inline void TqbmmMixTensorApiKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias,
                                               GM_ADDR perTokenScale, GM_ADDR cGM,
                                               const BatchMatMulV3TilingData& tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using BiasType = BIAS_TYPE;
    using X2ScaleType = X2_SCALE_TYPE;
    using X1ScaleType = X1_SCALE_TYPE;

    using BTypeTuple = AscendC::Std::tuple<BType, uint64_t>;

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, aLayout,
                                                                                bLayout, AType>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<
        FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadMultiBlockTQBMMMx, NON_CONTIGUOUS_TYPE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, aLayout, BTypeTuple, bLayout, OutType,
                                                    cLayout, BiasType, asc::te::nd_ext_layout_ptn>;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<OutType, BiasType, X2ScaleType, X1ScaleType,
                                                                       float>;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    using MmadParams = typename BlockMmad::Params;
    using EpilogueParams = typename BlockEpilogue::Params;

    const auto& tCubeTiling = tilingData.matMulTilingData.tCubeTiling;

    constexpr uint32_t x1QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERTOKEN_MODE);
    constexpr uint32_t x2QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERCHANNEL_MODE);
#if defined(ORIG_DTYPE_BIAS)
    constexpr uint32_t biasDtypeVal = static_cast<uint32_t>(ORIG_DTYPE_BIAS);
#else
    constexpr uint32_t biasDtypeVal = 0U;
#endif

    // kAL1/kBL1 are ELEMENTS of K per L1 load (depthA1/depthB1 are depth COUNTS, cf. fixpipe wrapper)
    const uint32_t kAL1Elems = static_cast<uint32_t>(tCubeTiling.stepKa) * tCubeTiling.baseK;
    const uint32_t kBL1Elems = static_cast<uint32_t>(tCubeTiling.stepKb) * tCubeTiling.baseK;

    Params params;
    params.problemShape = {tCubeTiling.M, tCubeTiling.N, tCubeTiling.Ka, tilingData.cBatchDimAll};

    MmadParams mmadParams;
    mmadParams.aGmAddr = aGM;
    mmadParams.bGmAddr = bGM;
    mmadParams.problemShape = params.problemShape;
    mmadParams.l0TileShape = BlockShape{static_cast<int64_t>(tCubeTiling.baseM),
                                        static_cast<int64_t>(tCubeTiling.baseN),
                                        static_cast<int64_t>(tCubeTiling.baseK), 0};
    mmadParams.kAL1 = static_cast<uint64_t>(kAL1Elems);
    mmadParams.kBL1 = static_cast<uint64_t>(kBL1Elems);
    mmadParams.l1BufferNum = tilingData.l1BufferNum;
    mmadParams.enableL0CPingPong = tCubeTiling.dbL0C > 1;
    params.mmadParams = mmadParams;

    params.schParams = {static_cast<int64_t>(tCubeTiling.baseM),       static_cast<int64_t>(tCubeTiling.baseN),
                        tilingData.matMulTilingData.mTailCnt,          tilingData.matMulTilingData.nTailCnt,
                        tilingData.matMulTilingData.mBaseTailSplitCnt, tilingData.matMulTilingData.nBaseTailSplitCnt,
                        tilingData.matMulTilingData.mTailMain,         tilingData.matMulTilingData.nTailMain};

    params.tqbmmParams = {kAL1Elems,
                          kBL1Elems,
                          tilingData.l1BufferNum,
                          static_cast<uint32_t>(tCubeTiling.baseM),
                          static_cast<uint32_t>(tCubeTiling.baseN),
                          static_cast<uint32_t>(tCubeTiling.baseK),
                          static_cast<uint32_t>(tCubeTiling.dbL0C),
                          1};

    EpilogueParams epilogueParams;
    epilogueParams.x2ScaleGmAddr = scale;
    epilogueParams.x1ScaleGmAddr = perTokenScale;
    epilogueParams.biasGmAddr = bias;
    epilogueParams.outGmAddr = cGM;
    epilogueParams.m = tCubeTiling.M;
    epilogueParams.n = tCubeTiling.N;
    epilogueParams.baseM = tCubeTiling.baseM;
    epilogueParams.baseN = tCubeTiling.baseN;
    epilogueParams.x1QuantMode = x1QuantMode;
    epilogueParams.x2QuantMode = x2QuantMode;
    epilogueParams.isBias = tCubeTiling.isBias != 0;
    epilogueParams.biasDtype = biasDtypeVal;
    epilogueParams.batch = tilingData.cBatchDimAll;
    params.epilogueParams = epilogueParams;

    MatmulKernel mm;
    mm(params);
}
