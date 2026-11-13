/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_mat_mul_elemwise.h
 * \brief FusedMatMul MM add/mul Blaze 组装(TENSOR_LEVEL)——对齐 mat_mul_pingpong_basic.h 的独立组装文件模式.
 * 基本路径挂 BlockEpilogueMatmulElemwise(basic专用独立epilogue); streamK直连
 * BlockEpilogueMatmulStreamK(Run按FUSED_OP_TYPE分派ADD/MUL融合分支).
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "blaze/epilogue/block/block_epilogue_matmul_elemwise.h"
#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "tensor_api/tensor.h"

// MatMulV3BasicTilingData 与 MAT_MUL_* 模板宏由宿主 fused_mat_mul.cpp 的 include 链提供
// (对齐 mat_mul_pingpong_basic.h 的薄依赖模式).

// MM add/mul Blaze assembly: out = matmul(A,B) (+|*) x3, dispatches to fixpipe_opti /
// al1 / bl1 / streamK kernels with the BlockEpilogueMatmulElemwise epilogue.
namespace FusedMatMulAdvanced {
// inner_precise = 0(高精度)累加器保持FP32并FP32域融合; 1(高性能)在元素域融合,
// 对齐legacy MatMulMixWithoutQueActKernel的MmOutTypeSelector语义.
template <typename CType, int8_t InnerPrecise>
struct FmmAccTypeSelector {
    static constexpr bool useFloat = (InnerPrecise == 0);
    using type = AscendC::Std::conditional_t<useFloat, float, CType>;
};

// basic路径: 按FULL_LOAD选policy, FusedOpType透传给epilogue.
template <typename AType, typename BType, typename CType, typename BiasType, bool ATran, bool BTran,
          uint64_t FusedOpType, int8_t InnerPrecise, uint64_t L0C2OutMode, uint64_t FullLoadMode>
__aicore__ inline void FusedMatMulAddMulKernel(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM, GM_ADDR yGM,
                                               GM_ADDR workspaceGM, const MatMulV3BasicTilingData& tiling)
{
    using AccType = typename FmmAccTypeSelector<CType, InnerPrecise>::type;
    using LayoutA = AscendC::Std::conditional_t<ATran, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<BTran, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    // B全载: bl1_full_load policy; A全载: al1_full_load policy; 其余: fixpipe_opti
    using DispatchPolicy = AscendC::Std::conditional_t<
        FullLoadMode == MAT_MUL_B_FULL_LOAD, Blaze::Gemm::MatmulMultiBlockBFullLoad<L0C2OutMode, FusedOpType>,
        AscendC::Std::conditional_t<FullLoadMode == MAT_MUL_A_FULL_LOAD,
                                    Blaze::Gemm::MatmulMultiBlockAFullLoad<MAT_MUL_A_FULL_LOAD, FusedOpType>,
                                    Blaze::Gemm::MatmulMultiBlockFixpipeOpti<L0C2OutMode, FusedOpType>>>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FullLoadMode, false, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, AccType, Layout,
                                                    BiasType, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulElemwise<
        DispatchPolicy, CType, AccType, /*FusionSameDomain=*/InnerPrecise == F_INNER_PRECISE_HIGH_PERFORMANCE>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    // FP32累加器需要两个AIV分M, 使累加镜像+x3/输出staging能放进UB
    // mmad侧对b16改走两条单目的地fixpipe, 双AIV各处理M/2行
    const uint8_t splitM = FullLoadMode == MAT_MUL_NO_FULL_LOAD ?
                               1U :
                               (AscendC::IsSameType<AccType, float>::value ? 1U : 0U);
    // ubPitchGran恒0: 行距由AccType自然表达; 按1V2传16与bl1的c0写入错配会行错位.
    constexpr uint32_t ubPitchGran = 0U;

    // BlockMmad::Params布局: al1无workspace; bl1含workspace/splitM; fixpipe_opti再加ubPitchGran.
    typename BlockMmad::Params mmadParams = {};
    if constexpr (FullLoadMode == MAT_MUL_A_FULL_LOAD) {
        mmadParams = {x1GM,
                      x2GM,
                      nullptr,
                      biasGM,
                      tiling.k,
                      tiling.mL1,
                      tiling.kL1,
                      tiling.baseM,
                      tiling.baseN,
                      tiling.baseK,
                      tiling.l1BufferNum,
                      tiling.l0cDB,
                      tiling.rowStride,
                      splitM,
                      1U};
    } else if constexpr (FullLoadMode == MAT_MUL_B_FULL_LOAD) {
        mmadParams = {
            x1GM,         x2GM,       nullptr,    biasGM,          nullptr,      workspaceGM,  tiling.k,
            tiling.mL1,   tiling.nL1, tiling.kL1, tiling.baseM,    tiling.baseN, tiling.baseK, tiling.l1BufferNum,
            tiling.l0cDB, splitM,     1U,         tiling.rowStride};
    } else {
        mmadParams = {
            x1GM,         x2GM,       nullptr,    biasGM,           nullptr,      workspaceGM,  tiling.k,
            tiling.mL1,   tiling.nL1, tiling.kL1, tiling.baseM,     tiling.baseN, tiling.baseK, tiling.l1BufferNum,
            tiling.l0cDB, splitM,     1U,         tiling.rowStride, ubPitchGran};
    }

    KernelParams params = {
        {tiling.m, tiling.n, tiling.k, 1},
        mmadParams,
        {},
        {tiling.mL1, tiling.nL1, tiling.kL1, tiling.baseM, tiling.baseN, tiling.baseK, tiling.mTailCnt, tiling.nTailCnt,
         tiling.mBaseTailSplitCnt, tiling.nBaseTailSplitCnt, tiling.mTailMain, tiling.nTailMain,
         static_cast<uint8_t>(tiling.mmadParam > 0 ? 1 : 0), static_cast<uint32_t>(tiling.l2CacheDisable),
         tiling.sliceM, tiling.srcNdStride, tiling.innerBatch, tiling.aswWindowLen}};
    // epilogue params: params-only Init契约(problemShape随Params, 对齐主线单参方向),
    // epilogue无workspace字段
    params.epilogueParams.cGmAddr = yGM;
    params.epilogueParams.x3GmAddr = x3GM;
    params.epilogueParams.x3BatchBroadcast = false;
    params.epilogueParams.x3M = tiling.m;
    params.epilogueParams.problemShape = params.problemShape;

    MatmulKernel kernel;
    kernel(params);
}

// streamK路径: 直连BlockEpilogueMatmulStreamK(七参Init主线契约含x3记账, kernel未改动),
// Run()按FUSED_OP_TYPE走ADD/MUL融合分支(workspace读取+K-split归约复用既有段)
template <typename AType, typename BType, typename CType, typename BiasType, bool ATran, bool BTran,
          uint64_t FusedOpType, int8_t InnerPrecise, uint64_t L0C2OutMode>
__aicore__ inline void FusedMatMulAddMulStreamKKernel(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM,
                                                      GM_ADDR yGM, GM_ADDR workspaceGM,
                                                      const MatMulV3BasicTilingData& tiling)
{
    using AccType = typename FmmAccTypeSelector<CType, InnerPrecise>::type;
    using LayoutA = AscendC::Std::conditional_t<ATran, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<BTran, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<
        static_cast<Blaze::Gemm::MatMulL0C2Out>(L0C2OutMode), FusedOpType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, float, Layout,
                                                    BiasType, Layout>;
    // streamK: 直连基类, 第4参传融合域判断.
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<
        float, CType, DispatchPolicy, AscendC::IsSameType<AccType, CType>::value>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    KernelParams params = {{tiling.m, tiling.n, tiling.k, 1},
                           {x1GM, x2GM, nullptr, biasGM, nullptr, workspaceGM, tiling.mL1, tiling.nL1, tiling.kL1,
                            tiling.baseM, tiling.baseN, tiling.baseK, tiling.l1BufferNum, tiling.l0cDB},
                           {},
                           {tiling.usedCoreNum, tiling.baseM, tiling.baseN, tiling.baseK, tiling.skSingleCoreK,
                            tiling.kL1, static_cast<uint8_t>(tiling.mmadParam > 0 ? 1 : 0),
                            static_cast<uint32_t>(tiling.l2CacheDisable), tiling.aswWindowLen}};
    // epilogue params: x3三件套随基类Params下发, windowLen由streamK kernel侧填充
    params.epilogueParams.cGmAddr = yGM;
    params.epilogueParams.workspaceGmAddr = workspaceGM;
    params.epilogueParams.x3GmAddr = x3GM;
    params.epilogueParams.x3BatchBroadcast = false;
    params.epilogueParams.x3M = tiling.m;

    MatmulKernel kernel;
    kernel(params);
}
} // namespace FusedMatMulAdvanced
