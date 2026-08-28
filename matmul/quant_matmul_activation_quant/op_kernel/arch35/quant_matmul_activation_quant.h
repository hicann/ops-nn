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
 * \file quant_matmul_activation_quant.h
 * \brief
 */
#pragma once
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_gelu_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "quant_matmul_activation_quant_tiling_data.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

template <class A_TYPE, class B_TYPE, class C_TYPE, class aLayout, class bLayout, class cLayout,
          uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QuantMatmulActivationQuantKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                        GM_ADDR x2_scale, GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace,
                                                        const void* tilingData)
{
    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = float;
    using OutType = C_TYPE;
    using MatmulOutType = float;
    const uint64_t L0C2UBMode = 1;
    // 定义BlockEpilogue类型
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluMxQuant<OutType, MatmulOutType>;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, aLayout,
                                                                                bLayout, AType>;

    // 定义MMAD类型
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
        FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadWithScaleMxActivationQuant, L0C2UBMode>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, aLayout, BType, bLayout, MatmulOutType,
                                                    cLayout, BiasType, cLayout>;

    // 定义Kernel类型
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const QMMAQ::QuantMatmulActivationQuantTilingData* qmmaqTilingData_;
    qmmaqTilingData_ = static_cast<const QMMAQ::QuantMatmulActivationQuantTilingData*>(tilingData);
    DequantBmm::BasicAPICubeTiling matmulTiling = qmmaqTilingData_->mmTilingData.matmulTiling;
    DequantBmm::SlidingWindowParams slidingWindowParams = qmmaqTilingData_->mmTilingData.adaptiveSlidingWin;
    using QBMMTiling = typename MatmulKernel::QBMMTiling;
    QBMMTiling qbmmParams{qmmaqTilingData_->mmTilingData.params.batchA1,
                          qmmaqTilingData_->mmTilingData.params.batchA2,
                          qmmaqTilingData_->mmTilingData.params.batchA3,
                          qmmaqTilingData_->mmTilingData.params.batchA4,
                          qmmaqTilingData_->mmTilingData.params.batchB1,
                          qmmaqTilingData_->mmTilingData.params.batchB2,
                          qmmaqTilingData_->mmTilingData.params.batchB3,
                          qmmaqTilingData_->mmTilingData.params.batchB4,
                          qmmaqTilingData_->mmTilingData.params.batchC1,
                          qmmaqTilingData_->mmTilingData.params.batchC2,
                          qmmaqTilingData_->mmTilingData.params.batchC3,
                          qmmaqTilingData_->mmTilingData.params.batchC4,
                          qmmaqTilingData_->mmTilingData.params.biasThreeDim,
                          matmulTiling.baseM,
                          matmulTiling.baseN,
                          matmulTiling.baseK,
                          static_cast<uint32_t>(matmulTiling.isBias),
                          static_cast<uint32_t>(matmulTiling.dbL0C)};
    Params params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, qmmaqTilingData_->mmTilingData.params.batchC},
        {x1, x2, y, bias, x1_scale, x2_scale}, // gm addr
        {y, y_scale, matmulTiling.baseM, matmulTiling.baseN,
         static_cast<Blaze::Epilogue::Block::GeluAlg>(static_cast<uint8_t>(qmmaqTilingData_->activationType)),
         static_cast<Blaze::Epilogue::Block::QuantAlg>(static_cast<uint8_t>(qmmaqTilingData_->scaleAlg)),
         static_cast<Blaze::Epilogue::Block::ROUND_MODE_FP4>(static_cast<uint8_t>(qmmaqTilingData_->roundMode)),
         qmmaqTilingData_->dstTypeMax},
        {matmulTiling.kBL1, matmulTiling.scaleKL1, matmulTiling.nBufferNum},
        {matmulTiling.baseM, matmulTiling.baseN, slidingWindowParams.mTailTile, slidingWindowParams.nTailTile,
         slidingWindowParams.mBaseTailSplitCnt, slidingWindowParams.nBaseTailSplitCnt, slidingWindowParams.mTailMain,
         slidingWindowParams.nTailMain},
        qbmmParams};
    MatmulKernel qbmm;
    qbmm(params);
}
