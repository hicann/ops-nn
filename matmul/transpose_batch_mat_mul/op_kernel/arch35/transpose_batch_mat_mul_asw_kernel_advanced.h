/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_batch_mat_mul_asw_kernel_advanced.h
 * \brief
 */
#ifndef TRANSPOSE_BATCH_MAT_MUL_ASW_KERNEL_ADVANCED_H
#define TRANSPOSE_BATCH_MAT_MUL_ASW_KERNEL_ADVANCED_H

#include "transpose_batch_mat_mul_asw_block_advanced.h"
#include "kernel_basic_intf.h"
#include "lib/matmul_intf.h"

namespace TransposeBatchMatMulAdvanced {

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, TBMM_MODE MODE,
          class BLOCK_TYPE = TransposeBatchMatMulAswBlock, const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD>
class TransposeBatchMatMulAswKernel {
public:
    __aicore__ inline TransposeBatchMatMulAswKernel() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR scalesGM,
        GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR scalesGM,
        GM_ADDR workspaceGM);
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR scalesGM);
    __aicore__ inline void Process(uint8_t enAtomic = 0);
    __aicore__ inline void End() { mm_.End(); }
    __aicore__ inline const BLOCK_TYPE GetBlock() { return block_; }
    __aicore__ inline void SetOrgShape();

protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    GlobalTensor<uint64_t> scalesGlobal_;
    TPipe *pipe_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, TBMM_MODE MODE, class BLOCK_TYPE,
          const MatmulConfig& MM_CFG>
__aicore__ inline void TransposeBatchMatMulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR scalesGM, GM_ADDR workspaceGM,
    const void* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    InitInputs(aGM, bGM, cGM, biasGM, scalesGM);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling, pipe_);
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, TBMM_MODE MODE, class BLOCK_TYPE,
          const MatmulConfig& MM_CFG>
__aicore__ inline void
TransposeBatchMatMulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR scalesGM)
{
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), block_.params_.aBatchDimAll *
        static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.M) *
        static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka));
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), block_.params_.bBatchDimAll *
        static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb) *
        static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N));
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), block_.params_.cBatchDimAll *
        static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.M) *
        static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N));
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM),
        block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, TBMM_MODE MODE, class BLOCK_TYPE,
          const MatmulConfig& MM_CFG>
__aicore__ inline void
TransposeBatchMatMulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG>::SetOrgShape()
{
    if constexpr (MODE == TBMM_MODE::TRANS_BMM_TRANS || MODE == TBMM_MODE::TRANS_BMM_TRANS_TRANS) {
        uint64_t mergeBatchK =
            block_.params_.cBatchDimAll * block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb;
        mm_.SetOrgShape(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.M,
                        block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N, mergeBatchK,
                        block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb,
                        block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N * block_.params_.cBatchDimAll /
                            block_.batchSplitFactor_);
    } else if constexpr (MODE == TBMM_MODE::BMM_TRANS || MODE == TBMM_MODE::BMM_TRANS_TRANS) {
        mm_.SetOrgShape(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.M,
                        block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N,
                        block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb,
                        block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb,
                        block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N * block_.params_.cBatchDimAll /
                            block_.batchSplitFactor_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, TBMM_MODE MODE, class BLOCK_TYPE,
          const MatmulConfig& MM_CFG>
__aicore__ inline void
TransposeBatchMatMulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR scalesGM, GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM, scalesGM);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, TBMM_MODE MODE, class BLOCK_TYPE,
          const MatmulConfig& MM_CFG>
__aicore__ inline void
TransposeBatchMatMulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG>::Process(uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }

    SetAtomicNone();
    mm_.SetHF32(block_.batchMatmulTilingData_->matMulTilingData.isHf32, 1);
    for (uint64_t j = 0; j < block_.params_.round; j++) {
        uint64_t newBlockIdx = GetBlockIdx();
        block_.UpdateBasicIndex(j, newBlockIdx); // 使能错位分核更新Index
        if (block_.params_.index < block_.params_.totalCnt) {
            block_.UpdateBlockParams(j);
            if (block_.params_.singleCoreM > 0 && block_.params_.singleCoreN > 0) {
                block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE>();

                mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                    block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.singleCoreK);
                mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
                mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
                if (block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.isBias) {
                    mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
                }
                mm_.Iterate();
                mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
            }
        }
    }
    mm_.SetHF32(false, 0);
    PipeBarrier<PIPE_ALL>();
}

} // namespace TransposeBatchMatMulAdvanced

#endif // TRANSPOSE_BATCH_MAT_MUL_ASW_KERNEL_ADVANCED_H