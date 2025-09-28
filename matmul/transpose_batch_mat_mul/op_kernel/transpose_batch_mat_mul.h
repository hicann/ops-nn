/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_batch_mat_mul.h
 * \brief
 */
#ifndef TRNASPOSE_BATCH_MAT_MUL_H
#define TRNASPOSE_BATCH_MAT_MUL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "transpose_batch_mat_mul_block.h"

using namespace AscendC;
using namespace matmul;
using namespace TransposeBatchMatmulSpace;
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE, class BLOCK_TYPE = TransposeBatchMatMulBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
class TransposeBatchMatMulKernel {
   public:
    __aicore__ inline TransposeBatchMatMulKernel() {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                GM_ADDR scalesGM, GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                              GM_ADDR scalesGM, GM_ADDR workspaceGM);

    __aicore__ inline void Process();

    __aicore__ inline void InnerProcess(uint64_t mTileIndex, uint64_t nTileIndex);

    __aicore__ inline void End() { mm_.End(); }

   protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB> mm_;
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
    TBuf<> ubBuf_;

   private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR scalesGM);
    __aicore__ inline void SetOrgShape();
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void TransposeBatchMatMulKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR scalesGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    // cube core cases, ignore vector core
    if ASCEND_IS_AIV {
        return;
    }
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);

    if (GetBlockIdx() >= block_.tbmmTilingData_->multiBatchInfo.batchUsedCoreNum) {
        return;
    }
    SetAtomicNone();
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM, scalesGM);

    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.tbmmTilingData_->matmulTiling.matmulTiling, pipe_);
    mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void TransposeBatchMatMulKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG,
                                                  MM_CB>::InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                                     GM_ADDR biasGM, GM_ADDR scalesGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                             block_.params_.aBatchDimAll *
                                 static_cast<uint64_t>(block_.tbmmTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.tbmmTilingData_->matmulTiling.matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             block_.params_.bBatchDimAll *
                                 static_cast<uint64_t>(block_.tbmmTilingData_->matmulTiling.matmulTiling.Kb) *
                                 block_.tbmmTilingData_->matmulTiling.matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             block_.params_.cBatchDimAll *
                                 static_cast<uint64_t>(block_.tbmmTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.tbmmTilingData_->matmulTiling.matmulTiling.N);

    if (block_.tbmmTilingData_->matmulTiling.matmulTiling.isBias) {
        auto biasSize = block_.params_.biasWithBatch
                            ? block_.params_.cBatchDimAll *
                                  static_cast<uint64_t>(block_.tbmmTilingData_->matmulTiling.matmulTiling.N)
                            : block_.tbmmTilingData_->matmulTiling.matmulTiling.N;
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), biasSize);
    }
    if constexpr (sizeof(C_T) == 1) {
        auto scalesSize = block_.params_.cBatchDimAll *
              static_cast<uint64_t>(block_.tbmmTilingData_->matmulTiling.matmulTiling.N);
        scalesGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(scalesGM), scalesSize);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
TransposeBatchMatMulKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG, MM_CB>::SetOrgShape()
{
    if constexpr (MODE == static_cast<int>(TBMM_MODE::TRANS_BMM_TRANS) || MODE == static_cast<int>(TBMM_MODE::TRANS_BMM_TRANS_TRANS)) {
        uint64_t mergeBatchK = block_.params_.batchCnt * block_.tbmmTilingData_->matmulTiling.matmulTiling.Kb;
        mm_.SetOrgShape(block_.tbmmTilingData_->matmulTiling.matmulTiling.M,
                        block_.tbmmTilingData_->matmulTiling.matmulTiling.N, mergeBatchK, mergeBatchK,
                        block_.tbmmTilingData_->matmulTiling.matmulTiling.N * block_.params_.batchCnt /
                        block_.batchSplitFactor_);
    }
    if constexpr (MODE == static_cast<int>(TBMM_MODE::BMM_TRANS) || MODE == static_cast<int>(TBMM_MODE::BMM_TRANS_TRANS)) {
        mm_.SetOrgShape(block_.params_.batchCnt * block_.tbmmTilingData_->matmulTiling.matmulTiling.M,
                        block_.tbmmTilingData_->matmulTiling.matmulTiling.N,
                        block_.tbmmTilingData_->matmulTiling.matmulTiling.Kb,
                        block_.tbmmTilingData_->matmulTiling.matmulTiling.Kb,
                        block_.tbmmTilingData_->matmulTiling.matmulTiling.N * block_.params_.batchCnt /
                        block_.batchSplitFactor_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void TransposeBatchMatMulKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG,
                                                  MM_CB>::UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                                             GM_ADDR biasGM, GM_ADDR scalesGM,
                                                                             GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM, scalesGM);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void TransposeBatchMatMulKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG,
                                                  MM_CB>::InnerProcess(uint64_t mTileIndex, uint64_t nTileIndex)
{
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        if (block_.params_.rowOrder == 0) {
            block_.UpdateBasicIndex(j, mTileIndex, nTileIndex);  // 使能错位分核更新Index
        }
        if (block_.params_.index < block_.params_.totalTileCnt) {
            block_.UpdateBlockParams(mTileIndex, nTileIndex);
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE>(mTileIndex, nTileIndex);
            mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                               block_.tbmmTilingData_->matmulTiling.matmulTiling.Ka);
            if constexpr(sizeof(typename C_TYPE::T)==1) {
                mm_.SetQuantVector(scalesGlobal_[block_.offsetScales_]);
            }
            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
            if (block_.tbmmTilingData_->matmulTiling.matmulTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
            mm_.IterateAll(cGlobal_[block_.offset_.offsetC]);
        }
        // 行/列优先走这
        block_.UpdateBlockIndex();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
TransposeBatchMatMulKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MODE, BLOCK_TYPE, MM_CFG, MM_CB>::Process()
{
    // cube core cases, ignore vector core
    if ASCEND_IS_AIV {
        return;
    }
    if (GetBlockIdx() >= block_.tbmmTilingData_->multiBatchInfo.batchUsedCoreNum) {
        return;
    }

    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }
    bool reverse = true;
    for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            block_.InitBlockIndex();
            InnerProcess(mTileIndex, nTileIndex);
        }
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm_.SetHF32(false, 0);
    return;
}
#endif // TRNASPOSE_BATCH_MAT_MUL_H