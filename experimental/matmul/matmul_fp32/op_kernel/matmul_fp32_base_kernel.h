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
 * \file matmul_fp32_base_kernel.h
 * \brief
 */
#ifndef __MATMUL_FP32_BASE_KERNEL_H__
#define __MATMUL_FP32_BASE_KERNEL_H__

#include "matmul_fp32_base_block.h"

using namespace AscendC;
using namespace matmul;

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulFp32BaseBlock,
    const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
class MatmulFp32BaseKernel {
public:
    __aicore__ inline MatmulFp32BaseKernel(){};

    __aicore__ inline void Init(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM, const void* tilingData,
        TPipe* pipe);

    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

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
    TPipe* pipe_;
};

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
    class MM_CB>
__aicore__ inline void MatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM, const void* tilingData, TPipe* pipe)
{
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.matmulFp32TilingData_->tCubeTiling, pipe_);
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
    class MM_CB>
__aicore__ inline void MatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(
        reinterpret_cast<__gm__ A_T*>(aGM), static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.M) *
                                                block_.matmulFp32TilingData_->tCubeTiling.Ka);
    bGlobal_.SetGlobalBuffer(
        reinterpret_cast<__gm__ B_T*>(bGM), static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.Kb) *
                                                block_.matmulFp32TilingData_->tCubeTiling.N);
    cGlobal_.SetGlobalBuffer(
        reinterpret_cast<__gm__ C_T*>(cGM), static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.M) *
                                                block_.matmulFp32TilingData_->tCubeTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGM), block_.matmulFp32TilingData_->tCubeTiling.N);
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
    class MM_CB>
__aicore__ inline void MatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Process(
    uint64_t index, uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    for (uint64_t j = 0; j < block_.params_.round; j++) {
        uint64_t blockIdx = GetCurrentBlockIdx();
        block_.UpdateBasicIndex(j, blockIdx); // update basic index
        if (block_.params_.index < block_.params_.totalCnt) {
            block_.UpdateBlockParams();
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            mm_.SetSingleShape(
                block_.params_.singleCoreM, block_.params_.singleCoreN,
                block_.matmulFp32TilingData_->tCubeTiling.singleCoreK);
            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], block_.params_.isTransposeA);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], block_.params_.isTransposeB);
            if (block_.matmulFp32TilingData_->tCubeTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
            mm_.Iterate();
            mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
        }
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    return;
}

#endif // __MATMUL_FP32_BASE_KERNEL_H__