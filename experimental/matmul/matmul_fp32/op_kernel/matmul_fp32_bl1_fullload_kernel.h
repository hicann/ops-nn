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
 * \file matmul_fp32_bl1_fullload_kernel.h
 * \brief
 */
#ifndef __MATMUL_FP32_BL1_FULLLOAD_KERNEL_H__
#define __MATMUL_FP32_BL1_FULLLOAD_KERNEL_H__

#include "matmul_fp32_base_block.h"

using namespace AscendC;
using namespace matmul;

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulFp32BaseBlock,
    const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD>
class MatmulFp32BL1FullLoadKernel {
public:
    __aicore__ inline MatmulFp32BL1FullLoadKernel(){};

    __aicore__ inline void Init(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM, const void* tilingData,
        TPipe* pipe);
    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using B_TYPE_NEW = MatmulType<AscendC::TPosition::TSCM, B_TYPE::format, B_T, B_TYPE::isTrans>;
    MatmulImpl<A_TYPE, B_TYPE_NEW, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    TQue<QuePosition::B1, 1> InQueueBL1_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe* pipe_;

private:
    __aicore__ inline void CopyInB1(const MatmulFp32TilingData &matmulFp32TilingData, LocalTensor<B_T> bl1Local);
};

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG>
__aicore__ inline void MatmulFp32BL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM, const void* tilingData, TPipe* pipe)
{
    block_.template Init<A_TYPE, B_TYPE_NEW, C_TYPE, BIAS_TYPE>(tilingData);
    pipe_ = pipe;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
        static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.M) *
        block_.matmulFp32TilingData_->tCubeTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
        static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.Kb) *
        block_.matmulFp32TilingData_->tCubeTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
        static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.M) *
        block_.matmulFp32TilingData_->tCubeTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM),
                                block_.matmulFp32TilingData_->tCubeTiling.N);
    mm_.SetSubBlockIdx(0);
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG>
__aicore__ inline void MatmulFp32BL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(
    uint64_t index, uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint64_t innerAlignedBlock = BLOCK_BYTE_SIZE / sizeof(B_T);
    uint64_t nAligned = B_TYPE_NEW::isTrans ?
        MMFp32CeilAlign(static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.singleCoreN), BLOCK_SIZE) :
        MMFp32CeilAlign(static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.singleCoreN), innerAlignedBlock);
    uint64_t kbAligned = B_TYPE_NEW::isTrans ?
        MMFp32CeilAlign(static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.Kb), innerAlignedBlock):
        MMFp32CeilAlign(static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.Kb), BLOCK_SIZE);
    pipe_->InitBuffer(InQueueBL1_, 1, nAligned * kbAligned * sizeof(B_T));
    LocalTensor<B_T> bl1Local = InQueueBL1_.AllocTensor<B_T>();
    CopyInB1(*(block_.matmulFp32TilingData_), bl1Local);
    // BL1搬运后matmul初始化
    mm_.Init(&block_.matmulFp32TilingData_->tCubeTiling, pipe_);
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
            mm_.SetTensorB(bl1Local[block_.offset_.offsetB], block_.params_.isTransposeB);
            if (block_.matmulFp32TilingData_->tCubeTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
            mm_.IterateAll(cGlobal_[block_.offset_.offsetC]);
        }
    }
    InQueueBL1_.FreeTensor(bl1Local);
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    return;
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatmulFp32BL1FullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::
    CopyInB1(const MatmulFp32TilingData &matmulFp32TilingData, LocalTensor<B_T> bl1Local)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = B_TYPE::isTrans ? static_cast<uint64_t>(matmulFp32TilingData.tCubeTiling.N) :
                                           static_cast<uint64_t>(matmulFp32TilingData.tCubeTiling.Kb);
    nd2nzParams.dValue = B_TYPE::isTrans ? static_cast<uint64_t>(matmulFp32TilingData.tCubeTiling.Kb) :
                                           static_cast<uint64_t>(matmulFp32TilingData.tCubeTiling.N);
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue =
        static_cast<uint64_t>(B_TYPE::isTrans ? matmulFp32TilingData.tCubeTiling.Kb : matmulFp32TilingData.tCubeTiling.N);
    nd2nzParams.dstNzC0Stride = MMFp32CeilAlign(nd2nzParams.nValue, BLOCK_SIZE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(bl1Local, bGlobal_, nd2nzParams);
    InQueueBL1_.EnQue(bl1Local);
    bl1Local = InQueueBL1_.DeQue<B_T>();
}

#endif // __MATMUL_FP32_BL1_FULLLOAD_KERNEL_H__