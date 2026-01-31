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
 * \file matmul_fp32_base_block.h
 * \brief
 */
#ifndef __OP_MATMUL_FP32_BASE_BLOCK_H__
#define __OP_MATMUL_FP32_BASE_BLOCK_H__

#include "matmul_fp32_common.h"
#include "matmul_fp32_tiling_data.h"

using namespace AscendC;
using namespace matmul;

struct BlockOffset {
    uint64_t offsetA = 0UL;
    uint64_t offsetB = 0UL;
    uint64_t offsetC = 0UL;
    uint64_t offsetBias = 0UL;
};

struct BaseBlockArgs {
    uint64_t index = 0UL;
    uint64_t mCntIndex = 0UL;
    uint64_t nCntIndex = 0UL;
    uint64_t mCnt = 0UL;
    uint64_t nCnt = 0UL;
    uint64_t totalCnt = 0UL;
    uint64_t singleCoreM = 0UL;
    uint64_t singleCoreN = 0UL;
    uint64_t mBaseTail = 0UL;
    uint64_t nBaseTail = 0UL;
    uint64_t round = 0UL;
    bool isTransposeA = false;
    bool isTransposeB = false;
};

class MatmulFp32BaseBlock {
public:
    __aicore__ inline MatmulFp32BaseBlock()
    {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void* tilingData);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, uint64_t blockIdx);
    __aicore__ inline void UpdateBlockParams();
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset();

public:
    BlockOffset offset_;
    BaseBlockArgs params_;
    const MatmulFp32TilingData* matmulFp32TilingData_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulFp32BaseBlock::Init(const void* tilingData)
{
    matmulFp32TilingData_ = static_cast<const MatmulFp32TilingData*>(tilingData);
    params_.isTransposeA = matmulFp32TilingData_->matmulFp32RunInfo.transA;
    params_.isTransposeB = matmulFp32TilingData_->matmulFp32RunInfo.transB;
    // 总的m方向base块个数
    params_.mCnt = MMFp32CeilDiv(matmulFp32TilingData_->tCubeTiling.M, matmulFp32TilingData_->tCubeTiling.singleCoreM);
    // 总的n方向base块个数
    params_.nCnt = MMFp32CeilDiv(matmulFp32TilingData_->tCubeTiling.N, matmulFp32TilingData_->tCubeTiling.singleCoreN);
    // m方向的base尾块
    params_.mBaseTail = static_cast<uint64_t>(matmulFp32TilingData_->tCubeTiling.M) -
                        (params_.mCnt - 1) * matmulFp32TilingData_->tCubeTiling.singleCoreM;
    // n方向的base尾块
    params_.nBaseTail = static_cast<uint64_t>(matmulFp32TilingData_->tCubeTiling.N) -
                        (params_.nCnt - 1) * matmulFp32TilingData_->tCubeTiling.singleCoreN;
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.round = MMFp32CeilDiv(params_.totalCnt, matmulFp32TilingData_->tCubeTiling.usedCoreNum);
}

__aicore__ inline void MatmulFp32BaseBlock::UpdateBasicIndex(uint64_t roundIdx, uint64_t blockIdx)
{
    params_.index = blockIdx + roundIdx * matmulFp32TilingData_->tCubeTiling.usedCoreNum;
    params_.mCntIndex = params_.index / params_.nCnt;
    params_.nCntIndex = params_.index % params_.nCnt;
}

__aicore__ inline void MatmulFp32BaseBlock::UpdateBlockParams()
{
    params_.singleCoreM =
        (params_.mCntIndex == (params_.mCnt - 1)) ? params_.mBaseTail : matmulFp32TilingData_->tCubeTiling.singleCoreM;
    params_.singleCoreN =
        (params_.nCntIndex == (params_.nCnt - 1)) ? params_.nBaseTail : matmulFp32TilingData_->tCubeTiling.singleCoreN;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulFp32BaseBlock::CalcGMOffset()
{
    if (params_.isTransposeA) {
        offset_.offsetA = params_.mCntIndex * matmulFp32TilingData_->tCubeTiling.singleCoreM;
    } else {
        offset_.offsetA =
            params_.mCntIndex * matmulFp32TilingData_->tCubeTiling.singleCoreM * matmulFp32TilingData_->tCubeTiling.Ka;
    }
    if (params_.isTransposeB) {
        offset_.offsetB =
            params_.nCntIndex * matmulFp32TilingData_->tCubeTiling.singleCoreN * matmulFp32TilingData_->tCubeTiling.Kb;
    } else {
        offset_.offsetB = params_.nCntIndex * matmulFp32TilingData_->tCubeTiling.singleCoreN;
    }
    offset_.offsetC = params_.nCntIndex * matmulFp32TilingData_->tCubeTiling.singleCoreN +
                      params_.mCntIndex * matmulFp32TilingData_->tCubeTiling.singleCoreM *
                          matmulFp32TilingData_->tCubeTiling.singleCoreN;
    if (matmulFp32TilingData_->tCubeTiling.isBias) {
        offset_.offsetBias = params_.nCntIndex * matmulFp32TilingData_->tCubeTiling.singleCoreN;
    }
    uint64_t blockIdx = GetCurrentBlockIdx();
}

#endif // __OP_MATMUL_FP32_BASE_BLOCK_H__