/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_batch_mat_mul_asw_block_advanced.h
 * \brief
 */
#ifndef TRANSPOSE_BATCH_MAT_MUL_ASW_BLOCK_ADVANCED_H
#define TRANSPOSE_BATCH_MAT_MUL_ASW_BLOCK_ADVANCED_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../../mat_mul_v3/mat_mul_v3_common.h"
#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"

namespace TransposeBatchMatMulAdvanced {

using namespace AscendC;
using namespace matmul;

enum class TBMM_MODE : int32_t { BMM_TRANS = 0, TRANS_BMM_TRANS = 1, BMM_TRANS_TRANS = 2, TRANS_BMM_TRANS_TRANS = 3 };

struct TBmmAswBlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;
    uint64_t offsetBias = 0;
};

struct TBmmAswBlockArgs {
    uint64_t aBatchDimAll = 1;
    uint64_t bBatchDimAll = 1;
    uint64_t cBatchDimAll = 1;
    uint64_t index = 0;
    uint64_t mCntIndex = 0;
    uint64_t nCntIndex = 0;
    uint64_t mCnt = 0;
    uint64_t nCnt = 0;
    uint64_t totalCnt = 0;
    uint64_t blockBaseM = 0;
    uint64_t blockBaseN = 0;
    uint64_t nBaseTail = 0;
    uint64_t mBaseTail = 0;
    uint64_t mBaseSplitCnt = 1;
    uint64_t nBaseSplitCnt = 1;
    uint64_t totalSplitCnt = 1;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreN = 0;
    uint64_t round = 0;
    uint64_t mainRow = 0;
    uint64_t mainWindow = 0;
    uint64_t tailWindow = 0;
};

class TransposeBatchMatMulAswBlock {
public:
    __aicore__ inline TransposeBatchMatMulAswBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void* tilingData);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx);
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, TBMM_MODE MODE>
    __aicore__ inline void CalcGMOffset();

public:
    TBmmAswBlockOffset offset_;
    TBmmAswBlockArgs params_;
    const BatchMatMulV3TilingData* batchMatmulTilingData_;
    uint64_t offsetScales_ = 0;
    int32_t batchSplitFactor_ = 1;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void TransposeBatchMatMulAswBlock::Init(const void* tilingData)
{
    batchMatmulTilingData_ = static_cast<const BatchMatMulV3TilingData*>(tilingData);
    params_.aBatchDimAll = batchMatmulTilingData_->aBatchDimAll;
    params_.bBatchDimAll = batchMatmulTilingData_->bBatchDimAll;
    params_.cBatchDimAll = batchMatmulTilingData_->cBatchDimAll;

    params_.blockBaseM = static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.baseN);
    params_.mCnt = (batchMatmulTilingData_->matMulTilingData.tCubeTiling.M + params_.blockBaseM - 1) /
                   params_.blockBaseM; // 总的m方向base块个数
    params_.nCnt = (batchMatmulTilingData_->matMulTilingData.tCubeTiling.N + params_.blockBaseN - 1) /
                   params_.blockBaseN; // 总的n方向base块个数
    params_.nBaseTail = batchMatmulTilingData_->matMulTilingData.tCubeTiling.N -
                        (params_.nCnt - 1) * params_.blockBaseN; // n方向上的base尾块
    params_.mBaseTail = batchMatmulTilingData_->matMulTilingData.tCubeTiling.M -
                        (params_.mCnt - 1) * params_.blockBaseM; // m方向上的base尾块
    params_.totalCnt = params_.cBatchDimAll * params_.mCnt * params_.nCnt;
    params_.round = (params_.totalCnt + batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum - 1) /
                    batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum;
    params_.mainWindow = AscendC::Std::min(static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.aswWindowLen),
                                           params_.mCnt);                     // 主划窗m方向的块个数
    params_.mainRow = params_.mCnt / params_.mainWindow - 1;                  // 主划窗数量
    params_.tailWindow = params_.mCnt - params_.mainRow * params_.mainWindow; // 尾划窗m方向的块个数
    batchSplitFactor_ = batchMatmulTilingData_->batchSplitFactor <= 0 ? 1 : batchMatmulTilingData_->batchSplitFactor;
}

__aicore__ inline void TransposeBatchMatMulAswBlock::UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx)
{
    params_.index = newBlockIdx + roundIdx * batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum;
    uint64_t matIndex = params_.index % (params_.mCnt * params_.nCnt);
    uint64_t rowIdx = matIndex / params_.nCnt / params_.mainWindow;
    if (rowIdx < params_.mainRow) {
        params_.mCntIndex = rowIdx * params_.mainWindow + matIndex % params_.mainWindow;
        params_.nCntIndex = (matIndex / params_.mainWindow) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = matIndex - params_.mainRow * params_.mainWindow * params_.nCnt;
        params_.mCntIndex = params_.mainRow * params_.mainWindow + tailIndex % params_.tailWindow;
        params_.nCntIndex = (tailIndex / params_.tailWindow) % params_.nCnt;
    }
    // mod 2 means even row, need reverse scan
    if (rowIdx % 2 != 0) {
        params_.nCntIndex = params_.nCnt - 1 - params_.nCntIndex;
    }
}

__aicore__ inline void TransposeBatchMatMulAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM = params_.mCntIndex != (params_.mCnt - 1) ? params_.blockBaseM : params_.mBaseTail;
    params_.singleCoreN = params_.nCntIndex != (params_.nCnt - 1) ? params_.blockBaseN : params_.nBaseTail;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, TBMM_MODE MODE>
__aicore__ inline void TransposeBatchMatMulAswBlock::CalcGMOffset()
{
    // batch, M 合轴 处理 index
    uint64_t totalBmTileIndex = params_.index / (params_.nCnt * params_.mCnt);
    if constexpr (MODE == TBMM_MODE::TRANS_BMM_TRANS || MODE == TBMM_MODE::BMM_TRANS) {
        offset_.offsetC = params_.mCntIndex * params_.blockBaseM * params_.cBatchDimAll *
                              batchMatmulTilingData_->matMulTilingData.tCubeTiling.N +
                          totalBmTileIndex * batchMatmulTilingData_->matMulTilingData.tCubeTiling.N +
                          params_.nCntIndex * params_.blockBaseN;
    }
    if constexpr (MODE == TBMM_MODE::TRANS_BMM_TRANS_TRANS || MODE == TBMM_MODE::BMM_TRANS_TRANS) {
        uint64_t innerBatch = params_.cBatchDimAll / batchSplitFactor_;
        uint64_t batchIdx1 = totalBmTileIndex / innerBatch;
        uint64_t batchIdx2 = totalBmTileIndex % innerBatch;
        offset_.offsetC = (batchIdx1 * batchMatmulTilingData_->matMulTilingData.tCubeTiling.M +
                           params_.mCntIndex * params_.blockBaseM) *
                              innerBatch * batchMatmulTilingData_->matMulTilingData.tCubeTiling.N +
                          batchIdx2 * batchMatmulTilingData_->matMulTilingData.tCubeTiling.N +
                          params_.nCntIndex * params_.blockBaseN;
    }
    if constexpr (MODE == TBMM_MODE::TRANS_BMM_TRANS || MODE == TBMM_MODE::TRANS_BMM_TRANS_TRANS) {
        uint64_t offsetAM = params_.mCntIndex * params_.blockBaseM;
        offset_.offsetA = offsetAM * params_.cBatchDimAll * batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka +
                          totalBmTileIndex * batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka;
    }
    if constexpr (MODE == TBMM_MODE::BMM_TRANS || MODE == TBMM_MODE::BMM_TRANS_TRANS) {
        offset_.offsetA =
            totalBmTileIndex * batchMatmulTilingData_->matMulTilingData.tCubeTiling.M *
                batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka +
            params_.mCntIndex * params_.blockBaseM * batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka;
    }
    uint64_t offsetBBatch = totalBmTileIndex *
                            static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.N) *
                            batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb;

    offset_.offsetB =
        B_TYPE::isTrans ?
            offsetBBatch +
                (params_.nCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb) *
                 params_.blockBaseN) :
            offsetBBatch + (params_.nCntIndex * params_.blockBaseN);

    if (batchMatmulTilingData_->matMulTilingData.tCubeTiling.isBias) {
        offset_.offsetBias = params_.nCntIndex * params_.blockBaseN;
    }
}

} // namespace TransposeBatchMatMulAdvanced

#endif // TRANSPOSE_BATCH_MAT_MUL_ASW_BLOCK_ADVANCED_H