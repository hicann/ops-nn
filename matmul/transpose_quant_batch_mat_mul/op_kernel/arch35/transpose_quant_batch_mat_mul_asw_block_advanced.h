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
 * \file transpose_quant_batch_mat_mul_asw_block_advanced.h
 * \brief
 */
#ifndef TRANSPOSE_QUANT_BATCH_MAT_MUL_ASW_BLOCK_ADVANCED_H
#define TRANSPOSE_QUANT_BATCH_MAT_MUL_ASW_BLOCK_ADVANCED_H

#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "../../common/cmct/utils/common_utils.h"

using namespace Cmct::Gemm;
using namespace AscendC;
namespace TransposeQuantBatchMatMulAdvanced {
struct ASWOffsetParam {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetBias;
    uint64_t offsetScale;
    uint64_t offsetPerTokenScale;
    uint64_t batchOffset;
};
struct ASWTilingParam {
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalCnt;
    uint64_t mCoreNum;
    uint64_t mTailCoreNum;
    uint64_t mBaseTail;
    uint64_t nBaseTail;
    uint64_t mIndex;
    uint64_t nIndex;
    uint64_t mainRow;
    uint64_t round;
    uint64_t index;
};

class TransposeQuantBatchMatMulAswBlock {
public:
    __aicore__ inline TransposeQuantBatchMatMulAswBlock() {}
    __aicore__ inline void Init(const BatchMatMulV3TilingData* tilingData, uint32_t blockIdx);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx);
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    __aicore__ inline void CalcGMOffset();

public:
    ASWTilingParam params_;
    ASWOffsetParam offset_;
    const BatchMatMulV3TilingData* tilingData_;

private:
    const uint64_t WINDOW_LEN = 4;
    static constexpr uint64_t CUBE_BLOCK = 16UL;
    uint32_t blockIdx_;
};

__aicore__ inline void TransposeQuantBatchMatMulAswBlock::Init(const BatchMatMulV3TilingData* tilingData,
                                                               uint32_t blockIdx)
{
    blockIdx_ = blockIdx;
    tilingData_ = tilingData;
    params_.mCnt = CeilDiv(static_cast<uint64_t>(tilingData_->matMulTilingData.tCubeTiling.M),
                           static_cast<uint64_t>(tilingData_->matMulTilingData.tCubeTiling.baseM));
    params_.nCnt = CeilDiv(static_cast<uint64_t>(tilingData_->matMulTilingData.tCubeTiling.N),
                           static_cast<uint64_t>(tilingData_->matMulTilingData.tCubeTiling.baseN));
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.round = CeilDiv(static_cast<uint64_t>(params_.totalCnt),
                            static_cast<uint64_t>(tilingData_->matMulTilingData.tCubeTiling.usedCoreNum));
    params_.mCoreNum = AscendC::Std::min(WINDOW_LEN, params_.mCnt);
    params_.mainRow = params_.mCnt / params_.mCoreNum - 1UL;
    params_.mTailCoreNum = params_.mCnt - params_.mCoreNum * params_.mainRow;
    params_.mBaseTail = tilingData_->matMulTilingData.tCubeTiling.M -
                        (params_.mCnt - 1) * tilingData_->matMulTilingData.tCubeTiling.baseM;
    params_.nBaseTail = tilingData_->matMulTilingData.tCubeTiling.N -
                        (params_.nCnt - 1) * tilingData_->matMulTilingData.tCubeTiling.baseN;
    offset_.offsetBias = 0UL;
}

__aicore__ inline void TransposeQuantBatchMatMulAswBlock::UpdateBasicIndex(uint64_t roundIdx)
{
    params_.index = blockIdx_ + roundIdx * tilingData_->matMulTilingData.tCubeTiling.usedCoreNum;
    uint64_t rowIdx = params_.index / params_.nCnt / params_.mCoreNum;
    if (rowIdx < params_.mainRow) {
        params_.mIndex = rowIdx * params_.mCoreNum + params_.index % params_.mCoreNum;
        params_.nIndex = (params_.index / params_.mCoreNum) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - params_.mainRow * params_.mCoreNum * params_.nCnt;
        params_.mIndex = params_.mainRow * params_.mCoreNum + tailIndex % params_.mTailCoreNum;
        params_.nIndex = (tailIndex / params_.mTailCoreNum) % params_.nCnt;
    }

    if (rowIdx & 1) {
        params_.nIndex = params_.nCnt - 1 - params_.nIndex;
    }
}

__aicore__ inline void TransposeQuantBatchMatMulAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM =
        params_.mIndex != (params_.mCnt - 1UL) ? tilingData_->matMulTilingData.tCubeTiling.baseM : params_.mBaseTail;
    params_.singleCoreN =
        params_.nIndex != (params_.nCnt - 1UL) ? tilingData_->matMulTilingData.tCubeTiling.baseN : params_.nBaseTail;
}

__aicore__ inline void TransposeQuantBatchMatMulAswBlock::CalcGMOffset()
{
    uint64_t mOffset = params_.mIndex * tilingData_->matMulTilingData.tCubeTiling.baseM;
    uint64_t nOffset = params_.nIndex * tilingData_->matMulTilingData.tCubeTiling.baseN;
    offset_.offsetA = mOffset * tilingData_->cBatchDimAll * tilingData_->matMulTilingData.tCubeTiling.Ka +
                      offset_.batchOffset * tilingData_->matMulTilingData.tCubeTiling.Ka;
    offset_.offsetB = offset_.batchOffset * tilingData_->matMulTilingData.tCubeTiling.N *
                          tilingData_->matMulTilingData.tCubeTiling.Kb +
                      nOffset;
    offset_.offsetC = mOffset * tilingData_->cBatchDimAll * tilingData_->matMulTilingData.tCubeTiling.N +
                      offset_.batchOffset * tilingData_->matMulTilingData.tCubeTiling.N + nOffset;
    offset_.offsetPerTokenScale = mOffset;
    offset_.offsetScale = nOffset;
}

} // namespace TransposeQuantBatchMatMulAdvanced

#endif // TRANSPOSE_QUANT_BATCH_MAT_MUL_ASW_BLOCK_ADVANCED_H