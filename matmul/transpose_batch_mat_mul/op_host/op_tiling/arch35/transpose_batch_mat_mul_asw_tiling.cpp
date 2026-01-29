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
 * \file transpose_batch_mat_mul_asw_tiling.cc
 * \brief
 */

#include "transpose_batch_mat_mul_asw_tiling.h"
#include "transpose_batch_mat_mul_tiling_strategy.h"
#include "transpose_batch_mat_mul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace transpose_batch_mat_mul_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(TransposeBatchMatMul, TransposeBatchMatMulAswTiling, ASCEND950, BASE);

template <typename T>
T GetAlignNumWithDataType(T size, ge::DataType dtype) {
    return size / static_cast<T>(ge::GetSizeByDataType(dtype));
}

void TransposeBatchMatMulAswTiling::ResetBasicBlock(uint64_t tempBaseM, uint64_t tempBaseN)
{
    uint64_t baseKAlignNum = (!args_.isATrans && args_.isBTrans) ?
                                GetAlignNumWithDataType(BASIC_BLOCK_K_256_BYTE, args_.aType) :
                                GetAlignNumWithDataType(BLOCK_BYTE_SIZE, args_.aType);
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args_.kValue), baseKAlignNum);
    uint64_t maxBaseK =
        GetAlignNumWithDataType(compileInfo_.l0ASize / DB_SIZE, args_.aType) / std::max(tempBaseM, tempBaseN);
    if (maxBaseK >= baseKAlignNum) {
        runInfo_.baseM = tempBaseM;
        runInfo_.baseN = tempBaseN;
        maxBaseK = ops::FloorAlign(maxBaseK, baseKAlignNum);
        runInfo_.baseK = std::min(kValueAlign, maxBaseK);
    }
}

void TransposeBatchMatMulAswTiling::BaseLoadBalance()
{
    uint64_t baseMAlignNum = args_.isATrans ? GetAlignNumWithDataType(BASIC_BLOCK_K_256_BYTE, args_.aType) :
                                BASIC_BLOCK_SIZE_16;
    uint64_t baseNAlignNum = !args_.isBTrans ? GetAlignNumWithDataType(BASIC_BLOCK_K_256_BYTE, args_.aType) :
                                BASIC_BLOCK_SIZE_16;
    uint64_t mMaxTile = ops::CeilDiv(args_.mValue, baseMAlignNum);
    uint64_t nMaxTile = ops::CeilDiv(args_.nValue, baseNAlignNum);
    uint64_t tempBaseM = runInfo_.baseM;
    uint64_t tempBaseN = runInfo_.baseN;
    uint64_t coreNumMN = compileInfo_.aicNum / batchInfo_->batchC;
    if (mMaxTile * nMaxTile >= coreNumMN || (!args_.isATrans && args_.isBTrans)) {
        uint64_t mCore = ops::CeilDiv(args_.mValue, runInfo_.baseM);
        uint64_t nCore = ops::CeilDiv(args_.nValue, runInfo_.baseN);
        if (mMaxTile < nMaxTile || (mMaxTile == nMaxTile && baseNAlignNum == BASIC_BLOCK_SIZE_16)) {
            tempBaseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), baseMAlignNum);
            mCore = ops::CeilDiv(args_.mValue, tempBaseM);
            nCore = coreNumMN / mCore;
            tempBaseN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCore), baseNAlignNum);
        } else {
            tempBaseN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCore), baseNAlignNum);
            nCore = ops::CeilDiv(args_.nValue, tempBaseN);
            mCore = coreNumMN / nCore;
            tempBaseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), baseMAlignNum);
        }

        while (tempBaseN >= tempBaseM * NUM_TWO && nCore < coreNumMN / NUM_TWO &&
            tempBaseN != baseNAlignNum) {
            nCore *= NUM_TWO;
            mCore = coreNumMN / nCore;
            tempBaseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCore), baseNAlignNum);
            mCore = ops::CeilDiv(args_.mValue, static_cast<uint64_t>(tempBaseM));
            nCore = ops::CeilDiv(args_.nValue, static_cast<uint64_t>(tempBaseN));
        }

        while (tempBaseM >= tempBaseN * NUM_TWO && mCore < coreNumMN / NUM_TWO &&
            tempBaseM != baseMAlignNum) {
            mCore *= NUM_TWO;
            nCore = coreNumMN / mCore;
            tempBaseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCore), baseNAlignNum);
            mCore = ops::CeilDiv(args_.mValue, static_cast<uint64_t>(tempBaseM));
            nCore = ops::CeilDiv(args_.nValue, static_cast<uint64_t>(tempBaseN));
        }
        ResetBasicBlock(tempBaseM, tempBaseN);
    }
}

ge::graphStatus TransposeBatchMatMulAswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    // load balance only support n=1024 k=4096
    bool isNeedLoadBalance = args_.mValue <= BASIC_BLOCK_SIZE_16 && args_.nValue == 1024 && args_.kValue == 4096;
    if (runInfo_.tailInfo.mCnt * runInfo_.tailInfo.nCnt * batchInfo_->batchC < compileInfo_.aicNum &&
        isNeedLoadBalance) {
        BaseLoadBalance();
    }
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    GetTransposeBatchMatMulInfo();
    return ge::GRAPH_SUCCESS;
}

void TransposeBatchMatMulAswTiling::GetTransposeBatchMatMulInfo()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    if (aPermList != nullptr && aPermList->GetSize() == ALLOW_DIM) {
        const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
        if ((aPerm[BATCH_IDX] == 1L) && (aPerm[M_IDX] == 0L) && (aPerm[KA_IDX] == 2L)) {
            permX1_ = TBMMPermX1::PERM_X1_1_0_2;
        } else if ((aPerm[BATCH_IDX] == 0L) && (aPerm[M_IDX] == 1L) && (aPerm[KA_IDX] == 2L)) {
            permX1_ = TBMMPermX1::PERM_X1_0_1_2;
        }
    }
    if (bPermList != nullptr && bPermList->GetSize() == ALLOW_DIM) {
        const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
        if ((bPerm[BATCH_IDX] == 0L) && (bPerm[M_IDX] == 1L) && (bPerm[KA_IDX] == 2L)) {
            permX2_ = TBMMPermX2::PERM_X2_0_1_2;
        } else if ((bPerm[BATCH_IDX] == 0L) && (bPerm[M_IDX] == 2L) && (bPerm[KA_IDX] == 1L)) {
            permX2_ = TBMMPermX2::PERM_X2_0_2_1;
        }
    }
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        batchSplitFactor_ = std::max(*(attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1)), 1);
        batchSplitMode_ = batchSplitFactor_ > 1 ? TBMMBatchSplit::BATCH_SPLIT_TRUE : TBMMBatchSplit::BATCH_SPLIT_FALSE;
    }
}

uint64_t TransposeBatchMatMulAswTiling::GetTilingKey() const
{
    uint64_t tilingKey =
        TBMMTilingKey().SetPermX1(permX1_).SetPermX2(permX2_).SetBatchSplitMode(batchSplitMode_).GetTilingKey();
    return tilingKey;
}

uint64_t TransposeBatchMatMulAswTiling::GetNumBlocks() const
{
    return compileInfo_.aicNum;
}

ge::graphStatus TransposeBatchMatMulAswTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3TilingData>(tiling);
}

ge::graphStatus TransposeBatchMatMulAswTiling::GetTilingDataProcess(BatchMatMulV3TilingData& tilingData) const
{
    tilingData.batchSplitFactor = batchSplitFactor_;
    return MatMulV3BaseTiling::GetTilingDataProcess(tilingData);
}

} // namespace transpose_batch_mat_mul_advanced
} // namespace optiling