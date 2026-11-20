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
 * \file transpose_batch_mat_mul_asw_tiling.cpp
 * \brief
 */

#include "transpose_batch_mat_mul_asw_tiling.h"

#include <memory>
#include <new>

#include "transpose_batch_mat_mul_tiling_strategy.h"
#include "transpose_batch_mat_mul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.h"

namespace optiling {
namespace transpose_batch_mat_mul_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(TransposeBatchMatMul, TransposeBatchMatMulAswTiling, DAV_3510, BASE);
MM_REGISTER_TILING_TEMPLATE(TransposeBatchMatMul, TransposeBatchMatMulAswTiling, DAV_9201, BASE);
MM_REGISTER_TILING_TEMPLATE(TransposeBatchMatMul, TransposeBatchMatMulAswTiling, DAV_9202, BASE);

// DAV_RESV (unknown or unmapped npuArch) never reaches this tiling class: the
// template above is registered only for DAV_3510/DAV_9201/DAV_9202, so
// DAV_RESV keeps using the legacy tiling/kernel path.
void TransposeBatchMatMulAswTiling::CheckModel()
{
    // BMN/MBN及非连续输入统一由MatMul layout切片，沿用原核内切K准入条件。
    if (MatMulV3TilingHelper::IsSplitSingleCoreK(compileInfo_, args_, true)) {
        runInfo_.splitKBase = MatMulV3TilingHelper::SelectSplitKBase(context_, args_.mValue, args_.nValue, args_.kValue,
                                                                     runInfo_.baseK);
        model_ = TBMMModel::TBMM_BASIC_SPLIT_K;
    }
}

ge::graphStatus TransposeBatchMatMulAswTiling::DoOpTiling()
{
    GetTransposeBatchMatMulInfo();
    // BMN/MBN统一复用BMM basic的分块、L1/L0 buffer和负载均衡策略。
    const auto ret = BatchMatMulV3AswBasicTiling::DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    if (nonContiguousType_ == TBMMNonContiguousType::B_NON_CONTIGUOUS ||
        nonContiguousType_ == TBMMNonContiguousType::AB_NON_CONTIGUOUS) {
        runInfo_.innerBatch = batchInfo_->batchC;
    }
    CheckModel();
    return ge::GRAPH_SUCCESS;
}

void TransposeBatchMatMulAswTiling::GetTransposeBatchMatMulInfo()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const gert::ContinuousVector* yPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    if ((aPerm[BATCH_IDX] == 1L) && (aPerm[M_IDX] == 0L) && (aPerm[KA_IDX] == NUM_TWO)) {
        permX1_ = TBMMPermX1::PERM_X1_1_0_2;
    } else if ((aPerm[BATCH_IDX] == 0L) && (aPerm[M_IDX] == 1L) && (aPerm[KA_IDX] == NUM_TWO)) {
        permX1_ = TBMMPermX1::PERM_X1_0_1_2;
    } else if (aPerm[BATCH_IDX] == 0L && aPerm[M_IDX] == NUM_TWO && aPerm[KA_IDX] == 1L) {
        permX1_ = TBMMPermX1::PERM_X1_0_2_1;
    }
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    if ((bPerm[BATCH_IDX] == 0L) && (bPerm[M_IDX] == 1L) && (bPerm[KA_IDX] == NUM_TWO)) {
        permX2_ = TBMMPermX2::PERM_X2_0_1_2;
    } else if ((bPerm[BATCH_IDX] == 0L) && (bPerm[M_IDX] == NUM_TWO) && (bPerm[KA_IDX] == 1L)) {
        permX2_ = TBMMPermX2::PERM_X2_0_2_1;
    } else if ((bPerm[BATCH_IDX] == 1L) && (bPerm[M_IDX] == 0L) && (bPerm[KA_IDX] == NUM_TWO)) {
        permX2_ = TBMMPermX2::PERM_X2_1_0_2; // kbn: x2实际存储(k,b,n), n连续
    } else if (bPerm[BATCH_IDX] == 1L && bPerm[KB_IDX] == NUM_TWO && bPerm[N_IDX] == 0L) {
        permX2_ = TBMMPermX2::PERM_X2_1_2_0; // nbk: x2实际存储(n,b,k), k连续
    }
    const int64_t* yPerm = static_cast<const int64_t*>(yPermList->GetData());
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        batchSplitFactor_ = std::max(*(attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1)), 1);
        batchSplitMode_ = batchSplitFactor_ > 1 ? TBMMBatchSplit::BATCH_SPLIT_TRUE : TBMMBatchSplit::BATCH_SPLIT_FALSE;
    }
    // 连续KBN/NBK由perm确定；保留旧021非连续视图的stride识别。
    const bool isBNonContiguous = permX2_ == TBMMPermX2::PERM_X2_1_0_2 || permX2_ == TBMMPermX2::PERM_X2_1_2_0 ||
                                  (permX2_ == TBMMPermX2::PERM_X2_0_2_1 && context_->InputIsView(1UL) &&
                                   MatMulV3TilingHelper::IsTransposeNonContiguous(context_, 1UL));
    const bool isABatchTranspose = permX1_ == TBMMPermX1::PERM_X1_1_0_2;
    if (yPerm[BATCH_IDX] == 0L) {
        if (isBNonContiguous) {
            nonContiguousType_ = isABatchTranspose ? TBMMNonContiguousType::AB_NON_CONTIGUOUS :
                                                     TBMMNonContiguousType::B_NON_CONTIGUOUS;
        } else {
            nonContiguousType_ = isABatchTranspose ? TBMMNonContiguousType::A_NON_CONTIGUOUS :
                                                     TBMMNonContiguousType::CONTIGUOUS;
        }
    } else {
        // MBN不支持非连续B，检查阶段已拦截；只需区分普通A与MBK A。
        nonContiguousType_ = isABatchTranspose ? TBMMNonContiguousType::AC_TRANSPOSE :
                                                 TBMMNonContiguousType::C_TRANSPOSE;
    }
}

uint64_t TransposeBatchMatMulAswTiling::GetTilingKey() const
{
    return TBMMTilingKey()
        .SetModel(model_)
        .SetApiLevel(TBMMApiLevel::TBMM_TENSOR_LEVEL)
        .SetPermX1(permX1_)
        .SetPermX2(permX2_)
        .SetBatchSplitMode(batchSplitMode_)
        .SetNonContiguousType(nonContiguousType_)
        .GetTilingKey();
}

uint64_t TransposeBatchMatMulAswTiling::GetNumBlocks() const { return compileInfo_.aicNum; }

ge::graphStatus TransposeBatchMatMulAswTiling::GetTilingData(TilingResult& tiling) const
{
    std::shared_ptr<TransposeBatchMatMulBasicTilingData> tilingData;
    try {
        tilingData = std::make_shared<TransposeBatchMatMulBasicTilingData>();
    } catch (const std::bad_alloc&) {
        OP_LOGE(context_->GetNodeName(), "Failed to allocate memory for tilingData");
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus ret = GetTilingDataProcess(*tilingData);
    tiling.tilingData = tilingData;
    tiling.tilingDataSize = sizeof(TransposeBatchMatMulBasicTilingData);
    return ret;
}

ge::graphStatus TransposeBatchMatMulAswTiling::GetTilingDataProcess(
    TransposeBatchMatMulBasicTilingData& tilingData) const
{
    tilingData.batchSplitFactor = batchSplitFactor_;
    const ge::graphStatus ret = BatchMatMulV3AswBasicTiling::GetTilingDataProcess(tilingData.matMulTilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    auto& matmulTiling = tilingData.matMulTilingData.matMulTilingData;
    // INT8 scale位于末槽A/B/bias之后，按实际下发的L1尺寸检查四缓冲容量。
    if (args_.cType == ge::DT_INT8 && matmulTiling.l1BufferNum == NUM_FOUR) {
        const uint64_t abL1Size = matmulTiling.kL1 *
                                  (matmulTiling.mL1 * args_.aDtypeSize + matmulTiling.nL1 * args_.bDtypeSize);
        // bias为空时kernel仍按DTYPE_X1预留bias占位，再放置scale。
        const uint64_t biasDtypeSize = args_.hasBias ? GetSizeByDataType(args_.biasType) : args_.aDtypeSize;
        const uint64_t tailL1Size = matmulTiling.nL1 * (biasDtypeSize + sizeof(uint64_t));
        if (abL1Size + tailL1Size > compileInfo_.l1Size / NUM_FOUR) {
            matmulTiling.l1BufferNum = NUM_TWO;
        }
    }
    if (nonContiguousType_ == TBMMNonContiguousType::C_TRANSPOSE ||
        nonContiguousType_ == TBMMNonContiguousType::AC_TRANSPOSE) {
        // MBN根据原始L1分块计算cache策略。
        matmulTiling.l2CacheDisable = SetDisableL2cache(
            runInfo_.baseM * runInfo_.stepM, runInfo_.baseK * runInfo_.stepKa, runInfo_.baseK * runInfo_.stepKb,
            runInfo_.baseN * runInfo_.stepN);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace transpose_batch_mat_mul_advanced
} // namespace optiling
