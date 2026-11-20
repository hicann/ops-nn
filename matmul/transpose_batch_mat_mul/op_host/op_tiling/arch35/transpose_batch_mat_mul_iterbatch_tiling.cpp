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
 * \file transpose_batch_mat_mul_iterbatch_tiling.cpp
 * \brief Reuse BMM IterBatch capability and tiling computation.
 */

#include "transpose_batch_mat_mul_iterbatch_tiling.h"
#include "transpose_batch_mat_mul_tiling_strategy.h"
#include "transpose_batch_mat_mul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.h"

namespace optiling {
namespace transpose_batch_mat_mul_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(TransposeBatchMatMul, TransposeBatchMatMulIterBatchTiling, DAV_3510, ITER_BATCH);

bool TransposeBatchMatMulIterBatchTiling::IsCapable()
{
    const auto* attrs = context_->GetAttrs();
    const bool hasScaleOrBatchSplit = context_->GetOptionalInputShape(SCALE_IDX) != nullptr ||
                                      (attrs->GetAttrNum() >= ATTR_NUM &&
                                       *attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1) != 1);
    // scale和batch split沿用原basic路径，不扩展到新iterbatch模板。
    if (hasScaleOrBatchSplit || args_.cType == ge::DT_INT8) {
        return false;
    }
    return BatchMatMulV3IterBatchBasicApiTiling::IsCapable();
}

ge::graphStatus TransposeBatchMatMulIterBatchTiling::DoOpTiling()
{
    const auto* attrs = context_->GetAttrs();
    const auto* aPerm = static_cast<const int64_t*>(attrs->GetAttrPointer<gert::ContinuousVector>(0)->GetData());
    const auto* bPerm = static_cast<const int64_t*>(attrs->GetAttrPointer<gert::ContinuousVector>(1)->GetData());
    const auto* yPerm = static_cast<const int64_t*>(attrs->GetAttrPointer<gert::ContinuousVector>(NUM_TWO)->GetData());
    permX1_ = aPerm[BATCH_IDX] == 1L ? TBMMPermX1::PERM_X1_1_0_2 :
                                       (args_.isATrans ? TBMMPermX1::PERM_X1_0_2_1 : TBMMPermX1::PERM_X1_0_1_2);
    if (bPerm[BATCH_IDX] == 1L) {
        permX2_ = args_.isBTrans ? TBMMPermX2::PERM_X2_1_2_0 : TBMMPermX2::PERM_X2_1_0_2;
    } else {
        permX2_ = args_.isBTrans ? TBMMPermX2::PERM_X2_0_2_1 : TBMMPermX2::PERM_X2_0_1_2;
    }
    const bool isBNonContiguous = permX2_ == TBMMPermX2::PERM_X2_1_0_2 || permX2_ == TBMMPermX2::PERM_X2_1_2_0 ||
                                  (args_.isBTrans && context_->InputIsView(1UL) &&
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

    const auto ret = BatchMatMulV3IterBatchBasicApiTiling::DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 仅适配TBMM非连续B的batch分组，容量、分块和Fixpipe选择全部复用BMM。
    if (isBNonContiguous) {
        runInfo_.innerBatch = batchInfo_->batchC;
        while (runInfo_.innerBatch % runInfo_.iterBatchL1 != 0UL) {
            --runInfo_.iterBatchL1;
        }
        runInfo_.iterBatchL0 = std::min(runInfo_.iterBatchL0, runInfo_.iterBatchL1);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t TransposeBatchMatMulIterBatchTiling::GetTilingKey() const
{
    return TBMMTilingKey()
        .SetApiLevel(TBMMApiLevel::TBMM_TENSOR_LEVEL)
        .SetPermX1(permX1_)
        .SetPermX2(permX2_)
        .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_FALSE)
        .SetModel(TBMMModel::TBMM_ITER_BATCH)
        .SetL0C2Out(static_cast<TBMML0C2Out>(l0C2Out_))
        .SetNonContiguousType(nonContiguousType_)
        .GetTilingKey();
}

} // namespace transpose_batch_mat_mul_advanced
} // namespace optiling
