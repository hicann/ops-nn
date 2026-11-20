/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "union_sparse_gen_l0.h"
#include "../op_host/union_sparse_gen_contract.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace Contract = UnionSparseGenContract;

namespace l0op {
OP_TYPE_REGISTER(UnionSparseGen);

UnionSparseGenOutputs UnionSparseGen(const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount,
                                     const aclTensor* cuSqlenQ, const aclTensor* cuSqlenKv, const aclTensor* seqUsedQ,
                                     const aclTensor* seqUsedKv, const aclIntArray* blockShape, int64_t unionSize,
                                     int64_t maskType, int64_t totalqUnionNum, int64_t maxLogicalBlocks,
                                     int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t layoutSparsePattern,
                                     bool isInputIdxSorted, int64_t residualBlockMode, aclOpExecutor* executor,
                                     aclnnStatus& status)
{
    const int64_t blockY = blockShape->GetData()[Contract::kBlockYAxis];
    status = ACLNN_SUCCESS;
    int64_t capacity = 0;
    if (!Contract::Capacity(sparseBlockIdx->GetViewShape().GetDim(Contract::kSparseEntryAxis), unionSize,
                            maxLogicalBlocks, residualBlockMode, capacity)) {
        status = ACLNN_ERR_PARAM_INVALID;
        OP_LOGE(status, "Invalid UnionSparseGen output capacity geometry.");
        return UnionSparseGenOutputs{nullptr, nullptr, nullptr};
    }
    L0_DFX(UnionSparseGen, sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, blockShape,
           unionSize, maskType, totalqUnionNum, maxLogicalBlocks, maxQSeqlen, maxKvSeqlen, layoutSparsePattern,
           isInputIdxSorted, residualBlockMode);
    const auto heads = sparseBlockIdx->GetViewShape().GetDim(Contract::kHeadAxis);
    Shape unionShape({heads, totalqUnionNum, capacity});
    Shape countShape({heads, totalqUnionNum});
    Shape maskShape({heads, totalqUnionNum, capacity * blockY});
    const aclTensor* unionIdx = executor->AllocTensor(unionShape, DataType::DT_INT32, Format::FORMAT_ND);
    const aclTensor* unionCount = executor->AllocTensor(countShape, DataType::DT_INT32, Format::FORMAT_ND);
    const aclTensor* exactMask = executor->AllocTensor(maskShape, DataType::DT_INT32, Format::FORMAT_ND);
    if (unionIdx == nullptr || unionCount == nullptr || exactMask == nullptr) {
        status = ACLNN_ERR_INNER_NULLPTR;
        OP_LOGE(status, "UnionSparseGen output allocation failed.");
        return UnionSparseGenOutputs{nullptr, nullptr, nullptr};
    }
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        UnionSparseGen, OP_INPUT(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv),
        OP_OUTPUT(unionIdx, unionCount, exactMask),
        OP_ATTR(blockShape, unionSize, maskType, totalqUnionNum, maxLogicalBlocks, maxQSeqlen, maxKvSeqlen,
                layoutSparsePattern, isInputIdxSorted, residualBlockMode));
    if (ret != ACLNN_SUCCESS) {
        status = ret;
        // The launcher already reports its cause. Add log context without
        // replacing that structured diagnostic with a generic internal error.
        OP_LOGE_WITHOUT_REPORT(status, "UnionSparseGen ADD_TO_LAUNCHER_LIST_AICORE failed, status=%d.", status);
        return UnionSparseGenOutputs{nullptr, nullptr, nullptr};
    }
    return {unionIdx, unionCount, exactMask};
}
} // namespace l0op
