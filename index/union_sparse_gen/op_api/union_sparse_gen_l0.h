/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_L0_H
#define UNION_SPARSE_GEN_L0_H

#include <array>
#include "../op_host/union_sparse_gen_contract.h"

#include "opdev/op_executor.h"

namespace l0op {
using UnionSparseGenOutputs = std::array<const aclTensor*, UnionSparseGenContract::kOutputCount>;

UnionSparseGenOutputs UnionSparseGen(const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount,
                                     const aclTensor* cuSqlenQ, const aclTensor* cuSqlenKv, const aclTensor* seqUsedQ,
                                     const aclTensor* seqUsedKv, const aclIntArray* blockShape, int64_t unionSize,
                                     int64_t maskType, int64_t totalqUnionNum, int64_t maxLogicalBlocks,
                                     int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t layoutSparsePattern,
                                     bool isInputIdxSorted, int64_t residualBlockMode, aclOpExecutor* executor,
                                     aclnnStatus& status);
} // namespace l0op

#endif
