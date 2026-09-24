/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unique_with_counts_and_sorting.h"
#include "../../unique/op_api/unique_common.h"
#include "op_api/aclnn_util.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(UniqueWithCountsAndSorting);

namespace {
// Three shape records (values/inverse/counts), each containing rank metadata and up to eight dimensions.
// Keep 3 * (1 + 8) slots even for disabled outputs, matching UniqueConsecutive's SHAPE_LEN/layout.
constexpr int64_t SHAPE_STORAGE_ELEMENTS = 27;

aclnnStatus LaunchValuesOnly(const aclTensor* self, bool sorted, aclTensor* valueOut, aclOpExecutor* executor)
{
    // Disabled outputs are private placeholders: UC writes shape=1 but no data.
    // Do not resize the caller's empty inverse/count tensors through these slots.
    auto inverse = executor->AllocTensor(op::Shape{1}, op::DataType::DT_INT64, op::Format::FORMAT_ND);
    auto counts = executor->AllocTensor(op::Shape{1}, op::DataType::DT_INT64, op::Format::FORMAT_ND);
    auto shape = executor->AllocTensor(op::Shape{SHAPE_STORAGE_ELEMENTS}, op::DataType::DT_INT64,
                                       op::Format::FORMAT_ND);
    if (inverse == nullptr || counts == nullptr || shape == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ADD_TO_LAUNCHER_LIST_AICORE(UniqueWithCountsAndSorting, OP_INPUT(self), OP_OUTPUT(valueOut, inverse, counts),
                                       OP_ATTR(false, false, sorted, static_cast<int64_t>(op::DataType::DT_INT64)),
                                       OP_OUTSHAPE({shape, 0}), OP_OUTSHAPE({shape, 1}), OP_OUTSHAPE({shape, 2}));
}
} // namespace

aclnnStatus UniqueWithCountsAndSorting(const aclTensor* self, bool sorted, bool returnInverse, bool returnCounts,
                                       aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                                       aclOpExecutor* executor)
{
    L0_DFX(UniqueWithCountsAndSorting, self, sorted, returnInverse, returnCounts, valueOut, inverseOut, countsOut);
    if (!returnInverse && !returnCounts && UniqueCommon::CanUseUniqueWithCountsAndSortingAicore(self, valueOut)) {
        return LaunchValuesOnly(self, sorted, valueOut, executor);
    }
    OP_LOGW("Using UniqueWithCountsAndSorting with count out");

    static internal::AicpuTaskSpace space("UniqueWithCountsAndSorting", ge::DEPEND_SHAPE_RANGE);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        UniqueWithCountsAndSorting, OP_ATTR_NAMES({"sorted", "return_inverse", "return_counts"}), OP_INPUT(self),
        OP_OUTPUT(valueOut, inverseOut, countsOut), OP_ATTR(sorted, returnInverse, returnCounts));
    return ret;
}

aclnnStatus UniqueWithCountsAndSorting(const aclTensor* self, bool sorted, bool returnInverse, aclTensor* valueOut,
                                       aclTensor* inverseOut, aclOpExecutor* executor)
{
    L0_DFX(UniqueWithCountsAndSorting, self, sorted, returnInverse, valueOut, inverseOut);
    if (!returnInverse && UniqueCommon::CanUseUniqueWithCountsAndSortingAicore(self, valueOut)) {
        return LaunchValuesOnly(self, sorted, valueOut, executor);
    }
    OP_LOGW("Using UniqueWithCountsAndSorting without count out");

    static internal::AicpuTaskSpace space("UniqueWithCountsAndSorting", ge::DEPEND_SHAPE_RANGE);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(UniqueWithCountsAndSorting, OP_ATTR_NAMES({"sorted", "return_inverse"}),
                                          OP_INPUT(self), OP_OUTPUT(valueOut, inverseOut),
                                          OP_ATTR(sorted, returnInverse));
    return ret;
}
} // namespace l0op
