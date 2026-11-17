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

#include <cstdint>
#include <initializer_list>
#include <limits>
#include "op_api/aclnn_util.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
namespace {
static const std::initializer_list<op::DataType> VALUES_ONLY_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64,  op::DataType::DT_INT32,   op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT64, op::DataType::DT_UINT32,  op::DataType::DT_UINT16, op::DataType::DT_UINT8,
    op::DataType::DT_BF16,   op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

// Input/output prerequisites for the Regbase values-only fast path. Callers must separately reject
// returnInverse/returnCounts. False selects the existing fallback, not an ACLNN parameter error.
bool CanUseValuesOnlyAicore(const aclTensor* self, const aclTensor* valueOut)
{
    if (!Ops::NN::AclnnUtil::IsRegbase() || self == nullptr || valueOut == nullptr) {
        return false;
    }
    const auto dtype = self->GetDataType();
    const bool supported = CheckType(dtype, VALUES_ONLY_DTYPE_SUPPORT_LIST);
    const int64_t elements = self->GetViewShape().GetShapeSize();
    // Tiling selects 32/64-bit radix counters from N; input dtype does not determine counter width.
    // Contiguous, zero-offset storage is required: input spans exactly N elements; output needs capacity
    // for N elements (not just the final unique count), since it is borrowed during sorting.
    if (!supported || valueOut->GetDataType() != dtype || elements <= 0 || !op::IsContiguous(self) ||
        !op::IsContiguous(valueOut) || self->GetViewOffset() != 0 || valueOut->GetViewOffset() != 0 ||
        self->GetStorageShape().GetShapeSize() != elements || valueOut->GetStorageShape().GetShapeSize() < elements) {
        return false;
    }
    auto input = reinterpret_cast<uintptr_t>(self->GetData());
    auto output = reinterpret_cast<uintptr_t>(valueOut->GetData());
    const auto elementBytes = op::TypeSize(dtype);
    if (elementBytes <= 0 ||
        static_cast<uint64_t>(elements) > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(elementBytes)) {
        return false;
    }
    const uint64_t bytes = static_cast<uint64_t>(elements) * elementBytes;
    // Known addresses must not overlap over the full N-element range. An unresolved address is allowed
    // for executor-owned intermediates: callers allocate distinct tensors and rely on the executor to
    // keep simultaneously live input/output storage separate until execution.
    return input == 0 || output == 0 || (input <= output ? output - input >= bytes : input - output >= bytes);
}

} // namespace

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
    if (!returnInverse && !returnCounts && CanUseValuesOnlyAicore(self, valueOut)) {
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
    if (!returnInverse && CanUseValuesOnlyAicore(self, valueOut)) {
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
