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

aclnnStatus UniqueWithCountsAndSorting(const aclTensor* self, bool sorted, bool returnInverse, bool returnCounts,
                                       aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                                       aclOpExecutor* executor) {
  OP_LOGW("Using UniqueWithCountsAndSorting with count out");
  L0_DFX(UniqueWithCountsAndSorting, self, sorted, returnInverse, returnCounts, valueOut, inverseOut, countsOut);

  static internal::AicpuTaskSpace space("UniqueWithCountsAndSorting", ge::DEPEND_SHAPE_RANGE);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(UniqueWithCountsAndSorting, OP_ATTR_NAMES({"sorted", "return_inverse", "return_counts"}),
                                        OP_INPUT(self), OP_OUTPUT(valueOut, inverseOut, countsOut),
                                        OP_ATTR(sorted, returnInverse, returnCounts));
  return ret;
}

aclnnStatus UniqueWithCountsAndSorting(const aclTensor* self, bool sorted, bool returnInverse, aclTensor* valueOut,
                                       aclTensor* inverseOut, aclOpExecutor* executor) {
  OP_LOGW("Using UniqueWithCountsAndSorting without count out");
  L0_DFX(UniqueWithCountsAndSorting, self, sorted, returnInverse, valueOut, inverseOut);

  static internal::AicpuTaskSpace space("UniqueWithCountsAndSorting", ge::DEPEND_SHAPE_RANGE);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(UniqueWithCountsAndSorting, OP_ATTR_NAMES({"sorted", "return_inverse"}), OP_INPUT(self),
                                        OP_OUTPUT(valueOut, inverseOut), OP_ATTR(sorted, returnInverse));
  return ret;
}
}  // namespace l0op
