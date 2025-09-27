/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file unique_with_counts_ext2.cpp
 * \brief
 */

#include "unique_with_counts_ext2.h"
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
OP_TYPE_REGISTER(UniqueWithCountsExt2);

aclnnStatus UniqueWithCountsExt2(const aclTensor* self, bool sorted, bool returnInverse, int64_t dim, aclTensor* valueOut,
                                 aclTensor* inverseOut, aclTensor* countsOut, aclOpExecutor* executor) {
  L0_DFX(UniqueWithCountsExt2, self, sorted, returnInverse, dim, valueOut, inverseOut, countsOut);

  const aclScalar *dimScalar = executor->AllocScalar(dim);
  const aclTensor *dimTensor = executor->ConvertToTensor(dimScalar, op::DataType::DT_INT64);
  aclTensor* idxOut = executor->AllocTensor(inverseOut->GetViewShape(), inverseOut->GetDataType());
  static internal::AicpuTaskSpace space("UniqueWithCountsExt2", ge::DEPEND_SHAPE_RANGE, false);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(UniqueWithCountsExt2, OP_ATTR_NAMES({"sorted", "return_inverse"}),
                                        OP_INPUT(self, dimTensor), OP_OUTPUT(valueOut, idxOut, countsOut, inverseOut),
                                        OP_ATTR(sorted, returnInverse));
  return ret;
}
}  // namespace l0op
