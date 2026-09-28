/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unsorted_segment_sum.cpp
 * \brief
 */

#include "unsorted_segment_sum.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(UnsortedSegmentSum);

const aclTensor* UnsortedSegmentSum(const aclTensor* x, const aclTensor* segmentIds, const aclTensor* numSegments,
                                    const op::Shape& outShape, aclOpExecutor* executor)
{
    L0_DFX(UnsortedSegmentSum, x, segmentIds, numSegments);
    auto out = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK_NULL(out, return nullptr);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(UnsortedSegmentSum, OP_INPUT(x, segmentIds, numSegments), OP_OUTPUT(out),
                                           OP_ATTR(false, false));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "UnsortedSegmentSum ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}
} // namespace l0op
