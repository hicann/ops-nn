/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "op_api/aclnn_util.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UniqueWithCountsExt2);
OP_TYPE_REGISTER(UniqueDim);

constexpr int64_t OUT_SHAPE_SIZE = 27;

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST_ASCEND950 = {
    op::DataType::DT_UINT8,  op::DataType::DT_INT8,  op::DataType::DT_UINT16,  op::DataType::DT_INT16,
    op::DataType::DT_UINT32, op::DataType::DT_INT32, op::DataType::DT_UINT64,  op::DataType::DT_INT64,
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_BOOL};

static bool CheckSupport4UniqueDim(const aclTensor* self)
{
    OP_CHECK(Ops::NN::AclnnUtil::IsRegbase(), OP_LOGW("UniqueDim AICore only support ASCEND950."), return false);
    OP_CHECK(CheckType(self->GetDataType(), X_DTYPE_SUPPORT_LIST_ASCEND950),
             OP_LOGW("UniqueDim AICore only support bf16/fp16/fp32 input."), return false);
    return true;
}

static aclnnStatus UniqueDimAiCore(const aclTensor* self, bool sorted, bool returnInverse, int64_t dim,
                                   aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                                   aclOpExecutor* executor)
{
    L0_DFX(UniqueDimAiCore, self, sorted, returnInverse, dim, valueOut, inverseOut, countsOut);

    Shape outShapeShape{OUT_SHAPE_SIZE};
    auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);

    aclnnStatus ret = ACLNN_SUCCESS;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(UniqueDim, OP_INPUT(self), OP_OUTPUT(valueOut, inverseOut, countsOut),
                                      OP_ATTR(dim, sorted, returnInverse, true), OP_OUTSHAPE({outShapeTensor, 0}),
                                      OP_OUTSHAPE({outShapeTensor, 1}), OP_OUTSHAPE({outShapeTensor, 2}));
    return ret;
}

aclnnStatus UniqueWithCountsExt2(const aclTensor* self, bool sorted, bool returnInverse, int64_t dim,
                                 aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                                 aclOpExecutor* executor)
{
    L0_DFX(UniqueWithCountsExt2, self, sorted, returnInverse, dim, valueOut, inverseOut, countsOut);

    if (CheckSupport4UniqueDim(self)) {
        return UniqueDimAiCore(self, sorted, returnInverse, dim, valueOut, inverseOut, countsOut, executor);
    }

    const aclScalar* dimScalar = executor->AllocScalar(dim);
    const aclTensor* dimTensor = executor->ConvertToTensor(dimScalar, op::DataType::DT_INT64);
    aclTensor* idxOut = executor->AllocTensor(inverseOut->GetViewShape(), inverseOut->GetDataType());
    static internal::AicpuTaskSpace space("UniqueWithCountsExt2", ge::DEPEND_SHAPE_RANGE, false);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(UniqueWithCountsExt2, OP_ATTR_NAMES({"sorted", "return_inverse"}),
                                          OP_INPUT(self, dimTensor), OP_OUTPUT(valueOut, idxOut, countsOut, inverseOut),
                                          OP_ATTR(sorted, returnInverse));
    return ret;
}
} // namespace l0op
