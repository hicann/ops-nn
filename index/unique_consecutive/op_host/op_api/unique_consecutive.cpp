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
 * \file unique_consecutive.cpp
 * \brief
 */
#include "unique_consecutive.h"
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
OP_TYPE_REGISTER(UniqueConsecutive);

constexpr int64_t NoneN = 1000;
constexpr int64_t OUT_SHAPE_SIZE = 27;

static const std::initializer_list<op::DataType> XY_DTYPE_SUPPORT_LIST_ASCEND910_95 = {
    op::DataType::DT_INT64,  op::DataType::DT_INT32,   op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT64, op::DataType::DT_UINT32,  op::DataType::DT_UINT16, op::DataType::DT_UINT8,
    op::DataType::DT_BF16,   op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> IDX_DTYPE_SUPPORT_LIST_ASCEND910_95 = {op::DataType::DT_INT32, op::DataType::DT_INT64};

bool CheckAttr4Aicore(bool returnInverse, int64_t dim)
{
    // Add Soc check for future....
    OP_CHECK(!returnInverse,
             OP_LOGW("UniqueConsecutive4Aicore only support returnInverse=False."),
             return false);
    OP_CHECK(NoneN == dim, OP_LOGW("UniqueConsecutive4Aicore only support flatten case."),
             return false);
    return true;
}

bool CheckTensorDtype4Aicore(const aclTensor* self, const aclTensor* valueOut, const aclTensor* inverseOut, const aclTensor* countsOut)
{
    OP_CHECK(CheckType(self->GetDataType(), XY_DTYPE_SUPPORT_LIST_ASCEND910_95),
             OP_LOGW("Unsupport input dtype for aicore UniqueConsecutive."), return false);
    OP_CHECK(CheckType(countsOut->GetDataType(), IDX_DTYPE_SUPPORT_LIST_ASCEND910_95),
             OP_LOGW("Unsupport count dtype for aicore UniqueConsecutive."), return false);
    OP_CHECK(inverseOut->GetDataType() == countsOut->GetDataType(),
             OP_LOGW("Inverse and Count should have same Dtype."), return false);
    OP_CHECK(self->GetDataType() == valueOut->GetDataType(), OP_LOGW("Inverse and Count should have same Dtype."),
             return false);
    return true;
}

bool CheckSupport4Aicore(const aclTensor* self, bool returnInverse, int64_t dim, aclTensor* valueOut,
                         aclTensor* inverseOut, aclTensor* countsOut)
{
    SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(SocVersion::ASCEND910_95 == version, OP_LOGW("Aicore UniqueConsecutive only support ASCEND910_95."),
             return false);

    OP_CHECK(CheckAttr4Aicore(returnInverse, dim), OP_LOGW("Invalid attrs for aicore, use aicpu."), return false);
    OP_CHECK(CheckTensorDtype4Aicore(self, valueOut, inverseOut, countsOut),
             OP_LOGW("Invalid tensor dtype for aicore, use aicpu."), return false);
    return true;
}

aclnnStatus UniqueConsecutiveAiCore(const aclTensor* self, bool returnInverse, bool returnCounts, int64_t dim, aclTensor* valueOut,
                                    aclTensor* inverseOut, aclTensor* countsOut, op::DataType outIdx, aclOpExecutor* executor)
{
    L0_DFX(UniqueConsecutiveAiCore, self, returnInverse, returnCounts, dim, valueOut, inverseOut, countsOut, outIdx);

    Shape outShapeShape{OUT_SHAPE_SIZE};
    auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);

    aclnnStatus ret = ACLNN_SUCCESS;
    if (!returnCounts) {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(UniqueConsecutive,
                                      OP_INPUT(self),
                                      OP_OUTPUT(valueOut, inverseOut, countsOut),
                                      OP_ATTR(returnInverse, returnCounts, NoneN, outIdx),
                                      OP_OUTSHAPE({outShapeTensor, 0})
        );
        return ret;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(UniqueConsecutive,
                                      OP_INPUT(self),
                                      OP_OUTPUT(valueOut, inverseOut, countsOut),
                                      OP_ATTR(returnInverse, returnCounts, NoneN, outIdx),
                                      OP_OUTSHAPE({outShapeTensor, 0}),
                                      OP_OUTSHAPE({outShapeTensor, 1}),
                                      OP_OUTSHAPE({outShapeTensor, 2})
    );
    return ret;
}

aclnnStatus UniqueConsecutive(const aclTensor* self, bool returnInverse, bool returnCounts, int64_t dim, aclTensor* valueOut,
                        aclTensor* inverseOut, aclTensor* countsOut, aclOpExecutor* executor) {
  L0_DFX(UniqueConsecutive, self, returnInverse, returnCounts, dim, valueOut, inverseOut, countsOut);

  if (CheckSupport4Aicore(self, returnInverse, dim, valueOut, inverseOut, countsOut)) {
    op::DataType outIdxType = op::DataType::DT_INT64;
    if (returnInverse) {
        outIdxType = inverseOut->GetDataType();
    } else if (returnCounts) {
        outIdxType = countsOut->GetDataType();
    }
    return UniqueConsecutiveAiCore(self, returnInverse, returnCounts, dim, valueOut, inverseOut, countsOut, outIdxType, executor);
  }

  static internal::AicpuTaskSpace space("UniqueConsecutive", ge::DEPEND_SHAPE_RANGE);
  aclnnStatus ret = ACLNN_SUCCESS;
  if (dim == NoneN) {
    ret = ADD_TO_LAUNCHER_LIST_AICPU(UniqueConsecutive, OP_ATTR_NAMES({"return_idx", "return_counts"}), OP_INPUT(self),
                                     OP_OUTPUT(valueOut, inverseOut, countsOut), OP_ATTR(returnInverse, returnCounts));
  } else {
    ret = ADD_TO_LAUNCHER_LIST_AICPU(UniqueConsecutive, OP_ATTR_NAMES({"return_idx", "return_counts", "axis"}),
                                     OP_INPUT(self), OP_OUTPUT(valueOut, inverseOut, countsOut),
                                     OP_ATTR(returnInverse, returnCounts, dim));
  }
  return ret;
}
}  // namespace l0op
