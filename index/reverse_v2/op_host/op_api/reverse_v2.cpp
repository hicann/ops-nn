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
 * \file reverse_v2.cpp
 * \brief
 */

#include "reverse_v2.h"
#include "opdev/data_type_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ReverseV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_INT64};

static const std::initializer_list<DataType> ASCEND910_95_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_UINT16,
  op::DataType::DT_INT32, op::DataType::DT_UINT32, op::DataType::DT_INT64, op::DataType::DT_UINT64,
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE,
  op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *self) {
  // ReverseV2只需要判断dtype
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
    return CheckType(self->GetDataType(), ASCEND910_95_DTYPE_SUPPORT_LIST);
  }
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static inline bool ReverseV2InferShape(const op::Shape &selfShape, op::Shape &outShape) {
  outShape = selfShape;
  return true;
}


const aclTensor *ReverseV2Aicpu(const aclTensor *self, const aclTensor *dim, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(ReverseV2Aicpu, self, dim, out)

  static internal::AicpuTaskSpace space("ReverseV2", ge::DEPEND_IN_SHAPE, true);
  op::DataType dtype = dim->GetDataType();
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ReverseV2, OP_ATTR_NAMES({"Tidx"}), OP_INPUT(self, dim), OP_OUTPUT(out), OP_ATTR(dtype));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

  return out;
}


const aclTensor *ReverseV2Aicore(const aclTensor *self, const aclTensor *dim, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(ReverseV2Aicore, self, dim, out)
  ADD_TO_LAUNCHER_LIST_AICORE(ReverseV2,
                              OP_INPUT(self, dim),
                              OP_OUTPUT(out));
  return out;
}

const aclTensor *ReverseV2(const aclTensor *self, const aclIntArray *dims, aclOpExecutor *executor) {
  op::Shape outShape;
  if (!ReverseV2InferShape(self->GetViewShape(), outShape)) {
    OP_LOGE(ACL_ERROR_INVALID_PARAM, "infer shape failed. StroageShape[%s], ViewShape[%s] ",
            op::ToString(self->GetStorageShape()).GetString(),
            op::ToString(self->GetViewShape()).GetString());
    return nullptr;
  }

  auto dim = executor->ConvertToTensor(dims, op::ToOpDataType(ACL_INT64));
  auto out = executor->AllocTensor(outShape, self->GetDataType());

  if (IsAiCoreSupport(self)) {
    return ReverseV2Aicore(self, dim, out, executor);
  } else {
    return ReverseV2Aicpu(self, dim, out, executor);
  }
}
} // l0op
