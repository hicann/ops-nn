/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "gather_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GatherV2);

static constexpr size_t MODE_GatherV2_NUM = 3;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_SELF = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_WITH_BF16 = {
    DataType::DT_FLOAT,  DataType::DT_INT32,     DataType::DT_INT64,      DataType::DT_FLOAT16, DataType::DT_BF16,
    DataType::DT_INT16,  DataType::DT_UINT16,    DataType::DT_INT8,       DataType::DT_UINT8,   DataType::DT_BOOL,
    DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_INDICES = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_AXIS = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static bool IsAiCoreSupport(const aclTensor *self,
                            const aclTensor *indices,
                            const aclTensor *axis) {
  if (Ops::NN::AclnnUtil::IsRegbase()) {
    auto checkSelfTypeD = CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_WITH_BF16);
    auto checkIndicesTypeD = CheckType(indices->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_INDICES);
    auto checkAxisTypeD = CheckType(axis->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_AXIS);
    return (checkSelfTypeD && checkIndicesTypeD && checkAxisTypeD);
  }
  auto checkSelfType = CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_SELF);
  auto checkIndicesType = CheckType(indices->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_INDICES);
  auto checkAxisType = CheckType(axis->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_AXIS);
  bool supportBF16 = self->GetDataType() == op::DataType::DT_BF16
                     && (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201);
  return ((checkSelfType || supportBF16) && checkIndicesType && checkAxisType);
}

const aclTensor *GatherV2AiCore(const aclTensor *self, const aclTensor *indices, const aclTensor *axis,
                           aclTensor *gatherV2Out, int batchDims, bool negativeIndexSupport, aclOpExecutor *executor) {
  L0_DFX(GatherV2AiCore, self, indices, axis, batchDims, negativeIndexSupport);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GatherV2,
                                         OP_INPUT(self, indices, axis),
                                         OP_OUTPUT(gatherV2Out),
                                         OP_ATTR(batchDims, negativeIndexSupport));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GatherV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return gatherV2Out;
}

static void GatherV2WithImplModeAiCore(const aclTensor *self, const aclTensor *indices, const aclTensor *axis,
                                       aclTensor *gatherV2Out, int batchDims, bool negativeIndexSupport,
                                       int64_t implMode, aclOpExecutor *executor) {
  L0_DFX(GatherV2WithImplModeAiCore, self, indices, axis, batchDims, negativeIndexSupport, implMode);

  OpImplMode mode[MODE_GatherV2_NUM] = {OpImplMode::IMPL_MODE_HIGH_PRECISION, OpImplMode::IMPL_MODE_HIGH_PERFORMANCE,
                                        OpImplMode::IMPL_MODE_SUPPORT_OUT_OF_BOUND_INDEX};
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(GatherV2,
                                               OP_INPUT(self, indices, axis),
                                               OP_OUTPUT(gatherV2Out),
                                               OP_ATTR(batchDims, negativeIndexSupport),
                                               OP_OPTION(mode[static_cast<size_t>(implMode)]));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return,
                                       "GatherV2WithImplMode add to aicore launch list failed.");
}

const aclTensor *GatherV2AiCPU(const aclTensor *self, const aclTensor *indices, const aclTensor *axis,
                               aclTensor *gatherV2Out, int batchDims, bool negativeIndexSupport,
                               aclOpExecutor *executor) {
  L0_DFX(GatherV2AiCPU, self, indices, axis, batchDims, negativeIndexSupport);

  static internal::AicpuTaskSpace space("GatherV2");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(GatherV2,
                                        OP_ATTR_NAMES({"batch_dims", "negative_index_support"}),
                                        OP_INPUT(self, indices, axis),
                                        OP_OUTPUT(gatherV2Out),
                                        OP_ATTR(batchDims, negativeIndexSupport));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GatherV2AiCPU ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return gatherV2Out;
}

const aclTensor *GatherV2WithImplMode(const aclTensor *self, int64_t axis, const aclTensor *indices,
  int64_t implMode, aclOpExecutor *executor,
  int batchDims, bool negativeIndexSupport) {
  int64_t selfDim = self->GetViewShape().GetDimNum() > 0 ? self->GetViewShape().GetDimNum() : 1;
  if (axis < 0) {
    axis += selfDim;
  }

  // 根据算子语义，推导算子输出shape
  op::Shape outShape;
  for (int64_t i = 0; i < axis; i++) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }
  for (size_t i = batchDims; i < indices->GetViewShape().GetDimNum(); i++) {
    outShape.AppendDim(indices->GetViewShape().GetDim(i));
  }
  for (size_t i = axis + 1; i < self->GetViewShape().GetDimNum(); i++) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }

  // 当self是零维tensor时，上述推导公式不再适用，不管一维indices中有多少个0，out始终是零维tensor
  if (self->GetViewShape().GetDimNum() == 0) {
    outShape = self->GetViewShape();
  }

  // 根据推导出的输出shape申请输出tensor
  auto gatherV2Out = executor->AllocTensor(outShape, self->GetDataType(), op::Format::FORMAT_ND);
  const aclTensor *axisTensor = executor->ConvertToTensor(&axis, 1, op::DataType::DT_INT64);
  if (IsAiCoreSupport(self, indices, axisTensor)) {
    GatherV2WithImplModeAiCore(self, indices, axisTensor, gatherV2Out, batchDims, negativeIndexSupport, implMode,
                               executor);
  } else {
    GatherV2AiCPU(self, indices, axisTensor, gatherV2Out, batchDims, negativeIndexSupport, executor);
  }
  return gatherV2Out;
}

const aclTensor *GatherV2(const aclTensor *self, int64_t axis, const aclTensor *indices, aclOpExecutor *executor,
                          int batchDims, bool negativeIndexSupport) {
  int64_t selfDim = self->GetViewShape().GetDimNum() > 0 ? self->GetViewShape().GetDimNum() : 1;
  if (axis < 0) {
    axis += selfDim;
  }

  // 根据算子语义，推导算子输出shape
  op::Shape outShape;
  for (int64_t i = 0; i < axis; i++) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }
  for (size_t i = batchDims; i < indices->GetViewShape().GetDimNum(); i++) {
    outShape.AppendDim(indices->GetViewShape().GetDim(i));
  }
  for (size_t i = axis + 1; i < self->GetViewShape().GetDimNum(); i++) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }

  // 当self是零维tensor时，上述推导公式不再适用，不管一维indices中有多少个0，out始终是零维tensor
  if (self->GetViewShape().GetDimNum() == 0) {
    outShape = self->GetViewShape();
  }

  // 根据推导出的输出shape申请输出tensor
  auto gatherV2Out = executor->AllocTensor(outShape, self->GetDataType(), op::Format::FORMAT_ND);
  const aclTensor *axisTensor = executor->ConvertToTensor(&axis, 1, op::DataType::DT_INT64);
  if (IsAiCoreSupport(self, indices, axisTensor)) {
    return GatherV2AiCore(self, indices, axisTensor, gatherV2Out, batchDims, negativeIndexSupport, executor);
  } else {
    return GatherV2AiCPU(self, indices, axisTensor, gatherV2Out, batchDims, negativeIndexSupport, executor);
  }
}
} // l0op