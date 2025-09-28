/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gather_nd.h"
#include "gather_nd.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

/* aclnnGatherNd的完整计算流程如下:
 *   self                indices
 *     |                    |
 *     |                    |
 *  Contiguous          Contiguous
 * (workspace_0)       (workspace_1)
 *     \                   /
 *      \                 /
 *       \               /
 *      GatherNd(workspace_2)
 *              |
 *              |
 *           ViewCopy
 *              |
 *             out
 */

static constexpr int64_t MIN_DIM_LEN = 1;
static constexpr int64_t MAX_DIM_LEN = 8;
static const std::initializer_list<DataType> EMPTY_LIST = {};

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_310P = {
  DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16, DataType::DT_INT8,
  DataType::DT_UINT8, DataType::DT_BOOL};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910 = {
  DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16, DataType::DT_INT8,
  DataType::DT_UINT8, DataType::DT_BOOL};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910B = {
  DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16, DataType::DT_INT8,
  DataType::DT_UINT8, DataType::DT_BOOL, DataType::DT_BF16};

static const std::initializer_list<DataType> INDICES_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static inline bool HasEmptyTensor(const aclTensor *self, const aclTensor *indices) {
  return self->IsEmpty() || indices->IsEmpty();
}

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *indices, const aclTensor *out) {
  // self、indices、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static const std::initializer_list<DataType>& GetOutDtypeSupportList() {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910_95:
    case SocVersion::ASCEND910_93:
    case SocVersion::ASCEND910B: {
      return DTYPE_SUPPORT_LIST_910B;
    }
    case SocVersion::ASCEND310P:
      return DTYPE_SUPPORT_LIST_310P;
    case SocVersion::ASCEND910:
      return DTYPE_SUPPORT_LIST_910;
    default: {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
      return EMPTY_LIST;
    }
  }
}

static inline bool CheckDtypeValid(const aclTensor *self, const aclTensor *indices, const aclTensor *out) {
  // self和out数据类型必须一样
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, GetOutDtypeSupportList(), return false);

  // 检查indices的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_SUPPORT_LIST, return false);

  return true;
}

static inline bool CheckOutShape(const aclTensor *self, const aclTensor *indices, const aclTensor *out) {
  // 根据算子语义，推导算子输出shape
  op::Shape outShape;
  int64_t rankSelf = self->GetViewShape().GetDimNum();
  int64_t rankIndices = indices->GetViewShape().GetDimNum();
  int64_t indicesLastElement = indices->GetViewShape().GetDim(rankIndices - 1);
  for (int64_t i = 0; i < rankIndices - 1; ++i) {
    outShape.AppendDim(indices->GetViewShape().GetDim(i));
  }
  for (int64_t i = indicesLastElement; i < rankSelf; ++i) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }

  // 判断out的shape与推导出的输出shape是否相等
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);

  return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *indices, const aclTensor *out) {
  // self、indices的数据维度不能超过8以及不能小于1
  OP_CHECK_MIN_DIM(self, MIN_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MIN_DIM(indices, MIN_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(indices, MAX_DIM_LEN, return false);

  int64_t rankSelf = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  int64_t rankIndices = static_cast<int64_t>(indices->GetViewShape().GetDimNum());
  int64_t indicesLastDim = static_cast<int64_t>(indices->GetViewShape().GetDim(rankIndices - 1));
  if (!(indicesLastDim >= 1 && indicesLastDim <= rankSelf)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices last dim %ld not in valid range: [1, %ld]", indicesLastDim, rankSelf);
    return false;
  }

  return CheckOutShape(self, indices, out);
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *indices, const aclTensor *out) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, indices, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, indices, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, indices, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGatherNdGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, bool negativeIndexSupport,
                                          aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnGatherNd, DFX_IN(self, indices, negativeIndexSupport), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, indices, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 算子的空tensor的支持情况
  if (HasEmptyTensor(self, indices)) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // indices如果非连续，需要转连续
  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子GatherNd进行计算
  auto gatherNdResult = l0op::GatherNd(selfContiguous, indicesContiguous, uniqueExecutor.get(), negativeIndexSupport);
  CHECK_RET(gatherNdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(gatherNdResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGatherNd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGatherNd);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
