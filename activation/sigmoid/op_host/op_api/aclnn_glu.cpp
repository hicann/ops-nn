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
 * \file aclnn_glu.cpp
 * \brief
 */
#include "aclnn_glu.h"
#include "sigmoid.h"
#include "level0/mul.h"
#include "level0/split_v.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* GLU 算子的完整计算流程如下:
 *       self                 dim
 *        |                    |
 *         \                   /
 *   Contiguous(workspace_0)  /
 *                  \        /
 *             SplitV(workspace_1)
 *               |          |
 *               |    Sigmoid(workspace_2)
 *               \         /
 *              Mul(workspace_3)
 *                     |
 *                  ViewCopy
 *                     |
 *                    out
 */

constexpr size_t MAX_DIM_LEN = 8;
constexpr int64_t SPLIT_NUM = 2;

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  const auto& supportList = GetDtypeSupportList();
  // 检查self数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查out的dtype
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

  // 检查self的dtype是否可以cast成out的dtype
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
  return true;
}

static bool CheckParamsDataAndShape(const aclTensor *self, int64_t dim, const aclTensor *out) {
  // self、out的维度大于 MAX_DIM_LEN
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

  // self的维度必须大于0
  OP_CHECK_MIN_DIM(self, 1, return false);

  // 入参dim超出了self的shape可选维度范围[-self.dim，self.dim-1]
  int64_t selfDim = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (dim < -selfDim || dim >= selfDim) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld).",
            -selfDim, (selfDim - 1), dim);
    return false;
  }

  // 获取dim的非负数维度值selfDim=3时(-3,-2,-1) -> (0,1,2)
  int64_t positiveDim = dim;
  if (dim < 0) {
    positiveDim += selfDim;
  }

  // 入参self根据指定的dim所对应的维度不能整除2
  int64_t splitShape = self->GetViewShape().GetDim(positiveDim);
  if (splitShape % SPLIT_NUM != 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Halving dimension must be even, but dimension %ld is size %ld.", dim, splitShape);
    return false;
  }

  // out的shape不等于self根据dim拆分后的shape
  op::Shape outShapeExpect = self->GetViewShape();
  int64_t outShapeExpectForDim = outShapeExpect.GetDim(static_cast<size_t>(positiveDim)) / SPLIT_NUM;
  outShapeExpect.SetDim(static_cast<size_t>(positiveDim), outShapeExpectForDim);
  if (outShapeExpect != out->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of out must be %s, but get %s.",
            op::ToString(outShapeExpect).GetString(), op::ToString(out->GetViewShape()).GetString());
    return false;
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t dim, const aclTensor *out) {
  // 1.检查入参参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入数据的有效性
  CHECK_RET(CheckParamsDataAndShape(self, dim, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGluGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnGlu, DFX_IN(self, dim), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 入参参数检查
  auto ret = CheckParams(self, dim, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果为空tensor，则直接返回空
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 入参self根据指定的dim所对应的维度除2,获取SplitV需要的splitSize
  int64_t selfDim = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  int64_t positiveDim = dim;
  if (dim < 0) {
    positiveDim += selfDim;
  }
  int64_t splitShape = self->GetViewShape().GetDim(positiveDim) / SPLIT_NUM;
  int64_t splitSizeValue[] = {splitShape, splitShape};
  aclIntArray* splitSize = uniqueExecutor.get()->AllocIntArray(splitSizeValue, SPLIT_NUM);

  // 将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用SplitV算子
  auto splitResult = l0op::SplitV(selfContiguous, splitSize, positiveDim, uniqueExecutor.get());
  CHECK_RET(splitResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (splitResult->Size() != static_cast<size_t>(SPLIT_NUM)) {
    OP_LOGE(ACLNN_ERR_INNER, "The result of SplitV must be equal 2, but get %zu.", splitResult->Size());
    return ACLNN_ERR_INNER;
  }

  auto splitFirst = (*splitResult)[0];
  auto splitSecond = (*splitResult)[1];

  // 调用Sigmoid算子Kernel,Inplace方式减少空间占用
  splitSecond = l0op::Sigmoid(splitSecond, uniqueExecutor.get());
  CHECK_RET(splitSecond != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Mul算子Kernel,Inplace方式减少空间占用
  splitFirst = l0op::Mul(splitFirst, splitSecond, uniqueExecutor.get());
  CHECK_RET(splitFirst != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果self和out数据类型不一致，需要将结果cast成out的dtype
  auto splitFirstCasted = splitFirst;
  if (self->GetDataType() != out->GetDataType()) {
    splitFirstCasted = l0op::Cast(splitFirst, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(splitFirstCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(splitFirstCasted, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGlu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGlu);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
