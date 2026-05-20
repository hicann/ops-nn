/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softshrink.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_softshrink.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline bool CheckSocVersionIsSupportBf16(void) {
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

// 检查输入和输出的数据类型是否在算子的支持列表内
static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  // 检查当前平台是否支持BF16数据类型
  bool bf16flag = CheckSocVersionIsSupportBf16();
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || out->GetDataType() == DataType::DT_BF16)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "bfloat16 dtype is unsupported by the current SOC version [%s], self dtype is %s and out dtype is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(out->GetDataType()).GetString());
    return false;
  }

  // 检查输入的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查输出的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

  return true;
}

// 检查输入和输出是否是空指针
static inline bool CheckNotNull(const aclTensor* self, const aclScalar* lambd, const aclTensor* out) {
  // 检查输入是否是空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(lambd, return false);

  // 检查输入是否是空指针
  OP_CHECK_NULL(out, return false);

  return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* out) {
  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  // self的维度必须小于等于8
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static inline bool CheckLambdValue(const aclScalar* lambd) {
  // 检查lambd是否在[0, +∞]范围内
  if (lambd->ToFloat() < 0.0f) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "lambd should be greater or equal to 0, but found to be [%f].", lambd->ToFloat());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* lambd, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, lambd, out), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入的数据的值是否合理
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
  // 4. 检查输入的lambd的值是否大于等于0
  CHECK_RET(CheckLambdValue(lambd), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static aclnnStatus ExecSoftshrinkGetWorkspaceSize(const aclTensor* self, const aclScalar* lambd, aclTensor* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor) {
  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, lambd, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 执行L0算子
  auto softShrinkOutRet = l0op::SoftShrink(selfContiguous, lambd->ToFloat(), uniqueExecutor.get());
  CHECK_RET(softShrinkOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto castOut = l0op::Cast(softShrinkOutRet, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftshrinkGetWorkspaceSize(const aclTensor* self, const aclScalar* lambd, aclTensor* out,
                                            uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnSoftshrink, DFX_IN(self, lambd), DFX_OUT(out));
  return ExecSoftshrinkGetWorkspaceSize(self, lambd, out, workspaceSize, executor);
}

aclnnStatus aclnnSoftshrink(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnSoftshrink);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
