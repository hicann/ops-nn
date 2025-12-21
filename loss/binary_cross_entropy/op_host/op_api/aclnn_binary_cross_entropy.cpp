/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_binary_cross_entropy.h"
#include "level0/fill.h"
#include "binary_cross_entropy.h"
#include "level0/ones_like.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static const int64_t NUM_TWO = 2;

static bool CheckNotNull(const aclTensor *self, const aclTensor *target, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(target, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16 };

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16 };

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weight,
    const aclTensor *out) {
  const auto& supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

  if (weight != nullptr && !CheckType(weight->GetDataType(), supportList)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight dtype %s should be in dtype support list [%s].",
            op::ToString(weight->GetDataType()).GetString(), op::ToString(supportList).GetString());
    return false;
  }

  if (weight == nullptr) {
    return true;
  }

  OP_CHECK_DTYPE_NOT_MATCH(self, target->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(self, weight->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

  return true;
}

static bool CheckShapeValid(const aclTensor *self, const aclTensor *target) {
  OP_CHECK_SHAPE_NOT_EQUAL(self, target, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *target,
    const aclTensor *weight, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, target, out), ACLNN_ERR_INNER_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, target, weight, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查tensor的shape是否合理
  CHECK_RET(CheckShapeValid(self, target), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnBinaryCrossEntropyGetWorkspaceSize(const aclTensor *self, const aclTensor *target,
    const aclTensor *weight, int64_t reduction, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnBinaryCrossEntropy, DFX_IN(self, target, weight, reduction), DFX_OUT(out));
  // 参数检查
  auto ret = CheckParams(self, target, weight, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  if ((reduction < 0) || (reduction > NUM_TWO)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reduction only support 0, 1, 2.");
    return ACLNN_ERR_PARAM_INVALID;
  }
  if ((reduction != 0) && (out->GetViewShape().GetDimNum() != 0)) {
    OP_LOGW("Reduction is %ld, out shape must [1].", reduction);
  }

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  const aclTensor *binaryCrossEntropyOut = nullptr;
  if (self->IsEmpty()) {
    // 根据场景填充tensor
    const aclScalar *valueScalar = nullptr;
    if (reduction == Reduction::Mean) {
      valueScalar = uniqueExecutor.get()->AllocScalar(NAN);
    } else {
      valueScalar = uniqueExecutor.get()->AllocScalar(0);
    }
    auto valueTensor = uniqueExecutor.get()->ConvertToTensor(valueScalar, self->GetDataType());
    auto fillShape = op::ToShapeVector(out->GetOriginalShape());
    aclIntArray *shapeArray = (uniqueExecutor.get())->AllocIntArray(fillShape.data(), fillShape.size());
    const aclTensor *dims = uniqueExecutor.get()->ConvertToTensor(fillShape.data(),
        fillShape.size(), op::DataType::DT_INT64);
    binaryCrossEntropyOut = l0op::Fill(dims, valueTensor, shapeArray, uniqueExecutor.get());
  } else {
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // if weight is null, give ones
    const aclTensor *weightContiguous = nullptr;
    if (weight == nullptr) {
      weightContiguous = l0op::OnesLike(selfContiguous, uniqueExecutor.get());
    } else {
      weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    }
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    static const std::string reductionStr[] = {"none", "mean", "sum"};
    binaryCrossEntropyOut = l0op::BinaryCrossEntropy(selfContiguous, targetContiguous,
        weightContiguous, reductionStr[reduction], uniqueExecutor.get());
  }
  CHECK_RET(binaryCrossEntropyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (self->IsEmpty()) {
    out->SetViewShape(binaryCrossEntropyOut->GetViewShape());
  }

  CHECK_RET(CheckReduceOutShape(binaryCrossEntropyOut, out), ACLNN_ERR_PARAM_INVALID);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(binaryCrossEntropyOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBinaryCrossEntropy(void *workspace,
                                    uint64_t workspaceSize,
                                    aclOpExecutor *executor,
                                    aclrtStream stream
                                    ) {
  L2_DFX_PHASE_2(aclnnBinaryCrossEntropy);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif