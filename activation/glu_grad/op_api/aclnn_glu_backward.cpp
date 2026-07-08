/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_glu_backward.h"
#include "glu_grad.h"
#include "sigmoid.h"
#include "level0/split_v.h"
#include "level0/mul.h"
#include "level0/sub.h"
#include "level0/concat.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* GLU反向算子的完整计算流程如下:
 *
 * A2路径 (SplitV + Sigmoid + Mul + Sub + ConcatD):
 *                                     self
 *                                      |
 *           gradOut        Contiguous(workspace_8)
 *              \                       |-----------------------|
 *               \                      |             dim        \
 *                \                     |            /  \         \
 *     Contiguous(workspace_0)   SplitV(workspace_1)     SplitV(workspace_1)
 *                         \         /   \                              / \
 *                          \          Sigmoid(workspace_2)            /   \
 *                           \            /        \         /--------/
 *                          Mul(workspace_3)      Mul(workspace_4)   /
 *                                    |  \                \         /
 *                                    |   \           Sub(workspace_5)
 *                                    |    \             /
 *                            dim     |    Mul(workspace_6)
 *                             \      |      /
 *                              \     |     /
 *                            ConcatD(workspace_7)
 *                                    |
 *                                ViewCopy
 *                                    |
 *                                  out
 *
 * A5路径 (GluGrad单kernel):
 *           gradOut              self
 *              |                  |
 *     Contiguous(workspace_0)  Contiguous(workspace_1)
 *              \                /
 *               \              /
 *                GluGrad(workspace_2)
 *                       |
 *                    ViewCopy
 *                       |
 *                      out
 */

constexpr size_t MAX_DIM_LEN = 8;
constexpr int64_t SPLIT_NUM = 2;

static bool CheckNotNull(const aclTensor *gradOut, const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(gradOut, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_DOUBLE};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (Ops::NN::AclnnUtil::IsRegbase(curArch)) {
    return ASCEND950_DTYPE_SUPPORT_LIST;
  } else if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
             GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *self, const aclTensor *out) {
  const auto& supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_MATCH(gradOut, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
  return true;
}

static bool CheckParamsDataAndShape(const aclTensor *gradOut, const aclTensor *self,
                                    int64_t dim, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(gradOut, MAX_DIM_LEN, return false);

  OP_CHECK_MIN_DIM(self, 1, return false);

  int64_t selfDim = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (dim < -selfDim || dim >= selfDim) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld).",
            -selfDim, (selfDim - 1), dim);
    return false;
  }

  int64_t positiveDim = dim;
  if (dim < 0) {
    positiveDim += selfDim;
  }

  int64_t splitShape = self->GetViewShape().GetDim(positiveDim);
  if (splitShape % SPLIT_NUM != 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Halving dimension must be even, but dimension %ld is size %ld.", dim, splitShape);
    return false;
  }

  op::Shape gradOutShapeExpect = self->GetViewShape();
  int64_t gradOutShapeExpectForDim = gradOutShapeExpect.GetDim(static_cast<size_t>(positiveDim)) / SPLIT_NUM;
  gradOutShapeExpect.SetDim(static_cast<size_t>(positiveDim), gradOutShapeExpectForDim);
  if (gradOutShapeExpect != gradOut->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of gradOut must be %s, but got %s.",
            op::ToString(gradOutShapeExpect).GetString(), op::ToString(gradOut->GetViewShape()).GetString());
    return false;
  }

  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOut, const aclTensor *self, int64_t dim, const aclTensor *out) {
  CHECK_RET(CheckNotNull(gradOut, self, out), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckDtypeValid(gradOut, self, out), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckParamsDataAndShape(gradOut, self, dim, out), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static bool IsRegbaseArch() {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  return Ops::NN::AclnnUtil::IsRegbase(curArch);
}

static inline aclIntArray *getDimAndSplitSize(const aclTensor *self, int64_t &positiveDim, int64_t dim,
                                              aclOpExecutor *executor) {
  int64_t selfDim = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (dim < 0) {
    positiveDim += selfDim;
  }
  int64_t splitShape = self->GetViewShape().GetDim(positiveDim) / SPLIT_NUM;
  int64_t splitSizeValue[] = {splitShape, splitShape};
  return executor->AllocIntArray(splitSizeValue, SPLIT_NUM);
}

static aclnnStatus PrepareInputs(const aclTensor *self, const aclTensor *gradOut, bool needCast,
                                 const aclTensor **selfContiguous, const aclTensor **gradOutContiguous,
                                 aclOpExecutor *executor) {
  *selfContiguous = l0op::Contiguous(self, executor);
  CHECK_RET(*selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *gradOutContiguous = l0op::Contiguous(gradOut, executor);
  CHECK_RET(*gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (needCast) {
    *selfContiguous = l0op::Cast(*selfContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(*selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *gradOutContiguous = l0op::Cast(*gradOutContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(*gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus ComputeGluBackward(const aclTensor *selfContiguous, const aclTensor *gradOutContiguous,
                                      aclIntArray *splitSize, int64_t positiveDim,
                                      const aclTensor **gradA, const aclTensor **gradB,
                                      aclOpExecutor *executor) {
  auto splitResult = l0op::SplitV(selfContiguous, splitSize, positiveDim, executor);
  CHECK_RET(splitResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (splitResult->Size() != static_cast<size_t>(SPLIT_NUM)) {
        OP_LOGE(ACLNN_ERR_INNER, "The result of SplitV must be equal 2, but get %zu.", splitResult->Size());
        return ACLNN_ERR_INNER;
    }

    auto splitFirst = (*splitResult)[0];
    auto splitSecond = (*splitResult)[1];
    splitSecond = l0op::Sigmoid(splitSecond, executor);
    CHECK_RET(splitSecond != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto aGrad = l0op::Mul(gradOutContiguous, splitSecond, executor);
  CHECK_RET(aGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sigmoidBA = l0op::Mul(splitFirst, splitSecond, executor);
    CHECK_RET(sigmoidBA != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto subResult = l0op::Sub(splitFirst, sigmoidBA, executor);
    CHECK_RET(subResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *gradB = l0op::Mul(subResult, aGrad, executor);
    CHECK_RET(*gradB != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *gradA = aGrad;
    return ACLNN_SUCCESS;
}

static aclnnStatus AssembleOutput(const aclTensor *gradA, const aclTensor *gradB, int64_t positiveDim,
                                  bool needCast, const aclTensor *out, aclOpExecutor *executor) {
  op::FVector<const aclTensor*> tensorListVector;
  tensorListVector.emplace_back(gradA);
  tensorListVector.emplace_back(gradB);
  auto tensorList = executor->AllocTensorList(tensorListVector.data(), tensorListVector.size());
  auto concatTensor = l0op::ConcatD(tensorList, positiveDim, executor);
  CHECK_RET(concatTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* gluBackwardOut = concatTensor;
    if (needCast) {
        gluBackwardOut = l0op::Cast(concatTensor, op::DataType::DT_FLOAT16, executor);
        CHECK_RET(gluBackwardOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

  auto viewCopyResult = l0op::ViewCopy(gluBackwardOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus ExecGluBackwardA2(const aclTensor *gradOut, const aclTensor *self, int64_t dim,
                                     const aclTensor *out, aclOpExecutor *executor) {
  int64_t positiveDim = dim;
  auto splitSize = getDimAndSplitSize(self, positiveDim, dim, executor);
  bool needCast = (self->GetDataType() == op::DataType::DT_FLOAT16);

  const aclTensor *selfContiguous = nullptr;
  const aclTensor *gradOutContiguous = nullptr;
  auto status = PrepareInputs(self, gradOut, needCast, &selfContiguous, &gradOutContiguous, executor);
  CHECK_RET(status == ACLNN_SUCCESS, status);

  const aclTensor *gradA = nullptr;
  const aclTensor *gradB = nullptr;
  status = ComputeGluBackward(selfContiguous, gradOutContiguous, splitSize, positiveDim,
                              &gradA, &gradB, executor);
  CHECK_RET(status == ACLNN_SUCCESS, status);

  status = AssembleOutput(gradA, gradB, positiveDim, needCast, out, executor);
  CHECK_RET(status == ACLNN_SUCCESS, status);

  return ACLNN_SUCCESS;
}

static aclnnStatus ExecGluBackwardA5(const aclTensor *gradOut, const aclTensor *self, int64_t dim,
                                     const aclTensor *out, aclOpExecutor *executor) {
  auto selfContiguous = l0op::Contiguous(self, executor);
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto gradOutContiguous = l0op::Contiguous(gradOut, executor);
  CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto result = l0op::GluGrad(gradOutContiguous, selfContiguous, dim, executor);
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(result, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGluBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, int64_t dim,
                                             const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnGluBackward, DFX_IN(gradOut, self, dim), DFX_OUT(out));
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = CheckParams(gradOut, self, dim, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  aclnnStatus status;
  if (IsRegbaseArch() && gradOut->GetDataType() != op::DataType::DT_DOUBLE) {
    status = ExecGluBackwardA5(gradOut, self, dim, out, uniqueExecutor.get());
  } else {
    status = ExecGluBackwardA2(gradOut, self, dim, out, uniqueExecutor.get());
  }
  CHECK_RET(status == ACLNN_SUCCESS, status);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGluBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGluBackward);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
