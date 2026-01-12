/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_swish_backward.h"

#include <climits>

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "swish_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/muls.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, op::DataType::DT_BF16};

static inline bool CheckSocVersionIsSupportBf16(void)
{
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
    GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(gradInput, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self,
                            const aclScalar* betaOptional, aclTensor* gradInput) {
    // 检查gradOutput的数据类型是否在SiluGrad/SwishGrad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);

    // 检查self的数据类型是否在SiluGrad/SwishGrad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    // 检查betaOptional的数据类型能否转换为FLOAT
    if (betaOptional != nullptr && !CanCast(betaOptional->GetDataType(), DataType::DT_FLOAT)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "betaOptional dtype %s can not cast to float32.",
            ToString(betaOptional->GetDataType()).GetString());
            return false;
    }

    // 检查gradOutput, self和gradInput的dtype是否一致
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, gradInput, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, gradInput, return false);

    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && self->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s is unsupported by the current SOC version [%s].",
        op::ToString(self->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }

    return true;
}

static bool CheckShapeValid(const aclTensor* gradOutput, const aclTensor* self, aclTensor* gradInput) {
    OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gradInput, MAX_SUPPORT_DIMS_NUMS, return false);

    // 检查gradOutput, self和gradInput的shape是否一致
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradOutput, gradInput->GetViewShape(), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(self, gradInput->GetViewShape(), return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self,
                               const aclScalar* betaOptional, aclTensor* gradInput) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, betaOptional, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查双输入shape是否符合要求
    CHECK_RET(CheckShapeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclIntArray* GetTensorShape(const aclTensor* x, aclOpExecutor* executor) {
    auto shape = x->GetViewShape();
    int64_t dimSize = x->GetViewShape().GetDimNum();

    std::vector<int64_t> valuePerm(dimSize);
    for (int i = 0; i < dimSize; i++) {
        valuePerm[i] = shape[i];
    }

    auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
    return perm;
}

static const aclTensor* ReshapeLongTensor(const aclTensor* x, aclOpExecutor* executor, size_t originalDimSize,
                                          aclIntArray* valuePerm = nullptr) {
  size_t dimSize = x->GetViewShape().GetDimNum();
  if (originalDimSize == dimSize && dimSize <= MAX_SUPPORT_DIMS_NUMS) {
    return x;
  }

  auto reshapeSelf = l0op::Reshape(x, valuePerm, executor);
  return reshapeSelf;
}

aclnnStatus aclnnSwishBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                               const aclScalar* betaOptional, aclTensor* gradInput,
                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSwishBackward, DFX_IN(gradOutput, self, betaOptional), DFX_OUT(gradInput));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, betaOptional, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，支持空tensor
    if (gradOutput->IsEmpty() || self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 当输入维度大于8时，reshape成一个长的1维tensor
    auto shapeOri = self->GetViewShape();
    size_t dimSize = self->GetViewShape().GetDimNum();
    auto shapeOriDetial = GetTensorShape(gradOutputContiguous, uniqueExecutor.get());
    auto reshapeGradOutput = gradOutputContiguous;
    auto reshapeSelf = selfContiguous;

    if (dimSize > MAX_SUPPORT_DIMS_NUMS) {
        int64_t AllDimValue = 1;
        for (size_t i = 0; i < dimSize; i++) {
            AllDimValue *= shapeOri[i];
        }
        int64_t AllDim[1] = {AllDimValue};
        auto shape1d = (uniqueExecutor)->AllocIntArray(AllDim, 1);
        reshapeGradOutput = ReshapeLongTensor(gradOutputContiguous, uniqueExecutor.get(), dimSize, shape1d);
        reshapeSelf = ReshapeLongTensor(selfContiguous, uniqueExecutor.get(), dimSize, shape1d);
    }

    // 将beta转换为float
    float scale = 1.0f;
    if (betaOptional != nullptr) {
        scale = betaOptional->ToFloat();
    }

    // 调用SwishGrad算子Kernel
    auto swishGradOpOut = l0op::SwishGrad(reshapeGradOutput, reshapeSelf, scale, uniqueExecutor.get());
    CHECK_RET(swishGradOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 把长1维tensor reshape回原来的维度
    auto reshapeSwishGradOpOut = swishGradOpOut;
    if (dimSize > MAX_SUPPORT_DIMS_NUMS) {
      reshapeSwishGradOpOut = ReshapeLongTensor(swishGradOpOut, uniqueExecutor.get(), dimSize, shapeOriDetial);
    }

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(reshapeSwishGradOpOut, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSwishBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSwishBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
