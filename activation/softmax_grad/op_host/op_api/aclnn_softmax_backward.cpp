/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_softmax_backward.h"
#include "softmax_grad.h"
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
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *out) {
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

// CPU支持 DT_BF16/DT_FLOAT/DT_DOUBLE，AIC支持 DT_FLOAT16/DT_FLOAT，AICPU不支持
static const std::initializer_list<op::DataType> dtype_support_list = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const int64_t AXIS_LIMIT = 8;  // 底层算子不支持超过8维

static inline bool CheckSocVersionIsSupportBf16(void) {
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *output) {
    // 检查self的数据类型是否在add算子的支持列表内
    if (!CheckSocVersionIsSupportBf16() && (gradOutput->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "Input dtype of aclnnSoftMaxBackward is not support bfloat16 in current socversion.");
        return false;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, dtype_support_list, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(output, dtype_support_list, return false);

    OP_CHECK_DTYPE_NOT_MATCH(output, gradOutput->GetDataType(), return false);
    return true;
}

static bool CheckDim(const aclTensor *self, int64_t dim) {
    // 检查指定dim是否在self的维度范围内
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    if (selfDimNum == 0) {
        selfDimNum++;
    }
    if (dim >= selfDimNum || dim < -selfDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "provided dim %ld not in the range of input size %ld.", dim, selfDimNum);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *gradOutput, const aclTensor *output, aclTensor *out) {
    OP_CHECK_MAX_DIM(gradOutput, AXIS_LIMIT, return false);

    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, output, return false);

    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, out, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *output, int64_t dim, aclTensor *out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, output, out), ACLNN_ERR_PARAM_NULLPTR);
    // 空tensor直接返回
    if ((gradOutput->IsEmpty()) || (output->IsEmpty())) {
        return ACLNN_SUCCESS;
    }
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查指定dim是否在grad_output的维度范围内
    CHECK_RET(CheckDim(gradOutput, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入shape和输出shape是否一致PReluGrad
    CHECK_RET(CheckShape(gradOutput, output, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, int64_t dim,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnSoftmaxBackward, DFX_IN(gradOutput, output, dim), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, output, dim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 支持空tensor
    if ((gradOutput->IsEmpty()) || (output->IsEmpty())) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入grad_output转换成连续的tensor
    auto grad_output_contiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(grad_output_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto output_contiguous = l0op::Contiguous(output, uniqueExecutor.get());
    CHECK_RET(output_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用SoftmaxGrad算子kernel
    auto op_out = l0op::SoftmaxGrad(grad_output_contiguous, output_contiguous, dim, uniqueExecutor.get());
    CHECK_RET(op_out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto cast_out = l0op::Cast(op_out, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(cast_out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto view_copy_result = l0op::ViewCopy(cast_out, out, uniqueExecutor.get());
    CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftmaxBackward(void *workspace, uint64_t workspaceSize,
                                 aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnSoftmaxBackward);
    OP_LOGD("Entering aclnnSoftmaxBackward");
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
