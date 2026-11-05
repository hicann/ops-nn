/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gelu_mul.h"
#include "gelu_mul_l0op.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "base/err_msg.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include <string>

using namespace op;

namespace {
const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16,
                                                            DataType::DT_FLOAT};

// 声明的out shape需与推导值一致：rank相同、除末维外各维相同、末维为input末维的1/2。
// 不一致时拒绝，避免kernel按input推导的元素数写入越界。
bool CheckOutShapeValid(const aclTensor* input, const aclTensor* out)
{
    const auto& inputShape = input->GetViewShape();
    const auto& outShape = out->GetViewShape();
    const size_t inputDimNum = inputShape.GetDimNum();
    if (outShape.GetDimNum() != inputDimNum) {
        return false;
    }
    for (size_t idx = 0; idx + 1 < inputDimNum; idx++) {
        if (outShape.GetDim(idx) != inputShape.GetDim(idx)) {
            return false;
        }
    }
    return inputDimNum == 0 || outShape.GetDim(inputDimNum - 1) == inputShape.GetDim(inputDimNum - 1) / 2;
}

aclnnStatus CheckParams(const aclTensor* input, const aclTensor* out)
{
    OP_CHECK_NULL(input, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, input->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    if (!CheckOutShapeValid(input, out)) {
        const std::string reason = "The declared output shape must keep the same dims as input except the last dim, "
                                   "which must be half of the input's last dim.";
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0010", (std::vector<const char*>{"param_names", "op_name", "incorrect_shapes", "reason"}),
            (std::vector<const char*>{"input,out", "aclnnGeluMul",
                                      (std::string(op::ToString(input->GetViewShape()).GetString()) + "," +
                                       op::ToString(out->GetViewShape()).GetString())
                                          .c_str(),
                                      reason.c_str()}));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnGeluMulGetWorkspaceSize(const aclTensor* input, char* approximateOptional, const aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // approximate为可选属性，空指针按默认值"none"处理，与生成接口行为一致
    const char* approximate = (approximateOptional != nullptr) ? approximateOptional : "none";
    L2_DFX_PHASE_1(aclnnGeluMul, DFX_IN(input, approximate), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(input, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理：无元素可算，返回空执行器不下发kernel与tiling（含末维为0的空输入），与资料"支持空Tensor"一致
    if (input->IsEmpty()) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Contiguous also normalizes the origin/storage shapes of contiguous subviews without copying data.
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const bool needsViewCopy = !IsContiguous(out);
    const aclTensor* kernelOut = nullptr;
    if (needsViewCopy) {
        // The output is write-only; allocate a result without copying the old output values.
        kernelOut = uniqueExecutor->AllocTensor(out->GetViewShape(), out->GetDataType(), Format::FORMAT_ND);
    } else {
        kernelOut = l0op::Contiguous(out, uniqueExecutor.get());
    }
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 桥接层失败即launcher侧tiling失败（维数/末维/approximate等tiling层约束），映射为561002与资料错误码表一致
    auto result = l0op::GeluMul(inputContiguous, approximate, kernelOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_TILING_ERROR);
    if (needsViewCopy) {
        CHECK_RET(l0op::ViewCopy(result, out, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeluMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGeluMul);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
