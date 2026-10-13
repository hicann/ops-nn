/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_pool2d_with_mask.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MaxPool2dWithMask);

aclnnStatus MaxPool2dWithMaskCustom(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
                                    const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
                                    aclTensor* out, aclTensor* indices, aclOpExecutor* executor)
{
    auto expectedOut = executor->AllocTensor(self->GetDataType(), out->GetStorageFormat(), out->GetStorageFormat());
    auto expectedIndices = executor->AllocTensor(DataType::DT_INT8, indices->GetStorageFormat(),
                                                 indices->GetStorageFormat());
    CHECK_RET(expectedOut != nullptr && expectedIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret = INFER_SHAPE(MaxPool2dWithMask, OP_INPUT(self), OP_OUTPUT(expectedOut, expectedIndices),
                           OP_ATTR(kernelSize, stride, padding, dilation, ceilMode));
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    OP_CHECK_SHAPE_NOT_EQUAL(out, expectedOut, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL(indices, expectedIndices, return ACLNN_ERR_PARAM_INVALID);

    // Singleton dimensions can be contiguous while their backing storage is larger.
    // Pass a compact tensor to the kernel so writes never include storage gaps.
    const bool contiguousOutput = IsContiguous(out) && out->GetViewShape() == out->GetStorageShape();
    const aclTensor* kernelOutput = contiguousOutput ? out : expectedOut;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(MaxPool2dWithMask, OP_INPUT(self), OP_OUTPUT(kernelOutput, indices),
                                      OP_ATTR(kernelSize, stride, padding, dilation, ceilMode));
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (!contiguousOutput) {
        CHECK_RET(l0op::ViewCopy(kernelOutput, out, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}
} // namespace l0op

extern "C" aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize,
                                                              const aclIntArray* stride, const aclIntArray* padding,
                                                              const aclIntArray* dilation, bool ceilMode,
                                                              aclTensor* out, aclTensor* indices,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMaxPool2dWithMask, DFX_IN(self, kernelSize, stride, padding, dilation, ceilMode),
                   DFX_OUT(out, indices));
    CHECK_RET(self != nullptr && kernelSize != nullptr && stride != nullptr && padding != nullptr &&
                  dilation != nullptr && out != nullptr && indices != nullptr && workspaceSize != nullptr &&
                  executor != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);
    const std::initializer_list<DataType> baseTypes = {DataType::DT_FLOAT16, DataType::DT_FLOAT};
    const std::initializer_list<DataType> a2Types = {DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};
    const auto& types = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ? a2Types : baseTypes;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, types, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(indices, DataType::DT_INT8, return ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsContiguous(indices), ACLNN_ERR_PARAM_INVALID);
    const size_t rank = self->GetViewShape().GetDimNum();
    CHECK_RET(rank == 3 || rank == 4, ACLNN_ERR_PARAM_INVALID);
    for (const aclTensor* tensor : {self, static_cast<const aclTensor*>(out), static_cast<const aclTensor*>(indices)}) {
        const auto format = tensor->GetStorageFormat();
        CHECK_RET(format == Format::FORMAT_ND || format == Format::FORMAT_NCHW, ACLNN_ERR_PARAM_INVALID);
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const auto ret = l0op::MaxPool2dWithMaskCustom(contiguousSelf, kernelSize, stride, padding, dilation, ceilMode, out,
                                                   indices, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnMaxPool2dWithMask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxPool2dWithMask);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
