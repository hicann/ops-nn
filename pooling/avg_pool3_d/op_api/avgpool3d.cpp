/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "avgpool3d.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AvgPool3D);
OP_TYPE_REGISTER(AvgPool3DV2);

constexpr int64_t DIM_D = 1;
constexpr int64_t DIM_H = 2;
constexpr int64_t DIM_W = 3;

static const std::initializer_list<DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                               DataType::DT_FLOAT16,
                                                                               DataType::DT_BF16};

static const std::initializer_list<DataType> AICORE_310P_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                DataType::DT_FLOAT16};

static const std::initializer_list<DataType> AICORE_950_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                 DataType::DT_FLOAT16,
                                                                                 DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(DataType inputDtype)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(inputDtype, AICORE_910B_DTYPE_SUPPORT_LIST);
    }

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        return CheckType(inputDtype, AICORE_310P_DTYPE_SUPPORT_LIST);
    }

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        return CheckType(inputDtype, AICORE_950_DTYPE_SUPPORT_LIST);
    }

    return false;
}

// AICORE算子kernel
static inline const aclTensor* AvgPool3DAiCore(
    const aclTensor* input, aclTensor* output, const aclIntArray *kernelSize, const aclIntArray *stride,
    const aclIntArray *pad, const bool ceilMode, const bool countIncludePad, const int64_t divisorOverride,
    const std::string &dataFormat, aclOpExecutor* executor)
{
    L0_DFX(AvgPool3DAiCore, input, output, kernelSize, stride, pad, ceilMode,
        countIncludePad, divisorOverride, dataFormat);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore AvgPool3D算子加入任务队列
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AvgPool3DV2, OP_INPUT(input), OP_OUTPUT(output),
        OP_ATTR(kernelSize, stride, pad, ceilMode, countIncludePad,
        divisorOverride, dataFormat));
        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
            "AvgPool3DV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }else{
        // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore AvgPool3D算子加入任务队列
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AvgPool3D, OP_INPUT(input), OP_OUTPUT(output),
        OP_ATTR(kernelSize, stride, pad, ceilMode, countIncludePad,
        divisorOverride, dataFormat));

        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
            "AvgPool3D ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }

    return output;
}

static inline int64_t AvgPool3DOutputShape(
    const int64_t inputSize, const int64_t kernelSize, const int64_t padL, const int64_t stride, const bool ceilMode) {
    int64_t outputSize = (stride == 0) ? -1 :
                         (inputSize + padL * 2 - kernelSize + (ceilMode ? stride - 1 : 0)) /stride + 1;

    if (ceilMode) {
        if ((outputSize - 1) * stride >= inputSize + padL) {
            --outputSize;
        }
    }
    return outputSize;
}

const aclTensor* AvgPool3D(
    const aclTensor* input, const aclIntArray *kernelSize, const aclIntArray *stride, const aclIntArray *pad,
    const bool ceilMode, const bool countIncludePad, const int64_t divisorOverride, const std::string &dataFormat,
    aclOpExecutor* executor)
{
    int64_t dimD = dataFormat == "NCDHW" ? DIM_D + 1 : DIM_D;
    int64_t dimH = dataFormat == "NCDHW" ? DIM_H + 1 : DIM_H;
    int64_t dimW = dataFormat == "NCDHW" ? DIM_W + 1 : DIM_W;

    const int64_t outDepth = AvgPool3DOutputShape(
        input->GetViewShape().GetDim(dimD), (*kernelSize)[0], (*pad)[0], (*stride)[0], ceilMode);
    const int64_t outHeight = AvgPool3DOutputShape(
        input->GetViewShape().GetDim(dimH), (*kernelSize)[1], (*pad)[1], (*stride)[1], ceilMode);
    const int64_t outWidth = AvgPool3DOutputShape(
        input->GetViewShape().GetDim(dimW), (*kernelSize)[2], (*pad)[2], (*stride)[2], ceilMode); // 2: index

    auto outputShape = input->GetViewShape();
    outputShape.SetDim(dimD, outDepth);
    outputShape.SetDim(dimH, outHeight);
    outputShape.SetDim(dimW, outWidth);
    auto output = executor->AllocTensor(outputShape, input->GetDataType(), input->GetStorageFormat());
    CHECK_RET(output != nullptr, nullptr);

    if (IsAiCoreSupport(input->GetDataType())) {
        return AvgPool3DAiCore(input, output,
            kernelSize, stride, pad, ceilMode, countIncludePad, divisorOverride, dataFormat, executor);
    }

    return nullptr;
}
}  // namespace l0op
