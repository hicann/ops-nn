// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

#include "max_pool2d_with_mask_backward.h"
#include "opdev/data_type_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/transdata.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(MaxPool2dWithMaskBackward);

static const std::initializer_list<op::DataType> NULL_DTYPE_SUPPORT_LIST = {};
static const std::initializer_list<op::DataType> GRAD_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> GRAD_DTYPE_SUPPORT_310P_LIST = {op::DataType::DT_FLOAT16,
                                                                                 op::DataType::DT_FLOAT};
static const size_t DHW_DIMS = 3;

static const inline std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_2201: {
            return GRAD_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_2002: {
            return GRAD_DTYPE_SUPPORT_310P_LIST;
        }
        case NpuArch::DAV_1001: {
            return NULL_DTYPE_SUPPORT_LIST;
        }
        default: {
            return NULL_DTYPE_SUPPORT_LIST;
        }
    }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* gradOutput)
{
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    if (!CheckType(gradOutput->GetDataType(), dtypeSupportList)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput dtype not supported. support list is %s",
                op::ToString(dtypeSupportList).GetString());
        return false;
    }
    return true;
}

// A short private-kernel tensor still needs one valid 32-byte DMA block.
// Allocate that block explicitly; the view preserves the public logical shape.
static aclTensor* Alloc310PTensor(const aclTensor* tensor, aclOpExecutor* executor)
{
    const auto& shape = tensor->GetViewShape();
    const int64_t block = tensor->GetDataType() == op::DataType::DT_FLOAT16 ? 16 : 8;
    if (shape.GetShapeSize() >= block) {
        return executor->AllocTensor(shape, tensor->GetDataType(), op::Format::FORMAT_NCDHW);
    }
    auto backing = executor->AllocTensor(op::Shape{block}, tensor->GetDataType());
    CHECK_RET(backing != nullptr, nullptr);
    auto view = executor->CreateView(backing, shape, 0);
    CHECK_RET(view != nullptr, nullptr);
    return const_cast<aclTensor*>(l0op::ReFormat(view, op::Format::FORMAT_NCDHW));
}

static const aclTensor* PadShort310PInput(const aclTensor* input, aclOpExecutor* executor)
{
    const int64_t block = input->GetDataType() == op::DataType::DT_FLOAT16 ? 16 : 8;
    if (input->GetViewShape().GetShapeSize() >= block) {
        return input;
    }
    const auto shape = input->GetViewShape();
    const int64_t count = shape.GetShapeSize();
    auto flatView = executor->CreateView(input, op::Shape{count}, 0);
    CHECK_RET(flatView != nullptr, nullptr);
    auto flat = l0op::ReFormat(flatView, op::Format::FORMAT_ND);
    CHECK_RET(flat != nullptr, nullptr);
    const int64_t paddingValues[] = {0, block - count};
    auto paddingFlat = executor->ConvertToTensor(paddingValues, 2, op::DataType::DT_INT64);
    CHECK_RET(paddingFlat != nullptr, nullptr);
    auto padding = executor->CreateView(paddingFlat, op::Shape{1, 2}, 0);
    CHECK_RET(padding != nullptr, nullptr);
    auto backing = l0op::Pad(flat, padding, executor);
    CHECK_RET(backing != nullptr, nullptr);
    auto logical = executor->CreateView(backing, shape, 0);
    CHECK_RET(logical != nullptr, nullptr);
    return l0op::ReFormat(logical, op::Format::FORMAT_NCDHW);
}

const inline aclTensor* MaxPool2dWithMaskBackwardAiCore(const aclTensor* gradOutput, const aclTensor* self,
                                                        const aclTensor* indices, const aclIntArray* kernelSize,
                                                        const aclIntArray* stride, const aclIntArray* padding,
                                                        const aclIntArray* dilation, bool ceilMode,
                                                        aclTensor* gradInput, aclOpExecutor* executor)
{
    L0_DFX(MaxPool2dWithMaskBackwardAiCore, gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode,
           gradInput);
    ADD_TO_LAUNCHER_LIST_AICORE(MaxPool2dWithMaskBackward, OP_INPUT(self, gradOutput, indices), OP_OUTPUT(gradInput),
                                OP_ATTR(kernelSize, stride, padding, dilation, ceilMode));
    return gradInput;
}

const aclTensor* MaxPool2dWithMaskBackward(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
                                           const aclIntArray* kernelSize, const aclIntArray* stride,
                                           const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
                                           aclOpExecutor* executor)
{
    const aclIntArray& kernelRef = *kernelSize;
    const int64_t kernelD = kernelRef[0];
    const int64_t kernelH = (kernelRef.Size() == 1) ? kernelD : kernelRef[1];
    const int64_t kernelW = (kernelRef.Size() == 1) ? kernelD : kernelRef[2];

    const aclIntArray& strideRef = *stride;
    const int64_t strideD = (strideRef.Size() == 0) ? kernelD : strideRef[0];
    const int64_t strideH = (strideRef.Size() == 0) ? kernelH : ((strideRef.Size() == 1) ? strideD : strideRef[1]);
    const int64_t strideW = (strideRef.Size() == 0) ? kernelW : ((strideRef.Size() == 1) ? strideD : strideRef[2]);

    const aclIntArray& dilationRef = *dilation;
    const int64_t dilationD = dilationRef[0];
    const int64_t dilationH = (dilationRef.Size() == 1) ? dilationD : dilationRef[1];
    const int64_t dilationW = (dilationRef.Size() == 1) ? dilationD : dilationRef[2];

    const aclIntArray& paddingRef = *padding;
    const int64_t paddingD = paddingRef[0];
    const int64_t paddingH = (paddingRef.Size() == 1) ? paddingD : paddingRef[1];
    const int64_t paddingW = (paddingRef.Size() == 1) ? paddingD : paddingRef[2];

    FVector<int64_t> kernelSizeData{kernelD, kernelH, kernelW};
    FVector<int64_t> strideSizeData{strideD, strideH, strideW};
    FVector<int64_t> paddingSizeData{paddingD, paddingH, paddingW};
    FVector<int64_t> dilationSizeData{dilationD, dilationH, dilationW};

    aclIntArray* kernelSize3 = executor->AllocIntArray(kernelSizeData.data(), DHW_DIMS);
    aclIntArray* stride3 = executor->AllocIntArray(strideSizeData.data(), DHW_DIMS);
    aclIntArray* padding3 = executor->AllocIntArray(paddingSizeData.data(), DHW_DIMS);
    aclIntArray* dilation3 = executor->AllocIntArray(dilationSizeData.data(), DHW_DIMS);

    const bool is310p = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002;
    auto gradInput = is310p ?
                         Alloc310PTensor(self, executor) :
                         executor->AllocTensor(self->GetViewShape(), self->GetDataType(), op::Format::FORMAT_NCDHW);
    CHECK_RET(gradInput != nullptr, nullptr);
    if (is310p) {
        gradOutput = PadShort310PInput(gradOutput, executor);
        if (kernelH != 1 || kernelW != 1) {
            indices = PadShort310PInput(indices, executor);
        }
        CHECK_RET(gradOutput != nullptr && indices != nullptr, nullptr);
    }

    if (IsAiCoreSupport(gradOutput)) {
        // 调用算子计算
        return MaxPool2dWithMaskBackwardAiCore(gradOutput, self, indices, kernelSize3, stride3, padding3, dilation3,
                                               ceilMode, gradInput, executor);
    } else {
        // 当前没有匹配的aicpu算子
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "no dtype support on AICPU");
        return nullptr;
    }
}
} // namespace l0op
