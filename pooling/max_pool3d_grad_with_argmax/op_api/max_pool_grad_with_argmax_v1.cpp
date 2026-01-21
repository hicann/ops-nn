/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "max_pool_grad_with_argmax_v1.h"
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

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(MaxPoolGradWithArgmaxV1);

// 输入为ND场景下，1980不支持任何数据类型
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {op::DataType::DT_FLOAT};
// 输入为5HD场景下
static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_910_LIST = {op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_910B_LIST = {op::DataType::DT_FLOAT16,
                                                                                 op::DataType::DT_FLOAT};
static const int DTYPE = 3;


static const inline std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion(const bool isMask) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910_93:
    case SocVersion::ASCEND910B: {
      return (isMask ? MASK_DTYPE_SUPPORT_910B_LIST : DTYPE_SUPPORT_910B_LIST);
    }
    case SocVersion::ASCEND910: {
      return (isMask ? MASK_DTYPE_SUPPORT_910_LIST : DTYPE_SUPPORT_910_LIST);
    }
    default: {
      return (isMask ? MASK_DTYPE_SUPPORT_910_LIST : DTYPE_SUPPORT_910_LIST);
    }
  }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *self) {
    // 如果为NC1HWC0格式，则indices为out求mask的kernel位置的bit值
    const bool isMask = (op::GetPrimaryFormat(self->GetStorageFormat()) == op::Format::FORMAT_NC1HWC0) ? true : false;
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion(isMask);
    if (!CheckType(self->GetDataType(), dtypeSupportList)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype not supported. support list is %s",
                op::ToString(dtypeSupportList).GetString());
        return false;
    }

    return true;
}

const inline aclTensor* MaxPoolGradWithArgmaxV1AiCore(const aclTensor *gradOutput, const aclTensor *self,
                                                      const aclTensor *indices, const aclIntArray *kernelSize,
                                                      const aclIntArray *stride, const aclIntArray *padding,
                                                      const aclIntArray *dilation, bool ceilMode,
                                                      aclTensor *gradInput,
                                                      aclOpExecutor *executor) {
    L0_DFX(MaxPoolGradWithArgmaxV1AiCore, gradOutput, self, indices, kernelSize,\
           stride, padding, dilation, ceilMode, gradInput);

    const aclIntArray &kernelRef = *kernelSize;
    const int64_t kH = kernelRef[0];
    const int64_t kW = (kernelRef.Size() == 1) ? kH : kernelRef[1];

    const aclIntArray &strideRef = *stride;
    const int64_t dH = (strideRef.Size() == 0) ? kH : strideRef[0];
    const int64_t dW = (strideRef.Size() == 0) ? kW : ((strideRef.Size() == 1) ? dH : strideRef[1]);

    const aclIntArray &paddingRef = *padding;
    const int64_t padH = paddingRef[0];
    const int64_t padW = (paddingRef.Size() == 1) ? padH : paddingRef[1];

    const aclIntArray &dilationRef = *dilation;
    const int64_t dilationH = dilationRef[0];
    const int64_t dilationW = (dilationRef.Size() == 1) ? dilationH : dilationRef[1];

    FVector<int64_t> vecKernelSize{1, kH, kW, 1};
    FVector<int64_t> vecStride{1, dH, dW, 1};
    FVector<int64_t> vecPadding{1, padH, padW, 1};
    FVector<int64_t> vecDilation{1, dilationH, dilationW, 1};

    aclIntArray *kernelSize4 = executor->AllocIntArray(vecKernelSize.data(), 4);
    aclIntArray *stride4 = executor->AllocIntArray(vecStride.data(), 4);
    aclIntArray *padding4 = executor->AllocIntArray(vecPadding.data(), 4);
    aclIntArray *dilation4 = executor->AllocIntArray(vecDilation.data(), 4);

    ADD_TO_LAUNCHER_LIST_AICORE(MaxPoolGradWithArgmaxV1,
                                OP_INPUT(self, gradOutput, indices),
                                OP_OUTPUT(gradInput),
                                OP_ATTR(kernelSize4, stride4, padding4, DTYPE, dilation4, ceilMode));

    return gradInput;
}


const aclTensor* MaxPoolGradWithArgmaxV1(const aclTensor *gradOutput, const aclTensor *self,
                                         const aclTensor *indices, const aclIntArray *kernelSize,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const aclIntArray *dilation, bool ceilMode,
                                         aclOpExecutor *executor) {
    auto gradInput = executor->AllocTensor(self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
    auto ret = INFER_SHAPE(MaxPoolGradWithArgmaxV1,
                           OP_INPUT(self, gradOutput, indices),
                           OP_OUTPUT(gradInput),
                           OP_ATTR(kernelSize, stride, padding, DTYPE, dilation, ceilMode));
    if (ret != ACLNN_SUCCESS) {
        // 根据输入tensor和属性参数推导得到的out的shape中的轴不能是0
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    if (IsAiCoreSupport(self)) {
        // 调用算子计算
        return MaxPoolGradWithArgmaxV1AiCore(gradOutput, self, indices, kernelSize, stride,
                                             padding, dilation, ceilMode, gradInput, executor);
    } else {
        // 当前没有匹配的aicpu算子
        return nullptr;
    }
}
} // l0op
