/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

 #include "max_pool_grad_with_argmax_v3.h"
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
 OP_TYPE_REGISTER(MaxPoolGradWithArgmaxV3);
 
 // 输入为ND场景下，1980不支持任何数据类型
 static const std::initializer_list<op::DataType> DTYPE_SUPPORT_DEFAULT_LIST = {};
 static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_95_LIST = {
     op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
 static const int DTYPE_INT32 = 3;
 static const int DTYPE_INT64 = 9;
 
 static const inline std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion() {
     auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
     switch (socVersion) {
         case SocVersion::ASCEND910_95:
             return DTYPE_SUPPORT_910_95_LIST;
         default:
             return DTYPE_SUPPORT_DEFAULT_LIST;
     }
 }
 
 // 根据芯片类型、dtype判断算子是否支持走aicore
 static inline bool IsAiCoreSupport(const aclTensor *self) {
     const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
     if (!CheckType(self->GetDataType(), dtypeSupportList)) {
         OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype not supported. support list is %s",
                 op::ToString(dtypeSupportList).GetString());
         return false;
     }
 
     return true;
 }
 
 const inline aclTensor* MaxPoolGradWithArgmaxV3AiCore(const aclTensor *gradOutput, const aclTensor *self,
                                                       const aclTensor *indices, const aclIntArray *kernelSize,
                                                       const aclIntArray *stride, const aclIntArray *padding,
                                                       int dtype, const aclIntArray *dilation, bool ceilMode,
                                                       std::string& dataFormat, aclTensor *gradInput,
                                                       aclOpExecutor *executor) {
     L0_DFX(MaxPoolGradWithArgmaxV3AiCore, gradOutput, self, indices, kernelSize,\
            stride, padding, dtype, dilation, ceilMode, dataFormat, gradInput);
 
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
 
     FVector<int64_t> vecKernelSize{kH, kW};
     FVector<int64_t> vecStride{dH, dW};
     FVector<int64_t> vecPadding{padH, padW};
     FVector<int64_t> vecDilation{dilationH, dilationW};
 
     aclIntArray *kernelSize2 = executor->AllocIntArray(vecKernelSize.data(), 2);
     aclIntArray *stride2 = executor->AllocIntArray(vecStride.data(), 2);
     aclIntArray *padding2 = executor->AllocIntArray(vecPadding.data(), 2);
     aclIntArray *dilation2 = executor->AllocIntArray(vecDilation.data(), 2);
 
     ADD_TO_LAUNCHER_LIST_AICORE(MaxPoolGradWithArgmaxV3,
                                 OP_INPUT(self, gradOutput, indices),
                                 OP_OUTPUT(gradInput),
                                 OP_ATTR(kernelSize2, stride2, padding2, dtype, dilation2, ceilMode, dataFormat));

     return gradInput;
 }
 
 
 const aclTensor* MaxPoolGradWithArgmaxV3(const aclTensor *gradOutput, const aclTensor *self,
                                          const aclTensor *indices, const aclIntArray *kernelSize,
                                          const aclIntArray *stride, const aclIntArray *padding,
                                          const op::DataType dtype, const aclIntArray *dilation, bool ceilMode,
                                          std::string& dataFormat, aclOpExecutor *executor) {
     auto gradInput = executor->AllocTensor(self->GetStorageShape(), self->GetOriginalShape(), self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
 
     if (IsAiCoreSupport(self)) {
         // 调用算子计算
         return MaxPoolGradWithArgmaxV3AiCore(gradOutput, self, indices, kernelSize, stride,
                                              padding, dtype, dilation, ceilMode, dataFormat, gradInput, executor);
     } else {
         // 当前没有匹配的aicpu算子
         return nullptr;
     }
 }
 } // l0op