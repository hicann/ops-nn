/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "max_pool_with_argmax_v1.h"
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
OP_TYPE_REGISTER(MaxPoolWithArgmaxV1);

// 输入为ND场景下，1980不支持任何数据类型
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_310P_LIST = {op::DataType::DT_FLOAT};
// 输入为5HD场景下
static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_910_LIST = {op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_910B_LIST = {op::DataType::DT_FLOAT16,
                                                                                 op::DataType::DT_FLOAT};
static const int DTYPE = 3;
const int64_t G_DIM_C = 1;
const int64_t INPUT_C0 = 16;
const int64_t SHAPE_FACTOR = 32;
const int64_t PAD_FACTOR = 2;
const int64_t SHAPE_H_DIM = 2;
const int64_t SHAPE_W_DIM = 3;
const int64_t CHW_DIM = 3;

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
    case SocVersion::ASCEND310P: {
      return (isMask ? DTYPE_SUPPORT_310P_LIST : DTYPE_SUPPORT_910_LIST);
    }
    default: { return (isMask ? MASK_DTYPE_SUPPORT_910_LIST : DTYPE_SUPPORT_910_LIST); }
  }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self, const bool isMask) {
  const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion(isMask);
  if (!CheckType(self->GetDataType(), dtypeSupportList)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype not supported. support list is %s",
            op::ToString(dtypeSupportList).GetString());
    return false;
  }

  return true;
}

static inline int64_t DivRtn(int64_t x, int64_t y) {
  if (y == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y value cannot be zero.");
    return x;
  }
  int64_t q = x / y;
  int64_t r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    --q;
  }
  return q;
}

static inline int64_t CallMaxShape(const int64_t ksize, const int64_t strides, const int64_t pad, bool ceilMode,
                           const int64_t dimSize) {
    int64_t exact_size = dimSize + PAD_FACTOR * pad - (ksize - 1) - 1 + ((ceilMode == 1) ? (strides - 1) : 0);
    int64_t outMaxShape = DivRtn(exact_size, strides) + 1;
    if (ceilMode) {
      if ((outMaxShape - 1) * strides >= dimSize + pad) {
        outMaxShape = outMaxShape - 1;
      }
    }
    return outMaxShape;
}

const inline std::tuple<aclTensor*, aclTensor*> MaxPoolWithArgmaxV1AiCore(
    const aclTensor* self, const aclIntArray* kernelSize4, const aclIntArray* stride4, const aclIntArray* padding4,
    const aclIntArray* dilation4, bool ceilMode, aclTensor* out, aclTensor* indices, aclOpExecutor* executor) {
  L0_DFX(MaxPoolWithArgmaxV1AiCore, self, kernelSize4, stride4, padding4, dilation4, ceilMode, out, indices);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MaxPoolWithArgmaxV1, OP_INPUT(self), OP_OUTPUT(out, indices),
                                         OP_ATTR(kernelSize4, stride4, padding4, DTYPE, dilation4, ceilMode));
  std::tuple<aclTensor*, aclTensor*> nullptrTuple(nullptr, nullptr);
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
           "MaxPoolWithArgmaxV1AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptrTuple);
  return std::tuple<aclTensor*, aclTensor*>(out, indices);
}

// indices参数直接下传到leve0，解决indices多次类型转换的问题。不支持非连续的indices tensor
const std::tuple<const aclTensor*, const aclTensor*> MaxPoolWithArgmaxV1(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclOpExecutor* executor) {
  const aclIntArray& kernelRef = *kernelSize;
  const int64_t kH = kernelRef[0];
  const int64_t kW = (kernelRef.Size() == 1) ? kH : kernelRef[1];

  const aclIntArray& strideRef = *stride;
  const int64_t dH = (strideRef.Size() == 0) ? kH : strideRef[0];
  const int64_t dW = (strideRef.Size() == 0) ? kW : ((strideRef.Size() == 1) ? dH : strideRef[1]);

  const aclIntArray& paddingRef = *padding;
  const int64_t padH = paddingRef[0];
  const int64_t padW = (paddingRef.Size() == 1) ? padH : paddingRef[1];

  const aclIntArray& dilationRef = *dilation;
  const int64_t dilationH = dilationRef[0];
  const int64_t dilationW = (dilationRef.Size() == 1) ? dilationH : dilationRef[1];

  FVector<int64_t> vecKernelSize{1, kH, kW, 1};
  FVector<int64_t> vecStride{1, dH, dW, 1};
  FVector<int64_t> vecPadding{1, padH, padW, 1};
  FVector<int64_t> vecDilation{1, dilationH, dilationW, 1};

  aclIntArray* kernelSize4 = executor->AllocIntArray(vecKernelSize.data(), 4);
  aclIntArray* stride4 = executor->AllocIntArray(vecStride.data(), 4);
  aclIntArray* padding4 = executor->AllocIntArray(vecPadding.data(), 4);
  aclIntArray* dilation4 = executor->AllocIntArray(vecDilation.data(), 4);

  const int64_t dimNum = self->GetViewShape().GetDimNum();
  const int64_t inputHDim = dimNum == CHW_DIM ? SHAPE_H_DIM - 1 : SHAPE_H_DIM;
  const int64_t inputWDim = dimNum == CHW_DIM ? SHAPE_W_DIM - 1 : SHAPE_W_DIM;

  // 申请out tensor和indices tensor
  const int64_t outHeight = CallMaxShape(kH, dH, padH, ceilMode, self->GetViewShape().GetDim(inputHDim));
  const int64_t outWidth = CallMaxShape(kW, dW, padW, ceilMode, self->GetViewShape().GetDim(inputWDim));
  auto outputShape = self->GetViewShape();
  outputShape.SetDim(inputHDim, outHeight);
  outputShape.SetDim(inputWDim, outWidth);
  auto outputStorageShape = self->GetStorageShape();
  outputStorageShape.SetDim(inputHDim, outHeight);
  outputStorageShape.SetDim(inputWDim, outWidth);
  auto out = executor->AllocTensor(outputStorageShape, outputShape, self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
  if (out == nullptr) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
      return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }

  // 如果为NC1HWC0格式，则indices为out求mask的kernel位置的bit值
  const bool isMask = (op::GetPrimaryFormat(self->GetStorageFormat()) == op::Format::FORMAT_NC1HWC0) ? true : false;
  auto dataTpye = isMask ? op::DataType::DT_UINT16 : op::DataType::DT_INT32;

  const int64_t maskHeight = kH * kW;
  const int64_t shapeFactor = isMask ? (SHAPE_FACTOR / sizeof(uint16_t)): (SHAPE_FACTOR / sizeof(int32_t));
  const int64_t maskWidth = ((outHeight * outWidth + INPUT_C0 - 1) / INPUT_C0 + G_DIM_C) * shapeFactor;
  auto maskShape = self->GetViewShape();
  maskShape.SetDim(inputHDim, maskHeight);
  maskShape.SetDim(inputWDim, maskWidth);
  auto maskStorageShape = self->GetStorageShape();
  maskStorageShape.SetDim(inputHDim, maskHeight);
  maskStorageShape.SetDim(inputWDim, maskWidth);
  auto indices = executor->AllocTensor(maskStorageShape, maskShape, dataTpye, self->GetStorageFormat(), self->GetOriginalFormat());
  if (indices == nullptr) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc indices tensor failed.");
      return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }

  if (IsAiCoreSupport(self, isMask)) {
    // 调用算子计算
    return MaxPoolWithArgmaxV1AiCore(self, kernelSize4, stride4, padding4, dilation4, ceilMode, out, indices, executor);
  }

  // 当前没有匹配的aicpu算子
  OP_LOGE(ACLNN_ERR_PARAM_INVALID, "no dtype not supported on ai cpu");
  return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
}
}  // namespace l0op
