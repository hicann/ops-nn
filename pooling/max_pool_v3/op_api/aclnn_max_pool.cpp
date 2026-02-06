/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_pool.h"
#include "max_pool_v3.h"
#include "pooling/max_pool3d_with_argmax_v2/op_api/max_pool3d_with_argmax_v2.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

static const int DIMENTION_3 = 3;
static const int DIMENTION_4 = 4;
static const int SIZE_1 = 1;
static const int SIZE_2 = 2;
static const int SIZE_3 = 3;
static const int SIZE_4 = 4;
static const int INDEX_1 = 1;
static const int INDEX_2 = 2;
static const int INDEX_3 = 3;
static const int DILATIONS_HW_DEFAULT = 1;
static const int64_t MAXPOOL3D_KSIZE_LIMIT = 1000;

// 1980仅支持float16
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_910_LIST = {op::DataType::DT_FLOAT16};
// 1971支持float16和float
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_910B_LIST = {op::DataType::DT_FLOAT16,
                                                                                     op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_REGBASE_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_INT32, op::DataType::DT_INT64,
  op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT16};

static bool CheckNotNullPtr(const aclTensor* self, const aclIntArray* kernelShape, const aclIntArray* strides,
                            const aclIntArray* pads, const aclIntArray* dilations, aclTensor* out) {
  // self、kernelShape、strides、pads、dilations、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(kernelShape, return false);
  OP_CHECK_NULL(strides, return false);
  OP_CHECK_NULL(pads, return false);
  OP_CHECK_NULL(dilations, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static const inline std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion() {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  switch (curArch) {
    case NpuArch::DAV_2201: {
      return SELF_OUT_DTYPE_SUPPORT_910B_LIST;
    }
    case NpuArch::DAV_3510: {
      return SELF_OUT_DTYPE_SUPPORT_REGBASE_LIST;
    }
    case NpuArch::DAV_1001: {
      return SELF_OUT_DTYPE_SUPPORT_910_LIST;
    }
    default: { return SELF_OUT_DTYPE_SUPPORT_910_LIST; }
  }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  // 检查self的数据类型是否在支持列表内
  const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();

  OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
  // 检查数据类型是否一致
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out) {
  if (self->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same, self [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    return false;
  }

  // 如果输入格式是私有格式，记录日志，直接报错
  if (op::IsPrivateFormat(self->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCHW or CHW");
    return false;
  }

  // 仅支持NCHW、CHW、4维ND
  if (self->GetViewShape().GetDimNum() != DIMENTION_3 && self->GetViewShape().GetDimNum() != DIMENTION_4) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "3D or 4D tensor expected for inputs");
    return false;
  }

  return true;
}

// tensor size只能是1或者2
static inline bool CheckAttrSize1Or2(const aclIntArray* param) {
  if ((param->Size() != SIZE_1) && (param->Size() != SIZE_2)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "param must be a list of single int, or two ints");
    return false;
  }
  return true;
}

// tensor size只能是0或1或者2
static inline bool CheckAttrSize0Or1Or2(const aclIntArray* param) {
  if ((param->Size() != 0) && (param->Size() != SIZE_1) && (param->Size() != SIZE_2)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "param must be a list of empty, single int, or two ints");
    return false;
  }
  return true;
}

// tensor size只能是0或1或2或者4
static inline bool CheckAttrSize0Or1Or2Or4(const aclIntArray* param) {
  if ((param->Size() != 0) && (param->Size() != SIZE_1) && (param->Size() != SIZE_2) && (param->Size() != SIZE_4)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "param must be a list of empty, single int, or a tuple of two or four ints");
    return false;
  }
  return true;
}

// Integer division rounding to -Infinity
static inline int64_t divRtn(const int64_t x, const int64_t y) {
  int64_t q = x / y;
  int64_t r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    --q;
  };
  return q;
}

// 计算经过MaxPool后的shape的h和w（n,c与input一致，不用计算）
static inline int64_t PoolingOutShape(const int64_t inputSize, const int64_t kernelSize, const int64_t padL,
                                            const int64_t padR, const int64_t stride, const int64_t dilation,
                                            const bool ceilMode) {
  int64_t outputSize =
      divRtn(inputSize + padL + padR - dilation * (kernelSize - 1) - 1 + (ceilMode ? stride - 1 : 0), stride) + 1;

  if (ceilMode) {
    if ((outputSize - 1) * stride >= inputSize + padL) {
      --outputSize;
    }
  }
  return outputSize;
}

static inline const aclIntArray* kernelShapeOps(const aclIntArray* kernelShape, aclOpExecutor* executor) {
  const aclIntArray& kernelRef = *kernelShape;
  const int64_t kH = kernelRef[0];
  const int64_t kW = (kernelRef.Size() == SIZE_1) ? kH : kernelRef[INDEX_1];

  FVector<int64_t> veckernelShape{1, 1, kH, kW};
  return executor->AllocIntArray(veckernelShape.data(), SIZE_4);
}

static inline const aclIntArray* stridesOps(const aclIntArray* strides, aclOpExecutor* executor) {
  const aclIntArray& stridesRef = *strides;
  const int64_t dH = (stridesRef.Size() == 0) ? 1 : stridesRef[0];
  const int64_t dW = (stridesRef.Size() == 0) ? 1 : ((stridesRef.Size() == SIZE_1) ? dH : stridesRef[INDEX_1]);

  FVector<int64_t> vecstrides{1, 1, dH, dW};
  return executor->AllocIntArray(vecstrides.data(), SIZE_4);
}

static inline const aclIntArray* padsOps(const aclIntArray* pads, aclOpExecutor* executor) {
  const aclIntArray& padsRef = *pads;
  int64_t padHL = 0;
  int64_t padHR = 0;
  int64_t padWL = 0;
  int64_t padWR = 0;
  if (padsRef.Size() == SIZE_1) {
    padHL = padsRef[0];
    padHR = padsRef[0];
    padWL = padsRef[0];
    padWR = padsRef[0];
  } else if (padsRef.Size() == SIZE_2) {
    padHL = padsRef[0];
    padHR = padsRef[0];
    padWL = padsRef[INDEX_1];
    padWR = padsRef[INDEX_1];
  } else if (padsRef.Size() == SIZE_4) {
    padHL = padsRef[0];
    padHR = padsRef[INDEX_2];
    padWL = padsRef[INDEX_1];
    padWR = padsRef[INDEX_3];
  }

  FVector<int64_t> vecpads{padHL, padHR, padWL, padWR};
  return executor->AllocIntArray(vecpads.data(), SIZE_4);
}

// 检查参数列表长度和数值是否满足要求
static aclnnStatus CheckParamsLogic(const aclIntArray* kernelShape, const aclIntArray* strides, const int64_t autoPad,
                                    const aclIntArray* pads, const aclIntArray* dilations) {
  OP_CHECK(CheckAttrSize1Or2(kernelShape),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "The size of kernelShape must be one or two, but got %zu.",
                   kernelShape->Size()),
           return ACLNN_ERR_PARAM_INVALID);
  OP_CHECK(CheckAttrSize0Or1Or2(strides),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "The size of strides must be zero, one or two, but got %zu.",
                   strides->Size()),
           return ACLNN_ERR_PARAM_INVALID);
  OP_CHECK(CheckAttrSize0Or1Or2Or4(pads),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "The size of pads must be zero, one, two or four, but got %zu.",
                   pads->Size()),
           return ACLNN_ERR_PARAM_INVALID);
  OP_CHECK(CheckAttrSize0Or1Or2Or4(dilations),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "The size of dilations must be zero, one, two or four, but got %zu.",
                   dilations->Size()),
           return ACLNN_ERR_PARAM_INVALID);

  if (dilations->Size() == SIZE_4) {
    OP_LOGW("The size of dilations is four, which is not recommended, consider using zero, one or two instead.");
  }

  const int64_t autoPadDefault = 0;
  if (autoPad != autoPadDefault) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "value of autoPad must be zero representing NOTSET, but got autoPad:%ld", autoPad);
    return ACLNN_ERR_PARAM_INVALID;
  }

  // dilations目前算子只支持取值1
  const aclIntArray& dilationsRef = *dilations;
  if (dilationsRef.Size() == SIZE_1) {
    const int64_t dilationsHW = dilationsRef[0];
    OP_CHECK((dilationsHW == 1),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation value along each spatial axis only support 1, but got %ld",
                     dilationsHW),
             return ACLNN_ERR_PARAM_INVALID);
  } else if (dilationsRef.Size() == SIZE_2) {
    const int64_t dilationsH = dilationsRef[0];
    const int64_t dilationsW = dilationsRef[INDEX_1];
    OP_CHECK(((dilationsH == 1) && (dilationsW == 1)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation value along each spatial axis only support 1, but got %ld, %ld",
                     dilationsH, dilationsW),
             return ACLNN_ERR_PARAM_INVALID);
  } else if (dilationsRef.Size() == SIZE_4) {
    const int64_t dilationsHL = dilationsRef[0];
    const int64_t dilationsHR = dilationsRef[INDEX_1];
    const int64_t dilationsWL = dilationsRef[INDEX_2];
    const int64_t dilationsWR = dilationsRef[INDEX_3];
    OP_CHECK(((dilationsHL == 1) && (dilationsHR == 1) && (dilationsWL == 1) && (dilationsWR == 1)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "dilation value along each spatial axis only support 1, but got %ld, %ld, %ld, %ld", dilationsHL,
                     dilationsHR, dilationsWL, dilationsWR),
             return ACLNN_ERR_PARAM_INVALID);
  }

  return ACLNN_SUCCESS;
}

static bool CheckOutputShape(const aclTensor* self, const int64_t kH, const int64_t kW, const int64_t padHL,
                                   const int64_t padHR, const int64_t padWL, const int64_t padWR, const int64_t dH,
                                   const int64_t dW, const int64_t dilationHW, const int64_t ceilMode, aclTensor* out) {
  const bool is3d = self->GetViewShape().GetDimNum() == DIMENTION_3;
  const int64_t nBatch = is3d ? 1 : self->GetViewShape().GetDim(0);
  const int64_t nInputPlane = is3d ? self->GetViewShape().GetDim(0) : self->GetViewShape().GetDim(INDEX_1);
  const int64_t height = is3d ? self->GetViewShape().GetDim(INDEX_1) : self->GetViewShape().GetDim(INDEX_2);
  const int64_t width = is3d ? self->GetViewShape().GetDim(INDEX_2) : self->GetViewShape().GetDim(INDEX_3);
  OP_CHECK(((nInputPlane != 0) && (height != 0) && (width != 0)),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Expected tensor with optional 0 dim batch size,\
                     but got nInputPlane:%ld, height:%ld, width:%ld",
                   nInputPlane, height, width),
           return false);

  const int64_t outHeight = PoolingOutShape(height, kH, padHL, padHR, dH, dilationHW, ceilMode);
  const int64_t outWidth = PoolingOutShape(width, kW, padWL, padWR, dW, dilationHW, ceilMode);
  OP_CHECK((outHeight > 0 && outWidth > 0),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Given input size %ldx%ldx%ld, calc out size %ldx%ldx%ld", nInputPlane,
                   height, width, nInputPlane, outHeight, outWidth),
           return false);

  const op::Shape calcOutShape =
      is3d ? op::Shape({nInputPlane, outHeight, outWidth}) : op::Shape({nBatch, nInputPlane, outHeight, outWidth});
  // 判断out的shape与推导出的输出shape是否相等
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, calcOutShape, return false);
  return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* kernelShape, const aclIntArray* strides,
                       const aclIntArray* pads, const int64_t ceilMode, aclTensor* out) {
  // 校验kernel的值
  const aclIntArray& kernelRef = *kernelShape;
  const int64_t kH = kernelRef[0];
  const int64_t kW = (kernelRef.Size() == SIZE_1) ? kH : kernelRef[INDEX_1];
  OP_CHECK(
      ((kH > 0) && (kW > 0)),
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of kernel should be greater than 0, but got kh:%ld, kw:%ld", kH, kW),
      return false);

  // 校验strides的值，strides的数值不能等于0
  const aclIntArray& stridesRef = *strides;
  const int64_t dH = (stridesRef.Size() == 0) ? 1 : stridesRef[0];
  const int64_t dW = (stridesRef.Size() == 0) ? 1 : ((stridesRef.Size() == SIZE_1) ? dH : stridesRef[INDEX_1]);
  OP_CHECK(
      ((dH > 0) && (dW > 0)),
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of strides should be greater than 0, but got dh:%ld, dw:%ld", dH, dW),
      return false);

  // 校验pads的值，pads的数值小于等于kernelShape
  const aclIntArray& padsRef = *pads;
  int64_t padHL = 0;
  int64_t padHR = 0;
  int64_t padWL = 0;
  int64_t padWR = 0;
  if (padsRef.Size() == SIZE_1) {
    padHL = padsRef[0];
    padHR = padsRef[0];
    padWL = padsRef[0];
    padWR = padsRef[0];
  } else if (padsRef.Size() == SIZE_2) {
    padHL = padsRef[0];
    padHR = padsRef[0];
    padWL = padsRef[INDEX_1];
    padWR = padsRef[INDEX_1];
  } else if (padsRef.Size() == SIZE_4) {
    padHL = padsRef[0];
    padHR = padsRef[INDEX_2];
    padWL = padsRef[INDEX_1];
    padWR = padsRef[INDEX_3];
  }
  OP_CHECK((((padHL + padHR) <= kH) && ((padWL + padWR) <= kW)),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "padding for the sum of beginning and ending along each spatial axis, should be smaller than, or "
                   "equal to value of kernel size, but got kh:%ld, kw:%ld, padHL:%ld, padHR:%ld, padWL:%ld, padWR:%ld",
                   kH, kW, padHL, padHR, padWL, padWR),
           return false);
  // 检查out的shape是否符合预期输出
  const bool ret =
      CheckOutputShape(self, kH, kW, padHL, padHR, padWL, padWR, dH, dW, DILATIONS_HW_DEFAULT, ceilMode, out);
  CHECK_RET(ret, false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* kernelShape, const aclIntArray* strides,
                               const int64_t autoPad, const aclIntArray* pads, const aclIntArray* dilations,
                               const int64_t ceilMode, aclTensor* out) {
  // 检查参数是否为空指针
  CHECK_RET(CheckNotNullPtr(self, kernelShape, strides, pads, dilations, out), ACLNN_ERR_PARAM_NULLPTR);

  // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  // 检查参数列表长度和数值是否满足要求
  CHECK_RET(CheckParamsLogic(kernelShape, strides, autoPad, pads, dilations) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

  // 检查输出shape是否合法
  CHECK_RET(CheckShape(self, kernelShape, strides, pads, ceilMode, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static inline const aclTensor* View3Das4D(const aclTensor* input, aclOpExecutor* executor) {
  // NCL -> unsqueeze -> reformat -> NCHW
  // unsqueeze input into 4D
  const int64_t dim = 0;
  auto unsqueezedInput = l0op::UnsqueezeNd(input, dim, executor);
  CHECK_RET(unsqueezedInput != nullptr, nullptr);
  // reformat to NCHW
  auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
  CHECK_RET(reformatInput != nullptr, nullptr);

  return reformatInput;
}

static inline const aclTensor* View4Das3D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor) {
  // NCHW -> squeeze -> reformat -> NCL
  // squeeze out into 3D
  const int64_t dim = 0;
  auto squeezedInput = l0op::SqueezeNd(input, dim, executor);
  CHECK_RET(squeezedInput != nullptr, nullptr);
  // reformat to NCL
  auto reformatInput = l0op::ReFormat(squeezedInput, format);
  CHECK_RET(reformatInput != nullptr, nullptr);

  return reformatInput;
}

static const aclTensor* Selfto5HDProcess(const aclTensor* self, aclOpExecutor* executor) {
  // 非将self的输入格式转换为5HD
  int64_t groups = 1;
  auto selfTrans = l0op::TransDataSpecial(self, Format::FORMAT_NC1HWC0, groups, executor);
  OP_CHECK(selfTrans != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The self TransData fail, return nullptr."),
           return nullptr);

  return selfTrans;
}

static const aclTensor* OuttoNDProcess(const aclTensor* output, aclOpExecutor* executor) {
  // 将输出转为NCHW格式
  int64_t groups = 1;
  auto outTrans = l0op::TransDataSpecial(output, Format::FORMAT_NCHW, groups, executor);
  OP_CHECK(outTrans != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output TransData fail, return nullptr."),
           return nullptr);

  return outTrans;
}

static const aclTensor* ExecMaxPoolV3(const aclTensor* self, const aclTensor* selfContiguous, const aclIntArray* kernelShape,
                                         const aclIntArray* strides, const aclIntArray* pads, const int64_t ceilMode, aclTensor* out, aclOpExecutor* executor) {

  // 如果self是3D，需要扩到4D再调用MaxPoolV3接口
  const bool isSelf3D = self->GetViewShape().GetDimNum() == DIMENTION_3;
  auto selfUnsqueezed = isSelf3D ? View3Das4D(selfContiguous, executor) : selfContiguous;
  CHECK_RET(selfUnsqueezed != nullptr, nullptr);

  const aclTensor* selfProcsResult = selfUnsqueezed;
  bool isDavid = Ops::NN::AclnnUtil::IsRegbase();
  if (!isDavid) {
    // ND格式-> 5HD格式 NHWC/ NCHW
    selfProcsResult = Selfto5HDProcess(selfUnsqueezed, executor);
    CHECK_RET(selfProcsResult != nullptr, nullptr);
  }

  // 调用MaxPoolV3算子，得到计算结果
  op::Format storageFormat = selfUnsqueezed->GetStorageFormat();
  std::string selfUnsqueFormat = op::ToString(storageFormat).GetString();

  auto kernelShape4 = kernelShapeOps(kernelShape, executor);
  auto strides4 = stridesOps(strides, executor);
  auto pads4 = padsOps(pads, executor);

  // ND 格式以NCHW处理
  const bool isSelf4D = selfProcsResult->GetViewShape().GetDimNum() == DIMENTION_4;
  if (isSelf4D && selfProcsResult->GetStorageFormat() == op::Format::FORMAT_ND) {
    selfUnsqueFormat = "NCHW";
  }
  auto maxPoolOutResult =
      l0op::MaxPoolV3(selfProcsResult, kernelShape4, strides4, pads4, selfUnsqueFormat, ceilMode, executor);
  CHECK_RET(maxPoolOutResult != nullptr, nullptr);

  const aclTensor* outProcsResult = maxPoolOutResult;
  if (!isDavid) {
    // 对返回结果进行处理，进行5HD格式转换
    outProcsResult = OuttoNDProcess(maxPoolOutResult, executor);
    CHECK_RET(outProcsResult != nullptr, nullptr);
  }

  const op::Format& dstFormat = out->GetStorageFormat();
  auto outSqueezed = isSelf3D ? View4Das3D(outProcsResult, dstFormat, executor) : outProcsResult;
  CHECK_RET(outSqueezed != nullptr, nullptr);

  return outSqueezed;
}

static inline const aclTensor* View3Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // CHW -> unsqueeze -> reformat -> 1C1HW(NCDHW, N=1, D=1)
    // unsqueeze input into 5D
    FVector<int64_t> dimData{0, 2}; // Unsqueeze at dimension 0, 2
    const aclIntArray* dim2 = executor->AllocIntArray(dimData.data(), 2);
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dim2, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View4Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCHW -> unsqueeze -> reformat -> NC1HW(NCDHW, D=1)
    // unsqueeze input into 5D
    const int64_t dim = 2; // Unsqueeze at dimension 2
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dim, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View5Das4D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // NC1HW(NCDHW, D=1) -> squeeze -> reformat -> NCHW
    // squeeze out into 4D
    const int64_t dim = 2; // Squeeze out dimension 2
    auto squeezedInput = l0op::SqueezeNd(input, dim, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View5Das3D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // 1C1HW(NCDHW, N=1, D=1) -> squeeze -> reformat -> CHW
    // squeeze out into 3D
    FVector<int64_t> dimData{0, 2}; // Squeeze out dimensions 0 and 2
    const aclIntArray* dim2 = executor->AllocIntArray(dimData.data(), 2);
    auto squeezedInput = l0op::SqueezeNd(input, dim2, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static std::tuple<aclIntArray*, aclIntArray*, aclIntArray*, aclIntArray*> ProcessAttrTo3DParams(
    const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, aclOpExecutor* executor)
{
    // 处理 kernel 尺寸
    const aclIntArray& kernelRef = *kernelSize;
    const int64_t kernelH = kernelRef[0];
    const int64_t kernelW = (kernelRef.Size() == 1) ? kernelH : kernelRef[INDEX_1];

    // 处理步长
    const aclIntArray& strideRef = *stride;
    const int64_t strideH = (strideRef.Size() == 0) ? 1 : strideRef[0];
    const int64_t strideW = (strideRef.Size() == 0) ? 1 : ((strideRef.Size() == 1) ? strideH : strideRef[INDEX_1]);

    // 处理填充
    const aclIntArray& paddingRef = *padding;
    const int64_t paddingH = (paddingRef.Size() == 0) ? 0 : paddingRef[0];
    const int64_t paddingW = (paddingRef.Size() == 0) ? 0 : ((paddingRef.Size() == 1) ? paddingH : paddingRef[INDEX_1]);

    // 构建三维参数数组
    FVector<int64_t> kernelSizeData{1, kernelH, kernelW};
    FVector<int64_t> strideSizeData{1, strideH, strideW};
    FVector<int64_t> paddingSizeData{0, paddingH, paddingW};
    FVector<int64_t> dilationSizeData{1, 1, 1};

    // 分配内存并返回
    aclIntArray* kernelSize3D = executor->AllocIntArray(kernelSizeData.data(), SIZE_3);
    aclIntArray* stride3D = executor->AllocIntArray(strideSizeData.data(), SIZE_3);
    aclIntArray* padding3D = executor->AllocIntArray(paddingSizeData.data(), SIZE_3);
    aclIntArray* dilation3D = executor->AllocIntArray(dilationSizeData.data(), SIZE_3);
    return {kernelSize3D, stride3D, padding3D, dilation3D};
}

static inline bool ExecMaxPool3DCapable(const aclIntArray* kernelShape, const aclIntArray* pads) {
  bool padLengthFlag = true;
  const aclIntArray& padsRef = *pads;
  // pads-4D:[H_top、W_left、H_bottom、W_right]
  if (pads->Size() == SIZE_4 && ((padsRef[0] != padsRef[INDEX_2]) || (padsRef[INDEX_1] != padsRef[INDEX_3]))) {
      padLengthFlag = false;
  }
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  bool supportMaxPool3DWithArgmaxV2 = curArch == NpuArch::DAV_2201;

  const aclIntArray& kernelRef = *kernelShape;
  const int64_t kH = kernelRef[0];
  const int64_t kW = (kernelRef.Size() == SIZE_1) ? kH : kernelRef[INDEX_1];
  bool kLengthFlag = kH > MAXPOOL3D_KSIZE_LIMIT || kW > MAXPOOL3D_KSIZE_LIMIT;

  return padLengthFlag && supportMaxPool3DWithArgmaxV2 && kLengthFlag;
}

static const aclTensor* ExecMaxPool3DWithArgmaxV2(const aclTensor* self, const aclTensor* selfContiguous, const aclIntArray* kernelShape,
                                         const aclIntArray* strides, const aclIntArray* pads, const int64_t ceilMode, aclTensor* out, aclOpExecutor* executor) {

    // 如果self是3d，需要扩到5d，再调用MaxPool3DWithArgmaxV2接口
    const bool isSelf3D = self->GetViewShape().GetDimNum() == DIMENTION_3;
    auto selfUnsqueezed = isSelf3D ? View3Das5D(selfContiguous, executor) : View4Das5D(selfContiguous, executor);
    CHECK_RET(selfUnsqueezed != nullptr, nullptr);

    // 更新2D属性适配3D
    auto [kernel3D, stride3D, padding3D, dilation3D] = ProcessAttrTo3DParams(kernelShape, strides, pads, executor);
    CHECK_RET((kernel3D != nullptr) && (stride3D != nullptr) && (padding3D != nullptr) && (dilation3D != nullptr), nullptr);
    // 调用3D接口
    std::string dataFormat = "NCDHW";
    auto [outResult, indicesResult] = l0op::MaxPool3DWithArgmaxV2Ncdhw(
        selfUnsqueezed, kernel3D, stride3D, padding3D, dilation3D, ceilMode, dataFormat, executor);
    CHECK_RET(outResult != nullptr, nullptr);
    CHECK_RET(indicesResult != nullptr, nullptr);
    const op::Format& outFormat = out->GetStorageFormat();
    auto outResultSqueezed = isSelf3D ? View5Das3D(outResult, outFormat, executor) :
                                        View5Das4D(outResult, outFormat, executor);
    CHECK_RET(outResultSqueezed != nullptr, nullptr);

    return outResultSqueezed;
}

aclnnStatus aclnnMaxPoolGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelShape,
                                         const aclIntArray* strides, const int64_t autoPad, const aclIntArray* pads,
                                         const aclIntArray* dilations, const int64_t ceilMode, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnMaxPool, DFX_IN(self, kernelShape, strides, autoPad, pads, dilations, ceilMode), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, kernelShape, strides, autoPad, pads, dilations, ceilMode, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 检查是否是空tensor（算子不支持空tensor）
  if (self->IsEmpty() || out->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* outSqueezed = nullptr;
  if (ExecMaxPool3DCapable(kernelShape, pads)){
    outSqueezed = ExecMaxPool3DWithArgmaxV2(self, selfContiguous, kernelShape, strides, pads, ceilMode, out, uniqueExecutor.get());
  }
  else{
    outSqueezed = ExecMaxPoolV3(self, selfContiguous, kernelShape, strides, pads, ceilMode, out, uniqueExecutor.get());
  }
  CHECK_RET(outSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto outResult = l0op::ViewCopy(outSqueezed, out, uniqueExecutor.get());
  CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxPool(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMaxPool);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
