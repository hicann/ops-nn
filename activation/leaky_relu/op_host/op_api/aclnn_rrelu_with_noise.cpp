/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "aclnn_rrelu_with_noise.h"
#include "level0/zero_op.h"
#include "aclnn_kernels/slice.h"
#include "level0/concat.h"
#include "aclnn_kernels/reshape.h"
#include "level0/less.h"
#include "leaky_relu.h"
#include "level0/select.h"
#include "level0/ones_like.h"
#include "level0/stateless_random_uniform_v2.h"
#include "level0/muls.h"
#include "level0/mul.h"
#include "level0/sub.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/dsa_random_uniform.h"
#include "opdev/platform.h"
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

static const int DIM_CONSTRAINT = 8;

static bool CheckNotNull(const aclTensor *self, const aclTensor *noise, const aclScalar *lower,
                         const aclScalar *upper, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(noise, return false);
  OP_CHECK_NULL(lower, return false);
  OP_CHECK_NULL(upper, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16 };

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> SCALAR_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_DOUBLE};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *noise, const aclScalar *lower,
                            const aclScalar *upper, const aclTensor *out) {
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(noise, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(lower, SCALAR_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(upper, SCALAR_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

  OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(noise, out->GetDataType(), return false);

  return true;
}

static bool CheckFormatValid(const aclTensor *self, const aclTensor *noise, const aclTensor *out) {
  const size_t max_support_dim = 1024;

  OP_CHECK_MAX_DIM(self, max_support_dim, return false);
  OP_CHECK_MAX_DIM(noise, max_support_dim, return false);

  if (self->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor self's format[%s] should be same with output's format[%s].",
    op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    return false;
  }

  if (noise->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor noise's format[%s] should be same with output's format[%s].",
    op::ToString(noise->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    return false;
  }

  return true;
}

static bool CheckShapeValid(const aclTensor *self, const aclTensor *out, bool training, int64_t selfTensorSize,
                            int64_t noiseTensorSize) {
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  if (training == false) {
    return true;
  }

  if (selfTensorSize > noiseTensorSize) {
    OP_LOGE(ACL_ERROR_INVALID_PARAM, "tesnor size of self larger than noise, self: %ld, noise: %ld.",
        selfTensorSize,
        noiseTensorSize);
    return false;
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *noise, const aclScalar *lower,
                               const aclScalar *upper, bool training, const aclTensor *out,
                               int64_t selfTensorSize, int64_t noiseTensorSize) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, noise, lower, upper, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, noise, lower, upper, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self.size 是否小于 noise.size
  CHECK_RET(CheckShapeValid(self, out, training, selfTensorSize, noiseTensorSize), ACLNN_ERR_PARAM_INVALID);

  // 4. Check format
  CHECK_RET(CheckFormatValid(self, noise, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnRReluWithNoiseGetWorkspaceSize(const aclTensor *self, const aclTensor *noise,
    const aclScalar *lower, const aclScalar *upper, bool training, int64_t seed,
    int64_t offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnRReluWithNoise, DFX_IN(self, noise, lower, upper, training, seed, offset), DFX_OUT(out));
  int64_t selfTensorSize = 0;
  int64_t noiseTensorSize = 0;
  if (self != nullptr) {
    selfTensorSize = self->Numel();
  }
  if (noise != nullptr) {
    noiseTensorSize = noise->Numel();
  }
  // 参数检查
  auto ret = CheckParams(self, noise, lower, upper, training, out, selfTensorSize, noiseTensorSize);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 空tensor场景支持情况
  if (self->IsEmpty() || noise->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto noiseContiguous = l0op::Contiguous(noise, uniqueExecutor.get());
  CHECK_RET(noiseContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 8维以上，需要reshape
  const aclTensor *selfReshapeOut = nullptr;
  op::Shape newSelfShape = {-1};
  op::Shape oldSelfShape = self->GetViewShape();
  op::Shape oldNoiseShape = noise->GetViewShape();
  if (selfContiguous->GetViewShape().GetDimNum() > DIM_CONSTRAINT ||
      self->GetViewShape() != noise->GetViewShape()) {
    selfReshapeOut = l0op::Reshape(selfContiguous, newSelfShape, uniqueExecutor.get());
    CHECK_RET(selfReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    selfReshapeOut = selfContiguous;
  }

  const aclTensor *reluOut = nullptr;
  const aclTensor *noiseOut = nullptr;
  if (training == true) {
    auto onesTensor = l0op::OnesLike(selfReshapeOut, uniqueExecutor.get());
    CHECK_RET(onesTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto zerosTensor = l0op::ZerosLike(selfReshapeOut, uniqueExecutor.get());
    CHECK_RET(zerosTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto maskTensor = l0op::Less(selfReshapeOut, zerosTensor, uniqueExecutor.get());
    CHECK_RET(zerosTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // uniform
    const aclTensor* computeOut = nullptr;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
      auto inputShape = op::ToShapeVector(selfReshapeOut->GetViewShape());
      auto inputShapeArray = uniqueExecutor.get()->AllocIntArray(inputShape.data(), inputShape.size());
      CHECK_RET(inputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto low = uniqueExecutor.get()->AllocScalar(lower->ToFloat());
      CHECK_RET(low != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto high = uniqueExecutor.get()->AllocScalar(upper->ToFloat());
      CHECK_RET(high != nullptr, ACLNN_ERR_INNER_NULLPTR);
      computeOut = l0op::DSARandomUniform(inputShapeArray, seed, offset, low, high, uniqueExecutor.get());
    } else {
      int32_t alg = 1;
      auto uniformOut = l0op::StatelessRandomUniformV2(selfReshapeOut, seed, offset, alg, uniqueExecutor.get());
      CHECK_RET(uniformOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto mulsOut = l0op::Muls(uniformOut, lower->ToFloat(), uniqueExecutor.get());
      CHECK_RET(mulsOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto fromTensor = uniqueExecutor.get()->ConvertToTensor(lower, mulsOut->GetDataType());
      auto subOut = l0op::Sub(mulsOut, fromTensor, uniqueExecutor.get());
      CHECK_RET(subOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

      auto mulsOut2 = l0op::Muls(uniformOut, upper->ToFloat(), uniqueExecutor.get());
      CHECK_RET(mulsOut2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

      computeOut = l0op::Sub(mulsOut2, subOut, uniqueExecutor.get());
    }
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto uniformTensor = l0op::Cast(computeOut, onesTensor->GetDataType(), uniqueExecutor.get());
    CHECK_RET(uniformTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selectOut = l0op::SelectV2(maskTensor, uniformTensor, onesTensor, uniqueExecutor.get());
    CHECK_RET(selectOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (self->GetViewShape() != noise->GetViewShape()) {
      // flatten
      auto noiseReshapeOut = l0op::Reshape(noiseContiguous, newSelfShape, uniqueExecutor.get());
      CHECK_RET(noiseReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // slice
      const int64_t offsetData[] = {selfTensorSize};
      aclIntArray* offsets = uniqueExecutor->AllocIntArray(offsetData, 1);
      const int64_t sizeData[] = {noiseTensorSize - selfTensorSize};
      aclIntArray* size = uniqueExecutor->AllocIntArray(sizeData, 1);
      auto noiseSliceOut = l0op::Slice(noiseReshapeOut, offsets, size, uniqueExecutor.get());
      CHECK_RET(noiseSliceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // concat
      op::FVector<const aclTensor *> tensorFVector = {selectOut, noiseSliceOut};
      auto concatList = uniqueExecutor->AllocTensorList(tensorFVector.data(), tensorFVector.size());
      auto concatOut = l0op::ConcatD(concatList, 0, noiseSliceOut->GetDataType(), uniqueExecutor.get());
      CHECK_RET(concatOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

      noiseOut = l0op::Reshape(concatOut, oldNoiseShape, uniqueExecutor.get());
    } else if (selfContiguous->GetViewShape().GetDimNum() > DIM_CONSTRAINT) {
      noiseOut = l0op::Reshape(selectOut, oldSelfShape, uniqueExecutor.get());
    } else {
      noiseOut = selectOut;
    }
    CHECK_RET(noiseOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // noise
    auto viewCopyResult = l0op::ViewCopy(noiseOut, noise, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    reluOut = l0op::Mul(selfReshapeOut, selectOut, uniqueExecutor.get());
  } else {
    auto negativeSlope = (lower->ToFloat() + upper->ToFloat()) / 2;
    reluOut = l0op::LeakyRelu(selfReshapeOut, negativeSlope, uniqueExecutor.get());
  }
  CHECK_RET(reluOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor *reluReshapeOut = nullptr;
  if (reluOut->GetViewShape() != oldSelfShape) {
    reluReshapeOut = l0op::Reshape(reluOut, oldSelfShape, uniqueExecutor.get());
    CHECK_RET(reluReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    reluReshapeOut = reluOut;
  }
  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(reluReshapeOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnRReluWithNoise(void *workspace,
                                uint64_t workspaceSize,
                                aclOpExecutor *executor,
                                const aclrtStream stream)
{
  L2_DFX_PHASE_2(aclnnRReluWithNoise);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceRReluWithNoiseGetWorkspaceSize(const aclTensor *self, const aclTensor *noise,
    const aclScalar *lower, const aclScalar *upper, bool training, int64_t seed,
    int64_t offset, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto out = const_cast<aclTensor*>(self);
    return aclnnRReluWithNoiseGetWorkspaceSize(self, noise, lower, upper, training, seed, offset,
        out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceRReluWithNoise(void *workspace,
                                       uint64_t workspaceSize,
                                       aclOpExecutor *executor,
                                       const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceRReluWithNoise);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif