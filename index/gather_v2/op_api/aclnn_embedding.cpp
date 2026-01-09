/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file aclnn_embedding.cpp
 * \brief
 */
#include "aclnn_embedding.h"
#include "gather_v2.h"
#include "embedding.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;

static const int64_t EMBEDDING_DIM = 0;

static const int64_t MAX_SUPPORT_DIM = 8;

static const int64_t HIGH_PRECISION = 0;
static const int64_t HIGH_PERFORMANCE = 1;
static const int64_t SUPPORT_OUT_OF_BOUND_INDEX = 2;
static const int64_t BLOCK_SIZE = 32;
static const int64_t WEIGHT_BYTE_BOUNDS = 98304;     //全缓存模板阈值，小于此阈值的默认DSL实现已是最优性能
static const int64_t LAST_DIM_BYTE_BOUNDS = 128;
static const int64_t MAGNIFICATION_BOUNDS = 100;
static const int64_t SIMT_THRES = 2048;
static const int64_t RATIO_THRES = 32;
static const int64_t WEIGHT_DIM_NUM = 2;
static const size_t NO_CONTIGUOUS_SUPPORT_DIM = 2;

static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64,
    op::DataType::DT_INT32, op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_BOOL, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64};
static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64,
    op::DataType::DT_INT32, op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_BOOL, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64, op::DataType::DT_BF16};
static const std::initializer_list<DataType> INDICES_DTYPE_SUPPORT_LIST =
    {op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> EMBEDDING_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,  DataType::DT_INT32,     DataType::DT_INT64,      DataType::DT_FLOAT16, DataType::DT_BF16,
    DataType::DT_INT16,  DataType::DT_UINT16,    DataType::DT_INT8,       DataType::DT_UINT8,   DataType::DT_BOOL,
    DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX32};

static bool CheckNotNull(const aclTensor *weight, const aclTensor *indices, const aclTensor *out) {
  OP_CHECK_NULL(weight, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *weight, const aclTensor *indices, const aclTensor *out) {
  bool is910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                           GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
                           GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95);
  const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST =
    is910BSocVersion ? WEIGHT_DTYPE_SUPPORT_LIST_910B : WEIGHT_DTYPE_SUPPORT_LIST_910;

  // 检查weight的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(weight, WEIGHT_DTYPE_SUPPORT_LIST, return false);

  // 检查out的数据类型是否和weight相等
  OP_CHECK_DTYPE_NOT_SAME(weight, out, return false);

  // 检查indices的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_DTYPE_SUPPORT_LIST, return false);

  return true;
}

static inline bool CheckMaxDimension(const aclTensor *weight, const aclTensor *indices, const aclTensor *out) {
  OP_CHECK_MAX_DIM(weight, MAX_SUPPORT_DIM, return false);
  OP_CHECK_MAX_DIM(indices, MAX_SUPPORT_DIM, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIM, return false);
  return true;
}

static bool CheckDimension(const aclTensor *out, const aclTensor *indices) {
  const auto outShape = out->GetViewShape();
  const auto indicesShape = indices->GetViewShape();
  if (outShape.GetDimNum() != indicesShape.GetDimNum() + 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dim [%zu] should be one larger than indices dim [%zu].",
            outShape.GetDimNum(), indicesShape.GetDimNum());
    return false;
  }
  size_t indicesDimNum = indicesShape.GetDimNum();
  for (size_t i = 0; i < indicesDimNum; i++) {
    if (outShape.GetDim(i) != indicesShape.GetDim(i)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape [%s] is not match with indices shape [%s].",
              op::ToString(out->GetViewShape()).GetString(), op::ToString(indices->GetViewShape()).GetString());
      return false;
    }
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *weight, const aclTensor *indices, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(weight, indices, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(weight, indices, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查最大维度是否超过8
  CHECK_RET(CheckMaxDimension(weight, indices, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查grad和indices的维度匹配关系
  CHECK_RET(CheckDimension(out, indices), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool CheckHighperf(const aclTensor *weight, const aclTensor *indices) {
  bool is910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                           GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
  const int64_t indicesSize = indices->GetViewShape().GetShapeSize();
  const op::Shape weightShape = weight->GetViewShape();
  const int64_t weightSize = weightShape.GetShapeSize();
  const int64_t lastDimIdx = weightShape.GetDimNum() - 1;
  const int64_t weightByteSize = weightSize * op::TypeSize(weight->GetDataType());
  const int64_t lastDimByteSize = weightShape.GetDim(lastDimIdx) * op::TypeSize(weight->GetDataType());
  const int64_t gatherMagnification = indicesSize / weightShape.GetDim(EMBEDDING_DIM);
  OP_LOGD("weightSize = %ld, lastDimByteSize = %ld, indicesSize = %ld.", weightSize, lastDimByteSize, indicesSize);
  if ((EMBEDDING_DIM != lastDimIdx) && (lastDimByteSize >= LAST_DIM_BYTE_BOUNDS) &&
      (lastDimByteSize % BLOCK_SIZE == 0) && (gatherMagnification > MAGNIFICATION_BOUNDS) &&
      is910BSocVersion && (weightByteSize > WEIGHT_BYTE_BOUNDS)) {
    OP_LOGD("gatherMagnification is big enough, choose high_performance mode.");
    return true;
  } else {
    return false;
  }
}

static bool CheckEmbeddingKernel(const aclTensor *weight, const aclTensor *indices) {
  bool checkResult = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95);
  if (!checkResult) {
    return false;
  }
  checkResult = CheckType(weight->GetDataType(), EMBEDDING_AICORE_DTYPE_SUPPORT_LIST);
  if (!checkResult) {
    return false;
  }
  auto weightShape = weight->GetViewShape();
  if (weightShape.GetDimNum() != WEIGHT_DIM_NUM) {
    return false;
  }
  int64_t innerSize = weightShape.GetDim(1);
  int64_t innerSizeByte = weightShape.GetDim(1) * op::TypeSize(weight->GetDataType());
  if (weightShape.GetDim(0) == 0) {
    return false;
  }
  if (innerSizeByte >= SIMT_THRES && indices->GetViewShape().GetShapeSize() >= RATIO_THRES) {
    return false;
  }
  if (weightShape.GetShapeSize() > INT32_MAX || indices->GetViewShape().GetShapeSize() * innerSize > INT32_MAX) {
    return false;
  }
  return true;
}

static bool IsUseNoContiguous(const aclTensor* weight, const aclTensor* indices, const aclTensor* out)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
        return false;
    }
    bool checkResult = CheckType(weight->GetDataType(), EMBEDDING_AICORE_DTYPE_SUPPORT_LIST);
    if (!checkResult) {
        return false;
    }
    auto selfDimNum = weight->GetViewShape().GetDimNum();
    auto indexDimNum = indices->GetViewShape().GetDimNum();
    if (selfDimNum != indexDimNum || indexDimNum != NO_CONTIGUOUS_SUPPORT_DIM) {
        return false;
    }
    if ((!IsContiguous(weight) || !IsContiguous(indices)) && IsContiguous(out)) {
        return true;
    }
    return false;
}

static const aclTensor* CalNoContiguous(const aclTensor* weight, const aclTensor* indices, aclOpExecutor* executor)
{
    const aclTensor* embeddingResult = nullptr;
    aclTensor* newWeight = executor->CreateView(
        weight, weight->GetViewShape(), weight->GetStorageShape(), weight->GetViewStrides(), weight->GetViewOffset());
    aclTensor* newIndices = executor->CreateView(
        indices, indices->GetViewShape(), indices->GetStorageShape(), indices->GetViewStrides(),
        indices->GetViewOffset());
    embeddingResult = l0op::Embedding(newWeight, newIndices, executor);
    CHECK_RET(embeddingResult != nullptr, nullptr);
    return embeddingResult;
}

aclnnStatus aclnnEmbeddingGetWorkspaceSize(const aclTensor *weight, const aclTensor *indices, const aclTensor *out,
                                           uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnEmbedding, DFX_IN(weight, indices), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(weight, indices, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (weight->IsEmpty() || indices->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  const aclTensor* embeddingResult = nullptr;
  if (IsUseNoContiguous(weight, indices, out)) {
    embeddingResult = CalNoContiguous(weight, indices, uniqueExecutor.get());
  } else {
    // weight如果非连续，需要转连续
    auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // indices如果非连续，需要转连续
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子GatherV2进行计算
    if (CheckHighperf(weight, indices)) {
      int64_t implMode = HIGH_PERFORMANCE;
      embeddingResult = l0op::GatherV2WithImplMode(weightContiguous, EMBEDDING_DIM, indicesContiguous, implMode,
                                                  uniqueExecutor.get());
    } else {
      if (CheckEmbeddingKernel(weight, indices)) {
        embeddingResult = l0op::Embedding(weightContiguous, indicesContiguous, uniqueExecutor.get());
      } else {
        embeddingResult = l0op::GatherV2(weightContiguous, EMBEDDING_DIM, indicesContiguous, uniqueExecutor.get());
      }
    }
  }
  CHECK_RET(embeddingResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(embeddingResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnEmbedding(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnEmbedding);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}