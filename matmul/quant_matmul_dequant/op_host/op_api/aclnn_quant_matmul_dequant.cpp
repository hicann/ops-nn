/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_dequant.h"
#include "quant_matmul_dequant.h"
#include "level0/padv3.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "op_math_util.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr uint64_t ND_DIMNUM = 2;
static constexpr uint64_t NZ_DIMNUM = 4;
static constexpr uint64_t FRACTAL_K_INT8 = 32;
static constexpr uint64_t FRACTAL_N_INT8 = 16;
static constexpr uint64_t N_IDX = 0;
static constexpr uint64_t K_IDX = 1;
static constexpr uint64_t X_IDX = 0;
static constexpr uint64_t WEIGHT_IDX = 1;
static constexpr uint64_t WEIGHT_SCALE_IDX = 2;
static constexpr uint64_t X_SCALE_IDX = 3;
static constexpr uint64_t SMOOTH_SCALE_IDX = 4;
static constexpr int64_t PAD_VEC_QMMDQ[4] = {0, 0, 0, 16};
static constexpr uint64_t PAD_VEC_SIZE_QMMDQ = 4;
static constexpr int64_t PERM_VEC_QMMDQ[4] = {2, 0, 1, 3};
static constexpr uint64_t PERM_VEC_SIZE_QMMDQ = 4;
static const std::string PAD_MODE_QMMDQ = "constant";

static const inline std::vector<std::vector<DataType>>& GetSupportDtypeList(SocVersion socVersion) {
  static const std::vector<std::vector<DataType>> emptyDtypes = {};
  static const std::map<SocVersion, std::vector<std::vector<DataType>>> dataTypeSupportedMap = {{ 
    SocVersion::ASCEND310P,{
      {DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_FLOAT, DataType::DT_FLOAT, DataType::DT_FLOAT16},
      {DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_INT64, DataType::DT_FLOAT16, DataType::DT_FLOAT16}
    }},
  };

  auto found = dataTypeSupportedMap.find(socVersion);
  if (found == dataTypeSupportedMap.end()) {
    return emptyDtypes;
  }

  return found->second;
}

static aclnnStatus CheckParamsNullOrNot(const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale,
                               const aclTensor* biasOptional, const aclTensor* xOffsetOptional,
                               const aclTensor* out) {
  OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
  OP_CHECK_NULL(weight, return ACLNN_ERR_PARAM_NULLPTR);
  OP_CHECK_NULL(weightScale, return ACLNN_ERR_PARAM_NULLPTR);
  OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

  if(biasOptional != nullptr || xOffsetOptional != nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "not support biasOptional and xOffsetOptional for now, should be nullptr");
    return ACLNN_ERR_PARAM_NULLPTR;
  }

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsDateType(const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale,
                               const aclTensor* xScaleOptional, const aclTensor* smoothScaleOptional,
                               const aclTensor* out) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();

  OP_CHECK_DTYPE_NOT_SAME(out, x, return ACLNN_ERR_PARAM_INVALID);

  const auto& DTYPE_SUPPORT_LIST_CURRENT = GetSupportDtypeList(socVersion);
  if (DTYPE_SUPPORT_LIST_CURRENT.empty()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
    return ACLNN_ERR_PARAM_INVALID;
  }

  for (size_t i = 0; i < DTYPE_SUPPORT_LIST_CURRENT.size(); i++) {
    // 检查输入的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_MATCH(x, DTYPE_SUPPORT_LIST_CURRENT[i][X_IDX], continue);
    OP_CHECK_DTYPE_NOT_MATCH(weight, DTYPE_SUPPORT_LIST_CURRENT[i][WEIGHT_IDX], continue);
    OP_CHECK_DTYPE_NOT_MATCH(weightScale, DTYPE_SUPPORT_LIST_CURRENT[i][WEIGHT_SCALE_IDX], continue);
    if (xScaleOptional != nullptr) {
      OP_CHECK_DTYPE_NOT_MATCH(xScaleOptional, DTYPE_SUPPORT_LIST_CURRENT[i][X_SCALE_IDX], continue);
    }
    if (smoothScaleOptional != nullptr) {
      OP_CHECK_DTYPE_NOT_MATCH(smoothScaleOptional, DTYPE_SUPPORT_LIST_CURRENT[i][SMOOTH_SCALE_IDX], continue);
    }
    return ACLNN_SUCCESS;
  }
  OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckParamsDateType failed, please check log");
  return ACLNN_ERR_PARAM_INVALID;
}

static op::Shape GetWeightNzShape(const aclTensor *weight)
{
    uint64_t k = weight->GetViewShape().GetDim(K_IDX);
    uint64_t k1 = CeilDiv(k, FRACTAL_K_INT8);
    uint64_t n = weight->GetViewShape().GetDim(N_IDX);
    uint64_t n1 = CeilDiv(n, FRACTAL_N_INT8);
    
    op::Shape weightNzShape;
    weightNzShape.AppendDim(n1);
    weightNzShape.AppendDim(FRACTAL_N_INT8);
    weightNzShape.AppendDim(k1);
    weightNzShape.AppendDim(FRACTAL_K_INT8);
    return weightNzShape;
}
}

aclnnStatus aclnnQuantMatmulDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                    const aclTensor *weightScale, const aclTensor *biasOptional,
                                                    const aclTensor *xScaleOptional, const aclTensor *xOffsetOptional, const aclTensor *smoothScaleOptional,
                                                    char *xQuantMode, bool transposeWeight, const aclTensor *out,
                                                    uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnQuantMatmulDequant, DFX_IN(x, weight, weightScale, biasOptional, xScaleOptional, xOffsetOptional, smoothScaleOptional, xQuantMode, transposeWeight), DFX_OUT(out));
  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if(!transposeWeight){
    return ACLNN_ERR_PARAM_INVALID;
  }

  // 参数检查
  auto ret = CheckParamsNullOrNot(x, weight, weightScale, biasOptional, xOffsetOptional, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  ret = CheckParamsDateType(x, weight, weightScale, xScaleOptional, smoothScaleOptional, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // QuantMatmulDequant算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (x->IsEmpty() || weight->IsEmpty() || weightScale->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = static_cast<uint64_t>(0);
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入转换成连续的tensor
  auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
  auto weightScaleContiguous = l0op::Contiguous(weightScale, uniqueExecutor.get());
  auto xScaleContiguous = xScaleOptional;
  if(xScaleOptional!=nullptr) {
    xScaleContiguous = l0op::Contiguous(xScaleOptional, uniqueExecutor.get());
  }
  auto smoothScaleContiguous = smoothScaleOptional;
  if(smoothScaleContiguous!=nullptr) {
    smoothScaleContiguous = l0op::Contiguous(smoothScaleOptional, uniqueExecutor.get());
  }
  if (!IsContiguous(out) || !IsContiguous(weight)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight and out must be contiguous, please check");
    return ACLNN_ERR_PARAM_INVALID;
  }

  //
  uint64_t weightDimNum = weight->GetViewShape().GetDimNum();
  if(weightDimNum == ND_DIMNUM && weight->GetStorageFormat()==op::Format::FORMAT_ND) {
    if(weight->GetViewShape().GetDim(N_IDX) % FRACTAL_N_INT8 != 0 || weight->GetViewShape().GetDim(K_IDX) % FRACTAL_N_INT8 != 0){
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight shape (N, K) must both align to 16, please check");
      return ACLNN_ERR_PARAM_INVALID;
    }
    if(weight->GetViewShape().GetDim(K_IDX) % FRACTAL_K_INT8 != 0){
      auto padArray = uniqueExecutor.get()->AllocIntArray(PAD_VEC_QMMDQ, PAD_VEC_SIZE_QMMDQ);
      auto padTensor = uniqueExecutor.get()->ConvertToTensor(padArray, DataType::DT_INT64);
      auto valueTensor = uniqueExecutor.get()->ConvertToTensor(uniqueExecutor.get()->AllocScalar(0), DataType::DT_INT8);
      auto weight_ = uniqueExecutor.get()->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
      weight_->SetStorageShape(weight->GetViewShape());
      weight = l0op::PadV3(weight_, padTensor, valueTensor, PAD_MODE_QMMDQ, true, uniqueExecutor.get());
    }
    op::Shape weightNzShape = GetWeightNzShape(weight);
    auto weight_ = uniqueExecutor.get()->CreateView(weight, weightNzShape, weight->GetViewOffset());
    auto perm = uniqueExecutor.get()->AllocIntArray(PERM_VEC_QMMDQ, PERM_VEC_SIZE_QMMDQ);
    weight = l0op::Transpose(weight_, perm, uniqueExecutor.get());
    weight_ = uniqueExecutor.get()->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
    weight_->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    weight = weight_;
  } else if(weightDimNum != NZ_DIMNUM || weight->GetStorageFormat()!=op::Format::FORMAT_FRACTAL_NZ){
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight is not in 2-dim shape (N, K) or 4-dim (K//32, N//16, 16, 32), please check");
    return ACLNN_ERR_PARAM_INVALID;
  }

  auto out_ = uniqueExecutor.get()->CreateView(out, out->GetViewShape(), out->GetViewOffset());
  out_->SetStorageShape(out->GetViewShape());
  out = out_;

  auto matmulRet = l0op::QuantMatmulDequant(xContiguous, weight, weightScaleContiguous, biasOptional, xScaleContiguous,
                                            xOffsetOptional, smoothScaleContiguous, xQuantMode, transposeWeight, out,
                                            uniqueExecutor.get());

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulDequant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnQuantMatmulDequant);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif