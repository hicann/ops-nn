/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_linear_online_max_sum.h"
#include "fused_linear_online_max_sum.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr size_t EXPECTED_DIM_ONE = 1;
static constexpr size_t EXPECTED_DIM_TWO = 2;
static constexpr size_t DIM_NUM_ZERO = 0;
static constexpr size_t DIM_NUM_ONE = 1;
static constexpr int64_t BIT_PER_UINT8 = 8;
static constexpr int64_t MAX_INPUT_DIM_ONE_SIZE = 65534;
static constexpr size_t LOGITS_MAX_LOCAL_OUT_IDX = 0;
static constexpr size_t SUM_EXP_LOGITS_LOCAL_OUT_IDX = 1;
static constexpr size_t PREDICTED_LOGITS_LOCAL_OUT_IDX = 2;
static constexpr size_t TARGET_MASK_OUT_IDX = 3;
static constexpr size_t MASKED_TARGET_OUT_IDX = 4;
static constexpr size_t VOCAB_PARALLEL_LOGTIS_OUT_OPTIONAL_IDX = 5;

static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                             DataType::DT_BF16};

static const std::initializer_list<op::DataType> INT_DTYPE_SUPPORT_LIST = {DataType::DT_INT32,
                                                                           DataType::DT_INT64};

struct FusedLinearOnlineMaxSumInputTensor {
  const aclTensor* input;
  const aclTensor* weight;
  const aclTensor* target;
};

struct FusedLinearOnlineMaxSumOutputTensor {
  aclTensor* logitsMaxLocalOut;
  aclTensor* sumExpLogitsLocalOut;
  aclTensor* predictedLogitsLocalOut;
  aclTensor* targetMaskOut;
  aclTensor* maskedTargetOut;
  aclTensor* vocabParallelLogitsOutOptional;
};

inline static bool CheckNotNull(FusedLinearOnlineMaxSumInputTensor& inputTensors,
                                FusedLinearOnlineMaxSumOutputTensor& outputTensors)
{
  OP_CHECK_NULL(inputTensors.input, return false);
  OP_CHECK_NULL(inputTensors.weight, return false);
  OP_CHECK_NULL(inputTensors.target, return false);

  OP_CHECK_NULL(outputTensors.logitsMaxLocalOut, return false);
  OP_CHECK_NULL(outputTensors.sumExpLogitsLocalOut, return false);
  OP_CHECK_NULL(outputTensors.predictedLogitsLocalOut, return false);
  OP_CHECK_NULL(outputTensors.targetMaskOut, return false);
  OP_CHECK_NULL(outputTensors.maskedTargetOut, return false);
  return true;
}

inline static bool CheckDtypeValid(FusedLinearOnlineMaxSumInputTensor& inputTensors,
                                   FusedLinearOnlineMaxSumOutputTensor& outputTensors)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(inputTensors.input, FLOAT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(inputTensors.weight, FLOAT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(inputTensors.target, INT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SAME(inputTensors.input, inputTensors.weight, return false);

  OP_CHECK_DTYPE_NOT_MATCH(outputTensors.logitsMaxLocalOut, DataType::DT_FLOAT, return false);
  OP_CHECK_DTYPE_NOT_MATCH(outputTensors.sumExpLogitsLocalOut, DataType::DT_FLOAT, return false);
  OP_CHECK_DTYPE_NOT_MATCH(outputTensors.predictedLogitsLocalOut, DataType::DT_FLOAT, return false);
  OP_CHECK_DTYPE_NOT_MATCH(outputTensors.targetMaskOut, DataType::DT_UINT8, return false);
  OP_CHECK_DTYPE_NOT_SAME(outputTensors.maskedTargetOut, inputTensors.target, return false);
  if (outputTensors.vocabParallelLogitsOutOptional != nullptr) {
    OP_CHECK_DTYPE_NOT_SAME(outputTensors.vocabParallelLogitsOutOptional, inputTensors.input, return false);
  }

  return true;
}

inline static bool CheckShapeValid(FusedLinearOnlineMaxSumInputTensor& inputTensors,
                                   FusedLinearOnlineMaxSumOutputTensor& outputTensors)
{
  OP_CHECK_WRONG_DIMENSION(inputTensors.input, EXPECTED_DIM_TWO, return false);
  OP_CHECK_WRONG_DIMENSION(inputTensors.weight, EXPECTED_DIM_TWO, return false);
  OP_CHECK_WRONG_DIMENSION(inputTensors.target, EXPECTED_DIM_ONE, return false);

  OP_CHECK_WRONG_DIMENSION(outputTensors.logitsMaxLocalOut, EXPECTED_DIM_ONE, return false);
  OP_CHECK_WRONG_DIMENSION(outputTensors.sumExpLogitsLocalOut, EXPECTED_DIM_ONE, return false);
  OP_CHECK_WRONG_DIMENSION(outputTensors.predictedLogitsLocalOut, EXPECTED_DIM_ONE, return false);
  OP_CHECK_WRONG_DIMENSION(outputTensors.targetMaskOut, EXPECTED_DIM_ONE, return false);
  OP_CHECK_WRONG_DIMENSION(outputTensors.maskedTargetOut, EXPECTED_DIM_ONE, return false);

  op::Shape inputShape = inputTensors.input->GetViewShape();
  op::Shape weightShape = inputTensors.weight->GetViewShape();

  auto m = inputShape.GetDim(DIM_NUM_ZERO);
  auto n = weightShape.GetDim(DIM_NUM_ZERO);
  if (n == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight.size(0) should be greater than 0.");
    return false;
  }

  if (inputShape.GetDim(DIM_NUM_ONE) > MAX_INPUT_DIM_ONE_SIZE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input.size(1) should be less than or equal to 65534, but got %ld",
            inputShape.GetDim(DIM_NUM_ONE));
    return false;
  }

  if (inputShape.GetDim(DIM_NUM_ONE) != weightShape.GetDim(DIM_NUM_ONE)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight.size(1) should be equal to input.size(1), but got weight %s, input %s",
            op::ToString(inputShape).GetString(), op::ToString(weightShape).GetString());
    return false;
  }

  op::Shape mShape = {m};
  op::Shape targetMaskOutShape = {(m + BIT_PER_UINT8 - 1) / BIT_PER_UINT8};
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensors.target, mShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensors.logitsMaxLocalOut, mShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensors.sumExpLogitsLocalOut, mShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensors.predictedLogitsLocalOut, mShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensors.targetMaskOut, targetMaskOutShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensors.maskedTargetOut, mShape, return false);

  if (outputTensors.vocabParallelLogitsOutOptional != nullptr) {
    op::Shape vocabParallelLogitsOutOptionalShape = {m, n};
    OP_CHECK_WRONG_DIMENSION(outputTensors.vocabParallelLogitsOutOptional, EXPECTED_DIM_TWO, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensors.vocabParallelLogitsOutOptional,
                                                vocabParallelLogitsOutOptionalShape, return false);
  }

  return true;
}

inline static bool CheckAttrValid(int64_t vocabStartIndex, int64_t vocabEndIndex)
{
  if (vocabStartIndex < 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "vocabStartIndex should be greater than or equal to 0, but got %ld",
            vocabStartIndex);
    return false;
  }

  if (vocabEndIndex < vocabStartIndex) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "vocabEndIndex should be greater than or equal to vocabStartIndex, but got "
            "vocabEndIndex %ld, vocabStartIndex %ld", vocabEndIndex, vocabStartIndex);
    return false;
  }

  return true;
}

inline static aclnnStatus CheckParam(FusedLinearOnlineMaxSumInputTensor& inputTensors,
                                     FusedLinearOnlineMaxSumOutputTensor& outputTensors, int64_t vocabStartIndex,
                                     int64_t vocabEndIndex)
{
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(inputTensors, outputTensors), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(inputTensors, outputTensors), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查Shape是否支持
  CHECK_RET(CheckShapeValid(inputTensors, outputTensors), ACLNN_ERR_PARAM_INVALID);
  // 4. 检查属性是否合法
  CHECK_RET(CheckAttrValid(vocabStartIndex, vocabEndIndex), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

inline static bool CheckTupleNotNullptr(
  std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>& result,
  bool vocabParallelLogitsOutFlag)
{
  if (std::get<LOGITS_MAX_LOCAL_OUT_IDX>(result) == nullptr ||
      std::get<SUM_EXP_LOGITS_LOCAL_OUT_IDX>(result) == nullptr ||
      std::get<PREDICTED_LOGITS_LOCAL_OUT_IDX>(result) == nullptr ||
      std::get<TARGET_MASK_OUT_IDX>(result) == nullptr ||
      std::get<MASKED_TARGET_OUT_IDX>(result) == nullptr) {
    return false;
  }

  if (vocabParallelLogitsOutFlag && std::get<VOCAB_PARALLEL_LOGTIS_OUT_OPTIONAL_IDX>(result) == nullptr) {
    return false;
  }

  return true;
}

static const aclTensor* ViewCopyTensors(
  FusedLinearOnlineMaxSumOutputTensor& outputTensors,
  std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>& result,
  bool vocabParallelLogitsOutFlag, aclOpExecutor* executor)
{
  auto viewCopyResult = l0op::ViewCopy(std::get<LOGITS_MAX_LOCAL_OUT_IDX>(result),
                                       outputTensors.logitsMaxLocalOut, executor);
  CHECK_RET(viewCopyResult != nullptr, nullptr);
  viewCopyResult = l0op::ViewCopy(std::get<SUM_EXP_LOGITS_LOCAL_OUT_IDX>(result),
                                  outputTensors.sumExpLogitsLocalOut, executor);
  CHECK_RET(viewCopyResult != nullptr, nullptr);
  viewCopyResult = l0op::ViewCopy(std::get<PREDICTED_LOGITS_LOCAL_OUT_IDX>(result),
                                  outputTensors.predictedLogitsLocalOut, executor);
  CHECK_RET(viewCopyResult != nullptr, nullptr);
  viewCopyResult = l0op::ViewCopy(std::get<TARGET_MASK_OUT_IDX>(result),
                                  outputTensors.targetMaskOut, executor);
  CHECK_RET(viewCopyResult != nullptr, nullptr);
  viewCopyResult = l0op::ViewCopy(std::get<MASKED_TARGET_OUT_IDX>(result),
                                  outputTensors.maskedTargetOut, executor);
  CHECK_RET(viewCopyResult != nullptr, nullptr);

  if (vocabParallelLogitsOutFlag) {
    viewCopyResult = l0op::ViewCopy(std::get<VOCAB_PARALLEL_LOGTIS_OUT_OPTIONAL_IDX>(result),
                                    outputTensors.vocabParallelLogitsOutOptional, executor);
    CHECK_RET(viewCopyResult != nullptr, nullptr);
  }

  return viewCopyResult;
}
}; // namespace

aclnnStatus aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(
    const aclTensor* input, const aclTensor* weight, const aclTensor* target, int64_t vocabStartIndex,
    int64_t vocabEndIndex, aclTensor* logitsMaxLocalOut, aclTensor* sumExpLogitsLocalOut,
    aclTensor* predictedLogitsLocalOut, aclTensor* targetMaskOut, aclTensor* maskedTargetOut,
    aclTensor* vocabParallelLogitsOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
{
  L2_DFX_PHASE_1(aclnnFusedLinearOnlineMaxSum, DFX_IN(input, weight, target, vocabStartIndex, vocabEndIndex),
                 DFX_OUT(logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut,
                         maskedTargetOut, vocabParallelLogitsOutOptional));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  FusedLinearOnlineMaxSumInputTensor inputTensors = {input, weight, target};
  FusedLinearOnlineMaxSumOutputTensor outputTensors = {logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut,
                                                       targetMaskOut, maskedTargetOut, vocabParallelLogitsOutOptional};

  // 入参检查
  auto ret = CheckParam(inputTensors, outputTensors, vocabStartIndex, vocabEndIndex);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (input->GetViewShape().GetDim(DIM_NUM_ZERO) == 0) {
    // 空tensor处理
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  bool vocabParallelLogitsOutFlag = true;
  if (vocabParallelLogitsOutOptional == nullptr) {
    vocabParallelLogitsOutFlag = false;
  }

  auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
  auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
  auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
  std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*> result =
    l0op::FusedLinearOnlineMaxSum(inputContiguous, weightContiguous, targetContiguous, vocabStartIndex, vocabEndIndex,
                                  vocabParallelLogitsOutFlag, uniqueExecutor.get());
  CHECK_RET(CheckTupleNotNullptr(result, vocabParallelLogitsOutFlag), ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = ViewCopyTensors(outputTensors, result, vocabParallelLogitsOutFlag, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取workspace
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedLinearOnlineMaxSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
  L2_DFX_PHASE_2(aclnnFusedLinearOnlineMaxSum);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
