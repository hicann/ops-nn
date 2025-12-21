/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "ada_layer_norm_v2.h"
#include "aclnn_ada_layer_norm_v2.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr int32_t MIN_X_DIM = 2;
static constexpr int32_t MAX_X_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(l0op::AdaLayerNormV2InputTensor& inputTensor, l0op::AdaLayerNormV2OutputTensor& outputTensor)
{
    OP_CHECK_NULL(inputTensor.x, return false);
    OP_CHECK_NULL(inputTensor.scale, return false);
    OP_CHECK_NULL(inputTensor.shift, return false);
    OP_CHECK_NULL(outputTensor.out, return false);
    return true;
}

static bool CheckDtypeValid(l0op::AdaLayerNormV2InputTensor& inputTensor, l0op::AdaLayerNormV2OutputTensor& outputTensor)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor.shift, inputTensor.x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor.scale, inputTensor.x->GetDataType(), return false); 
    if (inputTensor.weightOptional != nullptr && inputTensor.biasOptional != nullptr) {
        // weight和bias都不为空时，数据类型保持一致
        OP_CHECK_DTYPE_NOT_MATCH(inputTensor.weightOptional, inputTensor.biasOptional->GetDataType(), return false);
    }
    if (inputTensor.biasOptional != nullptr) {
        if (!outputTensor.useV2 || inputTensor.biasOptional->GetDataType() != op::DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_MATCH(inputTensor.biasOptional, inputTensor.x->GetDataType(), return false);
        }
    }
    if (inputTensor.weightOptional != nullptr) {
        if (!outputTensor.useV2 || inputTensor.weightOptional->GetDataType() != op::DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_MATCH(inputTensor.weightOptional, inputTensor.x->GetDataType(), return false);
        }
    }
    
    OP_CHECK_DTYPE_NOT_MATCH(outputTensor.out, inputTensor.x->GetDataType(), return false);
    if (outputTensor.rstdOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(outputTensor.rstdOutOptional, inputTensor.x->GetDataType(), return false);
    }
    if (outputTensor.meanOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(outputTensor.meanOutOptional, inputTensor.x->GetDataType(), return false);
    }
    return true;
}

static bool CheckShape(l0op::AdaLayerNormV2InputTensor& inputTensor, l0op::AdaLayerNormV2OutputTensor& outputTensor)
{
    int64_t xDim = inputTensor.x->GetViewShape().GetDimNum();
    OP_CHECK(
        xDim >= MIN_X_DIM && xDim <= MAX_X_DIM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x dim num should be between 2 and 8."), return false);

    op::Shape xShape = inputTensor.x->GetViewShape();
    int64_t S = xShape[xDim - MIN_X_DIM];
    int64_t H = xShape[xDim - 1];
    int64_t B = 1;
    op::Shape expectShape1;
    op::Shape expectShape2;
    for (int64_t i = 0; i < xDim - MIN_X_DIM; i++) {
        expectShape2.AppendDim(xShape[i]);
        expectShape1.AppendDim(xShape[i]);
        B *= xShape[i];
    }
    expectShape1.AppendDim(1);
    expectShape1.AppendDim(H);
    expectShape2.AppendDim(H);
    OP_CHECK(B > 0 && S > 0 && H > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x sizes should greater than 0."), return false);

    op::Shape shiftShape = inputTensor.shift->GetViewShape();
    op::Shape scaleShape = inputTensor.scale->GetViewShape();
    if (shiftShape != expectShape1 && shiftShape != expectShape2) {
        return false;
    }
    if (scaleShape != expectShape1 && scaleShape != expectShape2) {
        return false;
    }
    if (inputTensor.biasOptional != nullptr) {
        op::Shape biasExpectShape = {H};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor.biasOptional, biasExpectShape, return false);
    }
    if (inputTensor.weightOptional != nullptr) {
        op::Shape weightExpectShape = {H};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor.weightOptional, weightExpectShape, return false);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor.out, xShape, return false);
    xShape[xDim - 1] = 1;
    if (outputTensor.rstdOutOptional != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor.rstdOutOptional, xShape, return false);
    }
    if (outputTensor.meanOutOptional != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor.meanOutOptional, xShape, return false);
    }
    return true;
}

static aclnnStatus CheckParams(l0op::AdaLayerNormV2InputTensor& inputTensor, l0op::AdaLayerNormV2OutputTensor& outputTensor)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(inputTensor, outputTensor), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查Shape是否支持
    CHECK_COND(CheckShape(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    return ACLNN_SUCCESS;
}

static aclnnStatus AdaLayerNormV2Calculate(
    l0op::AdaLayerNormV2InputTensor& inputTensor, l0op::AdaLayerNormV2OutputTensor& outputTensor, double epsilon, aclOpExecutor* executor)
{
    // 如果非连续，需要转连续
    inputTensor.x = l0op::Contiguous(inputTensor.x, executor);
    inputTensor.scale = l0op::Contiguous(inputTensor.scale, executor);
    inputTensor.shift = l0op::Contiguous(inputTensor.shift, executor);
    CHECK_RET(inputTensor.x != nullptr && inputTensor.scale != nullptr && inputTensor.shift != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (inputTensor.biasOptional != nullptr) {
        inputTensor.biasOptional = l0op::Contiguous(inputTensor.biasOptional, executor);
        CHECK_RET(inputTensor.biasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (inputTensor.weightOptional != nullptr) {
        inputTensor.weightOptional = l0op::Contiguous(inputTensor.weightOptional, executor);
        CHECK_RET(inputTensor.weightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    std::tuple<aclTensor *, aclTensor *, aclTensor *> result = l0op::AdaLayerNormV2(
        inputTensor, static_cast<float>(epsilon), executor);
    const aclTensor *resultTensor = std::get<0>(result);
    const aclTensor *meanTensor = std::get<1>(result);
    const aclTensor *rstdTensor = std::get<2>(result);
    CHECK_RET(resultTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outResult = l0op::ViewCopy(resultTensor, outputTensor.out, executor);
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (outputTensor.rstdOutOptional != nullptr) {
        CHECK_RET(rstdTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto rstdResult = l0op::ViewCopy(rstdTensor, outputTensor.rstdOutOptional, executor);
        CHECK_RET(rstdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (outputTensor.meanOutOptional != nullptr) {
        CHECK_RET(meanTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanResult = l0op::ViewCopy(meanTensor, outputTensor.meanOutOptional, executor);
        CHECK_RET(meanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
}
}; // namespace

aclnnStatus aclnnAdaLayerNormV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* scale, const aclTensor* shift, const aclTensor* weightOptional,
    const aclTensor* biasOptional, double epsilon, aclTensor* out, aclTensor* meanOutOptional,
    aclTensor* rstdOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaLayerNormV2, DFX_IN(x, scale, shift, weightOptional, biasOptional, epsilon), DFX_OUT(out, meanOutOptional, rstdOutOptional));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    l0op::AdaLayerNormV2InputTensor inputTensor = {x, scale, shift, weightOptional, biasOptional};
    l0op::AdaLayerNormV2OutputTensor outputTensor = {out, meanOutOptional, rstdOutOptional, true};
    auto ret = CheckParams(inputTensor, outputTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    AdaLayerNormV2Calculate(inputTensor, outputTensor, epsilon, uniqueExecutor.get());

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaLayerNormV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaLayerNormV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
