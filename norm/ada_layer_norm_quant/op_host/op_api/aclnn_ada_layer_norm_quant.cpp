/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "op_api/aclnn_util.h"
#include "ada_layer_norm_quant.h"
#include "aclnn_ada_layer_norm_quant.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct AdaLayerNormQuantInputTensor {
    const aclTensor* x;
    const aclTensor* scale;
    const aclTensor* shift;
    const aclTensor* weightOptional;
    const aclTensor* biasOptional;
    const aclTensor* smoothScalesOptional;
};

struct AdaLayerNormQuantOutputTensor {
    aclTensor* out;
    aclTensor* quantScale;
    aclTensor* quantOffsetOptional;
};

static constexpr int32_t MIN_X_DIM = 2;
static constexpr int32_t MAX_X_DIM = 8;

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_INT8, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN};

static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static bool CheckNotNull(AdaLayerNormQuantInputTensor& inputTensor, AdaLayerNormQuantOutputTensor& outputTensor)
{
    OP_CHECK_NULL(inputTensor.x, return false);
    OP_CHECK_NULL(inputTensor.scale, return false);
    OP_CHECK_NULL(inputTensor.shift, return false);
    OP_CHECK_NULL(outputTensor.out, return false);
    OP_CHECK_NULL(outputTensor.quantScale, return false);
    return true;
}

static bool CheckDtypeValid(AdaLayerNormQuantInputTensor& inputTensor, AdaLayerNormQuantOutputTensor& outputTensor)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor.scale, inputTensor.x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor.shift, inputTensor.x->GetDataType(), return false);
    if (inputTensor.weightOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(inputTensor.weightOptional, inputTensor.x->GetDataType(), return false);
    }
    if (inputTensor.biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(inputTensor.biasOptional, inputTensor.x->GetDataType(), return false);
    }
    if (inputTensor.smoothScalesOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(inputTensor.smoothScalesOptional, inputTensor.x->GetDataType(), return false);
    }
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(outputTensor.out, OUT_DTYPE_SUPPORT_LIST_REGBASE, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(outputTensor.out, OUT_DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(outputTensor.quantScale, SCALE_DTYPE_SUPPORT_LIST, return false);
    if (outputTensor.quantOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantOffsetOptional should be nullptr.");
        return false;
    }
    return true;
}

static bool CheckShape(AdaLayerNormQuantInputTensor& inputTensor, AdaLayerNormQuantOutputTensor& outputTensor)
{
    int64_t xDim = inputTensor.x->GetViewShape().GetDimNum();
    OP_CHECK(
        xDim >= MIN_X_DIM && xDim <= MAX_X_DIM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x dim num should be between 2 and 8."), return false);

    op::Shape xShape = inputTensor.x->GetViewShape();
    op::Shape expectShape1;
    op::Shape expectShape2;
    op::Shape scaleExpectShape;
    int64_t B = 1;
    int64_t S = xShape[xDim - MIN_X_DIM];
    int64_t H = xShape[xDim - 1];
    for (int64_t i = 0; i < xDim - MIN_X_DIM; i++) {
        expectShape1.AppendDim(xShape[i]);
        expectShape2.AppendDim(xShape[i]);
        scaleExpectShape.AppendDim(xShape[i]);
        B *= xShape[i];
    }
    expectShape1.AppendDim(1);
    expectShape1.AppendDim(H);
    expectShape2.AppendDim(H);
    scaleExpectShape.AppendDim(S);
    OP_CHECK(B > 0 && S > 0 && H > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x sizes should greater than 0."), return false);

    op::Shape scaleShape = inputTensor.scale->GetViewShape();
    op::Shape shiftShape = inputTensor.shift->GetViewShape();
    if (scaleShape != expectShape1 && scaleShape != expectShape2) {
        return false;
    }
    if (shiftShape != expectShape1 && shiftShape != expectShape2) {
        return false;
    }
    op::Shape expectShape = {H};
    if (inputTensor.weightOptional != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor.weightOptional, expectShape, return false);
    }
    if (inputTensor.biasOptional != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor.biasOptional, expectShape, return false);
    }
    if (inputTensor.smoothScalesOptional != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor.smoothScalesOptional, expectShape, return false);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor.out, xShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor.quantScale, scaleExpectShape, return false);
    return true;
}

static bool CheckAttr(const char* quantMode)
{
    std::string mode = "dynamic";
    if (quantMode != nullptr) {
        mode = std::string(quantMode);
    }
    if (mode != "dynamic") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode only support dynamic.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    AdaLayerNormQuantInputTensor& inputTensor, AdaLayerNormQuantOutputTensor& outputTensor, const char* quantMode)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(inputTensor, outputTensor), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查Shape是否支持
    CHECK_COND(CheckShape(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    // 4. 检查Attr是否合法
    CHECK_COND(CheckAttr(quantMode), ACLNN_ERR_PARAM_INVALID, "CheckAttr failed!");

    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scale, const aclTensor* shift, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* smoothScalesOptional, double epsilon, const char* quantMode,
    aclTensor* out, aclTensor* quantScale, aclTensor* quantOffsetOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnAdaLayerNormQuant,
        DFX_IN(x, scale, shift, weightOptional, biasOptional, smoothScalesOptional, epsilon, quantMode),
        DFX_OUT(out, quantScale, quantOffsetOptional));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    AdaLayerNormQuantInputTensor inputTensor = {x, scale, shift, weightOptional, biasOptional, smoothScalesOptional};
    AdaLayerNormQuantOutputTensor outputTensor = {out, quantScale, quantOffsetOptional};
    auto ret = CheckParams(inputTensor, outputTensor, quantMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor直接返回
    if (x->IsEmpty() || scale->IsEmpty() || shift->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 如果非连续，需要转连续
    x = l0op::Contiguous(x, uniqueExecutor.get());
    scale = l0op::Contiguous(scale, uniqueExecutor.get());
    shift = l0op::Contiguous(shift, uniqueExecutor.get());
    CHECK_RET(x != nullptr && scale != nullptr && shift != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (weightOptional != nullptr) {
        weightOptional = l0op::Contiguous(weightOptional, uniqueExecutor.get());
        CHECK_RET(weightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (biasOptional != nullptr) {
        biasOptional = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (smoothScalesOptional != nullptr) {
        smoothScalesOptional = l0op::Contiguous(smoothScalesOptional, uniqueExecutor.get());
        CHECK_RET(smoothScalesOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    int32_t dstType = out->GetDataType();
    std::tuple<aclTensor*, aclTensor*> result = l0op::AdaLayerNormQuant(
        x, scale, shift, weightOptional, biasOptional, smoothScalesOptional, static_cast<float>(epsilon), quantMode,
        dstType, uniqueExecutor.get());
    const aclTensor* resultTensor = std::get<0>(result);
    const aclTensor* quantScaleTensor = std::get<1>(result);
    CHECK_RET(resultTensor != nullptr && quantScaleTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outResult = l0op::ViewCopy(resultTensor, out, uniqueExecutor.get());
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto scaleResult = l0op::ViewCopy(quantScaleTensor, quantScale, uniqueExecutor.get());
    CHECK_RET(scaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaLayerNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaLayerNormQuant);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
