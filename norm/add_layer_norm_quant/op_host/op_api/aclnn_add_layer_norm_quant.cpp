/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_add_layer_norm_quant.h"
#include "add_layer_norm_quant.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;
constexpr int IDX_4 = 4;

static const std::initializer_list<op::DataType> ASCEND910_95_DTYPE_SUPPORT_LIST_STC = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910_95_DTYPE_SUPPORT_LIST_DYN = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckPlatform()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(
        socVersion == SocVersion::ASCEND910_95,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Support for %s is not implemented.", op::ToString(socVersion).GetString()),
        return false);
    return true;
}

static inline bool CheckOptInputDtype(const aclTensor* tensorPtr, op::DataType dtype)
{
    if (tensorPtr == nullptr) {
        return true;
    }
    OP_CHECK_DTYPE_NOT_MATCH(tensorPtr, dtype, return false);
    return true;
}

static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta, const char* quantMode,
    aclTensor* y1Out, aclTensor* y2Out, const aclTensor* xOut, aclTensor* outScales1Out, aclTensor* outScales2Out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(beta, return false);
    OP_CHECK_NULL(y1Out, return false);
    OP_CHECK_NULL(y2Out, return false);
    OP_CHECK_NULL(xOut, return false);
    OP_CHECK_NULL(outScales1Out, return false);
    OP_CHECK_NULL(outScales2Out, return false);
    CHECK_RET(quantMode != nullptr, false);
    return true;
}

static bool IsDynamicQuant(const char* quantMode)
{
    return (strncmp(quantMode, "dynamic", strlen("dynamic")) == 0);
}

static bool GetQuantMode(const char* quantMode, bool& isDynQuant)
{
    // 检查参数 quantMode 是否合法
    OP_LOGD("GetQuantMode: quantMode: %s", quantMode);
    bool isValidquantMode =
        strncmp(quantMode, "dynamic", strlen("dynamic")) == 0 || strncmp(quantMode, "static", strlen("static")) == 0;
    OP_CHECK(
        isValidquantMode, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Got quantMode neither \"dynamic\" nor \"static\"."),
        return false);
    isDynQuant = IsDynamicQuant(quantMode);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const aclTensor* y1Out,
    const aclTensor* y2Out, const aclTensor* xOut, const aclTensor* outScales1Out, const aclTensor* outScales2Out,
    bool isDynQuant)
{
    // 检查 x 的数据类型是否在 AddLayerNormQuant 算子的支持列表内
    if (isDynQuant) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1, ASCEND910_95_DTYPE_SUPPORT_LIST_DYN, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1, ASCEND910_95_DTYPE_SUPPORT_LIST_STC, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(x2, x1->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gamma, x1->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(beta, x1->GetDataType(), return false);
    CHECK_RET(CheckOptInputDtype(biasOptional, x1->GetDataType()), false);

    OP_CHECK_DTYPE_NOT_MATCH(y1Out, op::DataType::DT_INT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(y2Out, op::DataType::DT_INT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(xOut, x1->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(outScales1Out, op::DataType::DT_FLOAT, return false);
    OP_CHECK_DTYPE_NOT_MATCH(outScales2Out, op::DataType::DT_FLOAT, return false);

    if (nullptr != scales1Optional) {
        if (isDynQuant) {
            OP_CHECK_DTYPE_NOT_SUPPORT(scales1Optional, ASCEND910_95_DTYPE_SUPPORT_LIST_DYN, return false);
            CHECK_RET(CheckOptInputDtype(scales1Optional, x1->GetDataType()), false);
        } else {
            OP_CHECK_DTYPE_NOT_SUPPORT(scales1Optional, ASCEND910_95_DTYPE_SUPPORT_LIST_STC, return false);
        }
        CHECK_RET(CheckOptInputDtype(scales2Optional, scales1Optional->GetDataType()), false);
        CHECK_RET(CheckOptInputDtype(zeroPoints1Optional, scales1Optional->GetDataType()), false);
        CHECK_RET(CheckOptInputDtype(zeroPoints2Optional, scales1Optional->GetDataType()), false);
    }
    return true;
}

static bool CheckShapeRel(const aclTensor* x1, const aclTensor* gamma, const aclTensor* outScales1Out, bool isDynQuant)
{
    OP_CHECK_MAX_DIM(x1, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gamma, MAX_SUPPORT_DIMS_NUMS, return false);

    size_t x1DimNums = x1->GetViewShape().GetDimNum();
    size_t gammaDimNums = gamma->GetViewShape().GetDimNum();
    OP_CHECK(
        (x1DimNums >= gammaDimNums), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DimNum of x1 should bigger than gamma"),
        return false);

    for (size_t i = 0; i < gammaDimNums; i++) {
        OP_CHECK(
            (x1->GetViewShape().GetDim(i + x1DimNums - gammaDimNums) == gamma->GetViewShape().GetDim(i)),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of gamma should be equal to few dims of shape x1"), return false);
    }

    if (isDynQuant) {
        size_t reduceDimNum = outScales1Out->GetViewShape().GetDimNum();
        OP_CHECK(
            ((x1DimNums >= 2) && (gammaDimNums == 1)),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In dyn quant, x1.dims() >= 2 and gamma.dims() == 1 should be followed"),
            return false);

        OP_CHECK(
            (x1DimNums == (1 + reduceDimNum)),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DimNum of outScales1Out should equals to (x1DimNum - 1)"), return false);
        for (size_t i = 0; i < reduceDimNum; i++) {
            OP_CHECK(
                (x1->GetViewShape().GetDim(i) == outScales1Out->GetViewShape().GetDim(i)),
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of outScales should equals to x1 without last dim"),
                return false);
        }
    }
    return true;
}

static bool CheckOptCombValid(
    const aclTensor* scales1Optional, const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional,
    const aclTensor* zeroPoints2Optional, bool isDynQuant)
{
    bool s1Exist = (nullptr != scales1Optional);
    bool s2Exist = (nullptr != scales2Optional);
    bool z1Exist = (nullptr != zeroPoints1Optional);
    bool z2Exist = (nullptr != zeroPoints2Optional);
    OP_LOGI(
        "Got aclnnAddLayerNormQuant opt input combination (%d,%d,%d,%d), isDyn=%d", s1Exist, s2Exist, z1Exist, z2Exist,
        isDynQuant);

    if (isDynQuant) {
        bool validDynQuantComb = (s1Exist && s2Exist && (!z1Exist) && (!z2Exist)) ||
                                 (s1Exist && (!s2Exist) && (!z1Exist) && (!z2Exist)) ||
                                 ((!s1Exist) && (!s2Exist) && (!z1Exist) && (!z2Exist));
        OP_CHECK(
            (validDynQuantComb),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid Optional input combination in dyn quant mode"), return false);
    } else {
        bool validStcQuantComb =
            (s1Exist && s2Exist && z1Exist && z2Exist) || (s1Exist && s2Exist && (!z1Exist) && (!z2Exist)) ||
            (s1Exist && (!s2Exist) && z1Exist && (!z2Exist)) || (s1Exist && (!s2Exist) && (!z1Exist) && (!z2Exist));
        OP_CHECK(
            (validStcQuantComb),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid Optional input combination in stc quant mode"), return false);
    }
    return true;
}

static bool ShapeEqs2Opt(const aclTensor* req, const aclTensor* opt)
{
    if (opt == nullptr) {
        return true;
    }
    return (req->GetViewShape() == opt->GetViewShape());
}

static bool CheckShapeEqs(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const aclTensor* y1Out,
    const aclTensor* y2Out, const aclTensor* xOut, const aclTensor* outScales1Out, const aclTensor* outScales2Out,
    bool isDynQuant)
{
    bool isDual = (nullptr != scales2Optional);
    bool eleShapeEqs = (x1->GetViewShape() == x2->GetViewShape()) && (x1->GetViewShape() == xOut->GetViewShape()) &&
                       (x1->GetViewShape() == y1Out->GetViewShape());
    eleShapeEqs = (isDual) ? eleShapeEqs && (x1->GetViewShape() == y2Out->GetViewShape()) : eleShapeEqs;
    OP_CHECK(eleShapeEqs, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensors x1/x2/y1/y2/x shape not same."), return false);

    bool weightShapeEqs = (gamma->GetViewShape() == beta->GetViewShape()) && ShapeEqs2Opt(gamma, scales1Optional) &&
                          ShapeEqs2Opt(gamma, scales2Optional) && ShapeEqs2Opt(gamma, zeroPoints1Optional) &&
                          ShapeEqs2Opt(gamma, zeroPoints2Optional);
    OP_CHECK(
        weightShapeEqs, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensors gamma/beta/s1/s2/z1/z2 shape not same."),
        return false);

    bool biasShapeEqs = ShapeEqs2Opt(gamma, biasOptional) || ShapeEqs2Opt(x1, biasOptional);
    OP_CHECK(biasShapeEqs, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of bias is not same as x1 or gamma."), return false);

    if (isDual && isDynQuant) {
        bool outScalesEqs = outScales1Out->GetViewShape() == outScales2Out->GetViewShape();
        OP_CHECK(outScalesEqs, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shapes of outScales not equal."), return false);
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, aclTensor* y1Out, aclTensor* y2Out,
    aclTensor* xOut, aclTensor* outScales1Out, aclTensor* outScales2Out, const char* quantMode)
{
    // 当前仅支持910_95
    CHECK_RET(CheckPlatform(), ACLNN_ERR_PARAM_INVALID);

    // 1. 检查参数是否为空指针
    CHECK_RET(
        CheckNotNull(x1, x2, gamma, beta, quantMode, y1Out, y2Out, xOut, outScales1Out, outScales2Out),
        ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数 dataFormat 是否合法
    bool isDynQuant = true;
    CHECK_RET(GetQuantMode(quantMode, isDynQuant), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查 x, gamma, beta, y, mean, variance 的数据类型是否合法
    CHECK_RET(
        CheckDtypeValid(
            x1, x2, gamma, beta, biasOptional, scales1Optional, scales2Optional, zeroPoints1Optional,
            zeroPoints2Optional, y1Out, y2Out, xOut, outScales1Out, outScales2Out, isDynQuant),
        ACLNN_ERR_PARAM_INVALID);

    // 4. 查 x1, gamma, outScales1Out 的等量关系
    CHECK_RET(CheckShapeRel(x1, gamma, outScales1Out, isDynQuant), ACLNN_ERR_PARAM_INVALID);

    // 5. 查 所有shape 的等量关系
    CHECK_RET(
        CheckShapeEqs(
            x1, x2, gamma, beta, biasOptional, scales1Optional, scales2Optional, zeroPoints1Optional,
            zeroPoints2Optional, y1Out, y2Out, xOut, outScales1Out, outScales2Out, isDynQuant),
        ACLNN_ERR_PARAM_INVALID);

    // 6. 查 可选输入组合
    CHECK_RET(
        CheckOptCombValid(scales1Optional, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, isDynQuant),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ComputeAddLayerNormQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const char* quantMode, double epsilon,
    bool additionalOutput, bool divMode, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* outScales1Out,
    aclTensor* outScales2Out, aclOpExecutor* executor)
{
    aclTensor* y1ComputeOut = nullptr;
    aclTensor* y2ComputeOut = nullptr;
    aclTensor* xComputeOut = nullptr;
    aclTensor* outScales1ComputeOut = nullptr;
    aclTensor* outScales2ComputeOut = nullptr;

    bool isDual = (nullptr != scales2Optional);
    bool isDyn = IsDynamicQuant(quantMode);

    auto addLayerNormQuantOuts = l0op::AddLayerNormQuant(
        x1, x2, gamma, beta, biasOptional, scales1Optional, scales2Optional, zeroPoints1Optional, zeroPoints2Optional,
        quantMode, epsilon, additionalOutput, divMode, executor);
    y1ComputeOut = std::get<IDX_0>(addLayerNormQuantOuts);
    y2ComputeOut = std::get<IDX_1>(addLayerNormQuantOuts);
    xComputeOut = std::get<IDX_2>(addLayerNormQuantOuts);
    outScales1ComputeOut = std::get<IDX_3>(addLayerNormQuantOuts);
    outScales2ComputeOut = std::get<IDX_4>(addLayerNormQuantOuts);

    CHECK_RET(
        y1ComputeOut != nullptr && y2ComputeOut != nullptr && xComputeOut != nullptr &&
            outScales1ComputeOut != nullptr && outScales2ComputeOut != nullptr,
        ACLNN_ERR_INNER_NULLPTR);

    // 将 y1ComputeOut 结果拷贝到 y1 上
    auto viewCopyY1Result = l0op::ViewCopy(y1ComputeOut, y1Out, executor);
    CHECK_RET(viewCopyY1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (isDual) {
        // 将 y2ComputeOut 结果拷贝到 y2 上
        auto viewCopyY2Result = l0op::ViewCopy(y2ComputeOut, y2Out, executor);
        CHECK_RET(viewCopyY2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将 xComputeOut 结果拷贝到 x 上
    auto viewCopyXResult = l0op::ViewCopy(xComputeOut, xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (isDyn) {
        // 将 outScales1ComputeOut 结果拷贝到 outScales1 上
        auto viewCopyOutScales1Result = l0op::ViewCopy(outScales1ComputeOut, outScales1Out, executor);
        CHECK_RET(viewCopyOutScales1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (isDual) {
            // 将 outScales2ComputeOut 结果拷贝到 outScales2 上
            auto viewCopyOutScales2Result = l0op::ViewCopy(outScales2ComputeOut, outScales2Out, executor);
            CHECK_RET(viewCopyOutScales2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}

const aclTensor* GetOptTensorContiguous(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const char* quantMode, double epsilon,
    bool additionalOutput, bool divMode, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* outScales1Out,
    aclTensor* outScales2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddLayerNormQuantGetWorkspaceSize");

    L2_DFX_PHASE_1(
        aclnnAddLayerNormQuant,
        DFX_IN(
            x1, x2, gamma, beta, biasOptional, scales1Optional, scales2Optional, zeroPoints1Optional,
            zeroPoints2Optional, quantMode, epsilon, additionalOutput, divMode),
        DFX_OUT(y1Out, y2Out, xOut, outScales1Out, outScales2Out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(
        x1, x2, gamma, beta, biasOptional, scales1Optional, scales2Optional, zeroPoints1Optional, zeroPoints2Optional,
        y1Out, y2Out, xOut, outScales1Out, outScales2Out, quantMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 支持空tensor
    bool hasEmptyTensor =
        x1->IsEmpty() || gamma->IsEmpty() || y2Out->IsEmpty() || outScales1Out->IsEmpty() || outScales2Out->IsEmpty();
    if (hasEmptyTensor) {
        OP_LOGW("Got empty tensor in aclnnAddLayerNormQuant!");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());
    auto betaCont = l0op::Contiguous(beta, uniqueExecutor.get());

    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(betaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto biasCont = GetOptTensorContiguous(biasOptional, uniqueExecutor.get());
    auto s1Cont = GetOptTensorContiguous(scales1Optional, uniqueExecutor.get());
    auto s2Cont = GetOptTensorContiguous(scales2Optional, uniqueExecutor.get());
    auto z1Cont = GetOptTensorContiguous(zeroPoints1Optional, uniqueExecutor.get());
    auto z2Cont = GetOptTensorContiguous(zeroPoints2Optional, uniqueExecutor.get());

    ret = ComputeAddLayerNormQuant(
        x1Cont, x2Cont, gammaCont, betaCont, biasCont, s1Cont, s2Cont, z1Cont, z2Cont, quantMode, epsilon,
        additionalOutput, divMode, y1Out, y2Out, xOut, outScales1Out, outScales2Out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddLayerNormQuantGetWorkspaceSize");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddLayerNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddLayerNormQuant);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
