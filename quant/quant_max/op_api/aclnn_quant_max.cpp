/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_max.h"
#include "quant_max.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include <string>
#include <cstring>

using namespace op;

constexpr int64_t ACLNN_MIN_SHAPE_RANK = 1;
constexpr int64_t ACLNN_MAX_SHAPE_RANK = 8;

// dstType 枚举值：34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN
constexpr int64_t DST_TYPE_HIFLOAT8 = 34;
constexpr int64_t DST_TYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DST_TYPE_FLOAT8_E4M3FN = 36;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

// dstType 对应的输出 dtype 映射
static op::DataType GetOutputDtype(int64_t dstType)
{
    switch (dstType) {
        case DST_TYPE_HIFLOAT8:
            return DataType::DT_HIFLOAT8;
        case DST_TYPE_FLOAT8_E5M2:
            return DataType::DT_FLOAT8_E5M2;
        case DST_TYPE_FLOAT8_E4M3FN:
            return DataType::DT_FLOAT8_E4M3FN;
        default:
            return DataType::DT_UNDEFINED;
    }
}

static bool IsShapeEqualsOne(const aclTensor* tensor)
{
    auto shape = tensor->GetViewShape();
    return shape.GetDimNum() == 1 && shape.GetDim(0) == 1;
}

static bool IsShapeEquals(const aclTensor* tensor1, const aclTensor* tensor2)
{
    auto shape1 = tensor1->GetViewShape();
    auto shape2 = tensor2->GetViewShape();
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x shape %ld, y shape %ld", shape1.GetShapeSize(), shape2.GetShapeSize());
    if (shape1.GetDimNum() != shape2.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < shape1.GetDimNum(); i++) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}

// 1. 空指针检查
static aclnnStatus CheckNullParams(const aclTensor* x, const aclTensor* scale, const aclTensor* y, const aclTensor* amax)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(y, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(amax, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

// 2. 输入参数检查（x 和 scale）
static aclnnStatus CheckInputParams(const aclTensor* x, const aclTensor* scale)
{
    // x 数据类型检查
    if (!CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dtype must be FLOAT/FLOAT16/BFLOAT16, got %d",
                static_cast<int>(x->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // x 维度检查：1-8 维
    auto xShape = x->GetViewShape();
    if (xShape.GetDimNum() < ACLNN_MIN_SHAPE_RANK || xShape.GetDimNum() > ACLNN_MAX_SHAPE_RANK) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dim must be in range [1, 8], got %zu", xShape.GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // scale 数据类型检查：必须为 FLOAT
    if (scale->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale dtype must be FLOAT, got %d",
                static_cast<int>(scale->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // scale shape 检查：必须为 [1]，不支持空 Tensor
    if (scale->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale does not support empty tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsShapeEqualsOne(scale)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale shape must be [1]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

// 3. dstType 检查
static aclnnStatus CheckDstType(int64_t dstType)
{
    if (dstType != DST_TYPE_HIFLOAT8 && dstType != DST_TYPE_FLOAT8_E5M2 && dstType != DST_TYPE_FLOAT8_E4M3FN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dstType must be 34(HIFLOAT8)/35(FLOAT8_E5M2)/36(FLOAT8_E4M3FN), got %ld",
                dstType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 4. roundMode 检查
// - FLOAT8_E5M2 (35) 或 FLOAT8_E4M3FN (36): 只支持 "rint"
// - HIFLOAT8 (34): 支持 "round" 或 "hybrid"
static aclnnStatus CheckRoundMode(int64_t dstType, const char* roundMode)
{
    // roundMode 不能为 nullptr
    if (roundMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roundMode cannot be nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // roundMode 不能为空字符串
    if (strlen(roundMode) == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roundMode cannot be empty string");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 根据 dstType 校验 roundMode
    if (dstType == DST_TYPE_FLOAT8_E5M2 || dstType == DST_TYPE_FLOAT8_E4M3FN) {
        // FLOAT8_E5M2 和 FLOAT8_E4M3FN 只支持 "rint"
        if (strcmp(roundMode, "rint") != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "roundMode must be 'rint' for dstType FLOAT8_E5M2(35) or FLOAT8_E4M3FN(36), got %s",
                    roundMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (dstType == DST_TYPE_HIFLOAT8) {
        // HIFLOAT8 只支持 "round" 或 "hybrid"
        if (strcmp(roundMode, "round") != 0 && strcmp(roundMode, "hybrid") != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "roundMode must be 'round' or 'hybrid' for dstType HIFLOAT8(34), got %s",
                    roundMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// 4. 输出参数检查（y 和 amax）
static aclnnStatus CheckOutputParams(const aclTensor* x, const aclTensor* y, const aclTensor* amax, int64_t dstType)
{
    // y 数据类型检查：必须与 dstType 对应
    auto expectedYDtype = GetOutputDtype(dstType);
    if (y->GetDataType() != expectedYDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dtype must be %d according to dstType, got %d",
                static_cast<int>(expectedYDtype), static_cast<int>(y->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // y shape 检查：必须与 x 一致
    if (!IsShapeEquals(y, x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y shape must be same as x shape");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // amax 数据类型检查：必须与 x 一致
    if (amax->GetDataType() != x->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "amax dtype must be same as x dtype, got amax=%d, x=%d",
                static_cast<int>(amax->GetDataType()), static_cast<int>(x->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // amax shape 检查：必须为 [1]
    if (!IsShapeEqualsOne(amax)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "amax shape must be [1]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclTensor* scale, int64_t dstType, const char* roundMode, const aclTensor* y,
    const aclTensor* amax)
{
    auto ret = CheckNullParams(x, scale, y, amax);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckInputParams(x, scale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckDstType(dstType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckRoundMode(dstType, roundMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckOutputParams(x, y, amax, dstType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMaxGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scale, char* roundMode, int64_t dstType, const aclTensor* y,
    const aclTensor* amax, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantMax, DFX_IN(x, scale), DFX_OUT(y, amax));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, scale, dstType, roundMode, y, amax);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scaleContiguous = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto yContiguous = l0op::Contiguous(y, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto amaxContiguous = l0op::Contiguous(amax, uniqueExecutor.get());
    CHECK_RET(amaxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // roundMode 已在 CheckParams 中校验，直接使用
    const aclTensor* opResult = l0op::QuantMax(
        xContiguous, scaleContiguous, roundMode, dstType, yContiguous, amaxContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(opResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMax);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
