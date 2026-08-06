/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_matmul_emu_split_weight.h"
#include "matmul_emu_split_weight.h"
#include "aclnn_kernels/contiguous.h"
#include <cmath>
#include <climits>
#include <cstring>

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"

using namespace op;
using namespace Ops::NN;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const char* const kOpName = "aclnnMatmulEmuSplitWeightGetWorkspaceSize";
static constexpr size_t DIM_LEN = 2;
static constexpr int8_t Y_DTYPE_FP32 = 0;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> Y_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor* x, const aclTensor* wHigh, const aclTensor* wLow, const aclTensor* y)
{
    OP_CHECK(x != nullptr,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x", "nullptr",
                                                   FormatString("The value of %s cannot be %s", "x", "null").c_str()),
             return false);
    OP_CHECK(wHigh != nullptr,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 kOpName, "w_high", "nullptr", FormatString("The value of %s cannot be %s", "w_high", "null").c_str()),
             return false);
    OP_CHECK(wLow != nullptr,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 kOpName, "w_low", "nullptr", FormatString("The value of %s cannot be %s", "w_low", "null").c_str()),
             return false);
    OP_CHECK(y != nullptr,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "y", "nullptr",
                                                   FormatString("The value of %s cannot be %s", "y", "null").c_str()),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* wHigh, const aclTensor* wLow, const aclTensor* y)
{
    auto fmtX = ge::GetPrimaryFormat(x->GetStorageFormat());
    auto fmtWHigh = ge::GetPrimaryFormat(wHigh->GetStorageFormat());
    auto fmtWLow = ge::GetPrimaryFormat(wLow->GetStorageFormat());
    auto fmtY = ge::GetPrimaryFormat(y->GetStorageFormat());
    OP_CHECK(fmtX == ge::FORMAT_ND && fmtWHigh == ge::FORMAT_ND && fmtWLow == ge::FORMAT_ND && fmtY == ge::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                 kOpName, "x, w_high, w_low, y",
                 FormatString("%d, %d, %d, %d", static_cast<int32_t>(fmtX), static_cast<int32_t>(fmtWHigh),
                              static_cast<int32_t>(fmtWLow), static_cast<int32_t>(fmtY))
                     .c_str(),
                 FormatString("The formats of %s must be %s", "x, w_high, w_low, y", "ND").c_str()),
             return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* wHigh, const aclTensor* wLow, const aclTensor* y,
                            int8_t yDtype)
{
    OP_CHECK(
        CheckType(x->GetDataType(), DTYPE_SUPPORT_LIST),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "x", op::ToString(x->GetDataType()).GetString(),
            FormatString("The dtype of %s must be in %s", "x", op::ToString(DTYPE_SUPPORT_LIST).GetString()).c_str()),
        return false);
    OP_CHECK(CheckType(wHigh->GetDataType(), DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                 kOpName, "w_high", op::ToString(wHigh->GetDataType()).GetString(),
                 FormatString("The dtype of %s must be in %s", "w_high", op::ToString(DTYPE_SUPPORT_LIST).GetString())
                     .c_str()),
             return false);
    OP_CHECK(CheckType(wLow->GetDataType(), DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                 kOpName, "w_low", op::ToString(wLow->GetDataType()).GetString(),
                 FormatString("The dtype of %s must be in %s", "w_low", op::ToString(DTYPE_SUPPORT_LIST).GetString())
                     .c_str()),
             return false);

    OP_CHECK(wHigh->GetDataType() == x->GetDataType(),
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 kOpName, "x, w_high",
                 FormatString("%s, %s", op::ToString(x->GetDataType()).GetString(),
                              op::ToString(wHigh->GetDataType()).GetString())
                     .c_str(),
                 FormatString("The dtypes of %s must be the same", "x and w_high").c_str()),
             return false);
    OP_CHECK(wLow->GetDataType() == x->GetDataType(),
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 kOpName, "x, w_low",
                 FormatString("%s, %s", op::ToString(x->GetDataType()).GetString(),
                              op::ToString(wLow->GetDataType()).GetString())
                     .c_str(),
                 FormatString("The dtypes of %s must be the same", "x and w_low").c_str()),
             return false);

    OP_CHECK(yDtype == Y_DTYPE_FP32,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 kOpName, "yDtype", FormatString("%d", yDtype).c_str(),
                 FormatString("The value of %s must be %d", "yDtype", Y_DTYPE_FP32).c_str()),
             return false);

    OP_CHECK(
        CheckType(y->GetDataType(), Y_DTYPE_SUPPORT_LIST),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "y", op::ToString(y->GetDataType()).GetString(),
            FormatString("The dtype of %s must be in %s", "y", op::ToString(Y_DTYPE_SUPPORT_LIST).GetString()).c_str()),
        return false);
    return true;
}

static bool CheckScaleValid(float wLowScale)
{
    OP_CHECK(!std::isnan(wLowScale),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 kOpName, "wLowScale", "NaN", FormatString("The value of %s cannot be %s", "wLowScale", "NaN").c_str()),
             return false);
    OP_CHECK(!std::isinf(wLowScale),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 kOpName, "wLowScale", "Inf", FormatString("The value of %s cannot be %s", "wLowScale", "Inf").c_str()),
             return false);
    constexpr float EXPECTED_SCALE = 0.00390625f;
    constexpr float SCALE_EPSILON = 1e-7f;
    OP_CHECK(std::fabs(wLowScale - EXPECTED_SCALE) <= SCALE_EPSILON,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 kOpName, "wLowScale", FormatString("%f", wLowScale).c_str(),
                 FormatString("The value of %s must be %f", "wLowScale", EXPECTED_SCALE).c_str()),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* wHigh, const aclTensor* wLow, const aclTensor* y)
{
    OP_CHECK(x->GetViewShape().GetDimNum() == DIM_LEN,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 kOpName, "x", FormatString("%zu", x->GetViewShape().GetDimNum()).c_str(),
                 FormatString("The shape dim of %s must be %zu", "x", DIM_LEN).c_str()),
             return false);
    OP_CHECK(wHigh->GetViewShape().GetDimNum() == DIM_LEN,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 kOpName, "w_high", FormatString("%zu", wHigh->GetViewShape().GetDimNum()).c_str(),
                 FormatString("The shape dim of %s must be %zu", "w_high", DIM_LEN).c_str()),
             return false);
    OP_CHECK(wLow->GetViewShape().GetDimNum() == DIM_LEN,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 kOpName, "w_low", FormatString("%zu", wLow->GetViewShape().GetDimNum()).c_str(),
                 FormatString("The shape dim of %s must be %zu", "w_low", DIM_LEN).c_str()),
             return false);
    OP_CHECK(y->GetViewShape().GetDimNum() == DIM_LEN,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 kOpName, "y", FormatString("%zu", y->GetViewShape().GetDimNum()).c_str(),
                 FormatString("The shape dim of %s must be %zu", "y", DIM_LEN).c_str()),
             return false);

    const auto& xShape = x->GetViewShape();
    const auto& wHighShape = wHigh->GetViewShape();
    const auto& wLowShape = wLow->GetViewShape();
    const auto& yShape = y->GetViewShape();

    int64_t m = xShape[0];
    int64_t k = xShape[1];
    int64_t wHighK = wHighShape[0];
    int64_t n = wHighShape[1];
    int64_t wLowK = wLowShape[0];
    int64_t wLowN = wLowShape[1];

    OP_CHECK(m > 0 && m <= static_cast<int64_t>(INT32_MAX) && n > 0 && n <= static_cast<int64_t>(INT32_MAX) && k > 0 &&
                 k <= static_cast<int64_t>(INT32_MAX),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 kOpName, "m, k, n", FormatString("%ld, %ld, %ld", m, k, n).c_str(),
                 FormatString("The values of %s must be in range %s", "m, k, n", "(0, INT32_MAX]").c_str()),
             return false);

    OP_CHECK(
        k == wHighK,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x K, w_high K", FormatString("%ld, %ld", k, wHighK).c_str(),
                                              FormatString("The value of %s must match %s", "x K", "w_high K").c_str()),
        return false);
    OP_CHECK(
        k == wLowK,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "x K, w_low K", FormatString("%ld, %ld", k, wLowK).c_str(),
                                              FormatString("The value of %s must match %s", "x K", "w_low K").c_str()),
        return false);
    OP_CHECK(n == wLowN,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 kOpName, "w_high N, w_low N", FormatString("%ld, %ld", n, wLowN).c_str(),
                 FormatString("The value of %s must match %s", "w_high N", "w_low N").c_str()),
             return false);

    OP_CHECK(
        yShape[0] == m && yShape[1] == n,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "y", op::ToString(yShape).GetString(),
                                              FormatString("The shape of %s must be [%ld, %ld]", "y", m, n).c_str()),
        return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* wHigh, const aclTensor* wLow, const aclTensor* y,
                               float wLowScale, int8_t yDtype)
{
    CHECK_RET(CheckNotNull(x, wHigh, wLow, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, wHigh, wLow, y, yDtype), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(x, wHigh, wLow, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckScaleValid(wLowScale), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, wHigh, wLow, y), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnMatmulEmuSplitWeightGetWorkspaceSize(const aclTensor* x, const aclTensor* wHigh, const aclTensor* wLow,
                                                      const aclTensor* y, float wLowScale, int8_t yDtype,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMatmulEmuSplitWeight, DFX_IN(x, wHigh, wLow, y, wLowScale, yDtype), DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, wHigh, wLow, y, wLowScale, yDtype);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool transposeX = Ops::NN::IsTransposeLastTwoDims(x);
    bool transposeW = Ops::NN::IsTransposeLastTwoDims(wHigh);

    auto contiguousX = x;
    if (transposeX) {
        contiguousX = uniqueExecutor->CreateView(x, Ops::NN::SwapLastTwoDimValue(x->GetViewShape()),
                                                 x->GetViewOffset());
        CHECK_RET(contiguousX != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        contiguousX = l0op::Contiguous(x, uniqueExecutor.get());
        CHECK_RET(contiguousX != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto contiguousWHigh = wHigh;
    if (transposeW) {
        contiguousWHigh = uniqueExecutor->CreateView(wHigh, Ops::NN::SwapLastTwoDimValue(wHigh->GetViewShape()),
                                                     wHigh->GetViewOffset());
        CHECK_RET(contiguousWHigh != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        contiguousWHigh = l0op::Contiguous(wHigh, uniqueExecutor.get());
        CHECK_RET(contiguousWHigh != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto contiguousWLow = wLow;
    if (transposeW) {
        contiguousWLow = uniqueExecutor->CreateView(wLow, Ops::NN::SwapLastTwoDimValue(wLow->GetViewShape()),
                                                    wLow->GetViewOffset());
        CHECK_RET(contiguousWLow != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        contiguousWLow = l0op::Contiguous(wLow, uniqueExecutor.get());
        CHECK_RET(contiguousWLow != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outDtype = op::DataType::DT_FLOAT;

    auto result = l0op::MatmulEmuSplitWeight(contiguousX, contiguousWHigh, contiguousWLow, outDtype, wLowScale,
                                             static_cast<int32_t>(yDtype), transposeX, transposeW,
                                             uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmulEmuSplitWeight(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMatmulEmuSplitWeight);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
