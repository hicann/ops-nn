/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_dequant_bias.h"
#include "dequant_bias.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_X_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_WS_LIST = {op::DataType::DT_FLOAT,
                                                                          op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_AS_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_BIAS_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_OUT_LIST = {op::DataType::DT_FLOAT16,
                                                                           op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* x, const aclTensor* weightScale, const aclTensor* activateScale,
                         const aclTensor* bias, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weightScale, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* weightScale, const aclTensor* activateScale,
                            const aclTensor* bias, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_X_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, DTYPE_SUPPORT_WS_LIST, return false);
    if (activateScale != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(activateScale, DTYPE_SUPPORT_AS_LIST, return false);
    }
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_BIAS_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_OUT_LIST, return false);
    return true;
}

static bool CheckOutputDtypeValid(int64_t outputDtype)
{
    OP_CHECK(
        outputDtype == 1 || outputDtype == 27,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputDtype should be 1 (FLOAT16) or 27 (BFLOAT16), got %ld.", outputDtype),
        return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    // out 的 view shape 必须与 x 一致（kernel 按 x 的 [M, N] 生成输出）
    OP_CHECK_SHAPE_NOT_EQUAL(x, out, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* weightScale, const aclTensor* activateScale,
                               const aclTensor* bias, int64_t outputDtype, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(x, weightScale, activateScale, bias, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, weightScale, activateScale, bias, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutputDtypeValid(outputDtype), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDequantBiasGetWorkspaceSize(const aclTensor* x, const aclTensor* weightScale,
                                             const aclTensor* activateScaleOptional, const aclTensor* biasOptional,
                                             int64_t outputDtype, const aclTensor* out, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnDequantBias, DFX_IN(x, weightScale, activateScaleOptional, biasOptional), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, weightScale, activateScaleOptional, biasOptional, outputDtype, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightScaleContiguous = l0op::Contiguous(weightScale, uniqueExecutor.get());
    CHECK_RET(weightScaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* activateScaleContiguous = nullptr;
    if (activateScaleOptional != nullptr) {
        activateScaleContiguous = l0op::Contiguous(activateScaleOptional, uniqueExecutor.get());
        CHECK_RET(activateScaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* biasContiguous = nullptr;
    if (biasOptional != nullptr) {
        biasContiguous = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 输出out连续且 storage 与 view 一致、dtype 与 outputDtype 推导一致时，
    // 直接绑定 out 作为 kernel 输出（kernel 直接写入调用方内存）；
    // 非连续时先算到连续张量，再按 out 的 stride 回写
    const bool directOut = IsContiguous(out) && out->GetStorageShape() == out->GetViewShape() &&
                           ((outputDtype == 1 && out->GetDataType() == op::DataType::DT_FLOAT16) ||
                            (outputDtype == 27 && out->GetDataType() == op::DataType::DT_BF16));
    const aclTensor* dequantBiasOut = nullptr;
    if (directOut) {
        dequantBiasOut = l0op::DequantBias(xContiguous, weightScaleContiguous, activateScaleContiguous, biasContiguous,
                                           outputDtype, uniqueExecutor.get(), out);
        CHECK_RET(dequantBiasOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        dequantBiasOut = l0op::DequantBias(xContiguous, weightScaleContiguous, activateScaleContiguous, biasContiguous,
                                           outputDtype, uniqueExecutor.get());
        CHECK_RET(dequantBiasOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(dequantBiasOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDequantBias(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDequantBias);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
