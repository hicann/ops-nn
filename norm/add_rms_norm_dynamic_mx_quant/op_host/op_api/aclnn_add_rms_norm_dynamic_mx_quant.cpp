/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/format_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"

#include "add_rms_norm_dynamic_mx_quant.h"
#include "aclnn_add_rms_norm_dynamic_mx_quant.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;

static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, aclTensor* yOut, aclTensor* xOut,
    aclTensor* mxscaleOut, bool outputRstd, aclTensor* rstdOut)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(gamma, return false);
    // beta 是可选输入，不校验nullptr
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(xOut, return false);
    OP_CHECK_NULL(mxscaleOut, return false);
    // rstdOut 根据 outputRstd 决定是否校验
    if (outputRstd) {
        OP_CHECK_NULL(rstdOut, return false);
    }
    return true;
}

static aclnnStatus ComputeAddRmsNormDynamicMxQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta, double epsilon,
    int64_t scaleAlg, char* roundMode, int64_t dstType, bool outputRstd, aclTensor* yOut, aclTensor* xOut,
    aclTensor* mxscaleOut, aclTensor* rstdOut, aclOpExecutor* executor)
{
    // 创建输出Tensor
    aclTensor* y_output = executor->AllocTensor(yOut->GetViewShape(), yOut->GetDataType(), yOut->GetViewFormat());
    aclTensor* x_output = executor->AllocTensor(xOut->GetViewShape(), xOut->GetDataType(), xOut->GetViewFormat());
    aclTensor* mxscale_output =
        executor->AllocTensor(mxscaleOut->GetViewShape(), mxscaleOut->GetDataType(), mxscaleOut->GetViewFormat());
    aclTensor* rstd_output =
        (outputRstd) ?
            executor->AllocTensor(rstdOut->GetViewShape(), rstdOut->GetDataType(), rstdOut->GetViewFormat()) :
            executor->AllocTensor(op::Shape({0}), op::DataType::DT_FLOAT, op::Format::FORMAT_ND);

    auto addRmsNormDynamicMxQuantOuts = l0op::AddRmsNormDynamicMxQuant(
        x1, x2, gamma, beta, epsilon, scaleAlg, roundMode, dstType, outputRstd, y_output, x_output, mxscale_output,
        rstd_output, executor);

    auto yComputeOut = std::get<IDX_0>(addRmsNormDynamicMxQuantOuts);
    auto xComputeOut = std::get<IDX_1>(addRmsNormDynamicMxQuantOuts);
    auto mxscaleComputeOut = std::get<IDX_2>(addRmsNormDynamicMxQuantOuts);
    auto rstdComputeOut = std::get<IDX_3>(addRmsNormDynamicMxQuantOuts);

    // 校验输出不为空
    CHECK_RET(yComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(xComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(mxscaleComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将结果拷贝到输出tensor
    auto viewCopyYResult = l0op::ViewCopy(yComputeOut, yOut, executor);
    CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyXResult = l0op::ViewCopy(xComputeOut, xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyMxscaleResult = l0op::ViewCopy(mxscaleComputeOut, mxscaleOut, executor);
    CHECK_RET(viewCopyMxscaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (outputRstd) {
        CHECK_RET(rstdComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyRstdResult = l0op::ViewCopy(rstdComputeOut, rstdOut, executor);
        CHECK_RET(viewCopyRstdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta, double epsilon,
    int64_t scaleAlg, char* roundMode, int64_t dstType, bool outputRstd, aclTensor* yOut, aclTensor* xOut,
    aclTensor* mxscaleOut, aclTensor* rstdOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize.");
    L2_DFX_PHASE_1(
        aclnnAddRmsNormDynamicMxQuant, DFX_IN(x1, x2, gamma, beta, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        DFX_OUT(yOut, xOut, mxscaleOut, rstdOut));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 检查必选输入/输出是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, gamma, yOut, xOut, mxscaleOut, outputRstd, rstdOut), ACLNN_ERR_PARAM_NULLPTR);

    // 校验x1除尾轴外的维度不能为0
    auto x1Shape = x1->GetViewShape();
    auto gammaDimNum = gamma->GetViewShape().GetDimNum();
    for (int64_t i = 0; i < static_cast<int64_t>(x1Shape.GetDimNum() - gammaDimNum); ++i) {
        if (x1Shape.GetDim(i) == 0) {
            OP_LOGW("Got empty tensor in aclnnAddRmsNormDynamicMxQuant!");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());

    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* betaCont = (beta == nullptr) ? nullptr : l0op::Contiguous(beta, uniqueExecutor.get());

    auto ret = ComputeAddRmsNormDynamicMxQuant(
        x1Cont, x2Cont, gammaCont, betaCont, epsilon, scaleAlg, roundMode, dstType, outputRstd, yOut, xOut, mxscaleOut,
        rstdOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormDynamicMxQuant(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddRmsNormDynamicMxQuant);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
