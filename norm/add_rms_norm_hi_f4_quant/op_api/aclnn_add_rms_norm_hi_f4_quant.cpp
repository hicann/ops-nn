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

#include "add_rms_norm_hi_f4_quant.h"
#include "aclnn_add_rms_norm_hi_f4_quant.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;

static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, aclTensor* yOut,
                         aclTensor* xOut, aclTensor* scaleOut, bool outputRstd, aclTensor* rstdOut)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(xOut, return false);
    OP_CHECK_NULL(scaleOut, return false);
    if (outputRstd) {
        OP_CHECK_NULL(rstdOut, return false);
    }
    return true;
}

static aclnnStatus CheckScaleAlg(int64_t scaleAlg)
{
    if (scaleAlg != 0 && scaleAlg != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale_alg should be 0 or 1, but got %ld.", scaleAlg);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ComputeAddRmsNormHiF4Quant(const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma,
                                              const aclTensor* beta, double epsilon, bool outputRstd, int64_t scaleAlg,
                                              aclTensor* yOut, aclTensor* xOut, aclTensor* scaleOut, aclTensor* rstdOut,
                                              aclOpExecutor* executor)
{
    aclTensor* y_output = executor->AllocTensor(yOut->GetViewShape(), yOut->GetDataType(), yOut->GetViewFormat());
    aclTensor* x_output = executor->AllocTensor(xOut->GetViewShape(), xOut->GetDataType(), xOut->GetViewFormat());
    aclTensor* scale_output = executor->AllocTensor(scaleOut->GetViewShape(), scaleOut->GetDataType(),
                                                    scaleOut->GetViewFormat());
    aclTensor* rstd_output = (outputRstd) ?
                                 executor->AllocTensor(rstdOut->GetViewShape(), rstdOut->GetDataType(),
                                                       rstdOut->GetViewFormat()) :
                                 executor->AllocTensor(op::Shape({0}), op::DataType::DT_FLOAT, op::Format::FORMAT_ND);

    auto outs = l0op::AddRmsNormHiF4Quant(x1, x2, gamma, beta, epsilon, outputRstd, scaleAlg, y_output, x_output,
                                          scale_output, rstd_output, executor);

    auto yComputeOut = std::get<IDX_0>(outs);
    auto xComputeOut = std::get<IDX_1>(outs);
    auto scaleComputeOut = std::get<IDX_2>(outs);
    auto rstdComputeOut = std::get<IDX_3>(outs);

    CHECK_RET(yComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(xComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(scaleComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyYResult = l0op::ViewCopy(yComputeOut, yOut, executor);
    CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyXResult = l0op::ViewCopy(xComputeOut, xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyScaleResult = l0op::ViewCopy(scaleComputeOut, scaleOut, executor);
    CHECK_RET(viewCopyScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (outputRstd) {
        CHECK_RET(rstdComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyRstdResult = l0op::ViewCopy(rstdComputeOut, rstdOut, executor);
        CHECK_RET(viewCopyRstdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormHiF4QuantGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma,
                                                     const aclTensor* beta, double epsilon, bool outputRstd,
                                                     int64_t scaleAlg, aclTensor* yOut, aclTensor* xOut,
                                                     aclTensor* scaleOut, aclTensor* rstdOut, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddRmsNormHiF4QuantGetWorkspaceSize.");
    L2_DFX_PHASE_1(aclnnAddRmsNormHiF4Quant, DFX_IN(x1, x2, gamma, beta, epsilon, outputRstd, scaleAlg),
                   DFX_OUT(yOut, xOut, scaleOut, rstdOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckScaleAlg(scaleAlg) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotNull(x1, x2, gamma, yOut, xOut, scaleOut, outputRstd, rstdOut), ACLNN_ERR_PARAM_NULLPTR);

    auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());

    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* betaCont = (beta == nullptr) ? nullptr : l0op::Contiguous(beta, uniqueExecutor.get());

    auto ret = ComputeAddRmsNormHiF4Quant(x1Cont, x2Cont, gammaCont, betaCont, epsilon, outputRstd, scaleAlg, yOut,
                                          xOut, scaleOut, rstdOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddRmsNormHiF4QuantGetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormHiF4Quant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddRmsNormHiF4Quant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
