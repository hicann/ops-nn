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
#include "aclnn_kernels/reshape.h"
#include "op_api/aclnn_util.h"

#include "rms_norm_quant_v3.h"
#include "aclnn_rms_norm_quant_v3.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const size_t DIMS_ONE_NUMS = 1;
static const size_t DIMS_TWO_NUMS = 2;
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr int32_t IDX_0 = 0;
static constexpr int32_t IDX_1 = 1;

static bool CheckNotNull(const aclTensor* x, const aclTensor* gamma, const aclTensor* scale, aclTensor* y,
                         bool outputRstd, aclTensor* rstd)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(scale, return false);
    // offset 和 beta 是可选输入，不校验nullptr
    OP_CHECK_NULL(y, return false);
    // rstd 根据 outputRstd 决定是否校验
    if (outputRstd) {
        OP_CHECK_NULL(rstd, return false);
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* x, const aclTensor* gamma, const aclTensor* scale, const aclTensor* y)
{
    // gamma 维度校验：1~2维
    int64_t gammaDimNum = static_cast<int64_t>(gamma->GetViewShape().GetDimNum());
    if (gammaDimNum < static_cast<int64_t>(DIMS_ONE_NUMS) || gammaDimNum > static_cast<int64_t>(DIMS_TWO_NUMS)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DimNum of gamma must be 1 or 2, but get [%ld]", gammaDimNum);
        return false;
    }
    // scale 维度校验：必须为1维
    int64_t scaleDimNum = static_cast<int64_t>(scale->GetViewShape().GetDimNum());
    if (scaleDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DimNum of scale must be 1, but get [%ld]", scaleDimNum);
        return false;
    }
    int64_t xDimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    int64_t xLastDim = x->GetViewShape().GetDim(xDimNum - 1);
    int64_t yDimNum = static_cast<int64_t>(y->GetViewShape().GetDimNum());
    int64_t outLastDim = y->GetViewShape().GetDim(yDimNum - 1);
    if (y->GetDataType() == op::DataType::DT_INT32) {
        OP_CHECK(xLastDim % INT4_NUMS_IN_INT32_SPACE == 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "if outType is int32, the last dim of x must be a multiple of 8,"
                         " but got %ld.",
                         xLastDim),
                 return false);
        OP_CHECK(xLastDim == outLastDim * INT4_NUMS_IN_INT32_SPACE,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "if outType is int32, the out last dim must be 1/8 of x last dim,"
                         " x last dim is (%ld), out last dim is (%ld).",
                         xLastDim, outLastDim),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* gamma, const aclTensor* scale, aclTensor* y,
                               bool outputRstd, aclTensor* rstd)
{
    CHECK_RET(CheckNotNull(x, gamma, scale, y, outputRstd, rstd), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckShapeValid(x, gamma, scale, y), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus Int42Int32PackedTensor(const aclTensor* y, const aclTensor*& outTensor, aclOpExecutor* executor)
{
    auto viewShape = y->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    OP_LOGD("aclnnRmsNormQuantV3 output real dtype is int4, pack to int32 to out.");

    return ACLNN_SUCCESS;
}

static aclnnStatus PreDealData(const aclTensor*& gamma, const aclTensor*& beta, aclOpExecutor* executor)
{
    if (gamma != nullptr && gamma->GetViewShape().GetDimNum() == DIMS_TWO_NUMS) {
        auto gammaReshape = l0op::Reshape(gamma, {gamma->GetViewShape()[1]}, executor);
        CHECK_RET(gammaReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gamma = gammaReshape;
    }
    if (beta != nullptr && beta->GetViewShape().GetDimNum() == DIMS_TWO_NUMS) {
        auto betaReshape = l0op::Reshape(beta, {beta->GetViewShape()[1]}, executor);
        CHECK_RET(betaReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        beta = betaReshape;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ComputeRmsNormQuantV3(const aclTensor* x, const aclTensor* gamma, const aclTensor* scale,
                                         const aclTensor* offset, const aclTensor* beta, double epsilon, int32_t yType,
                                         bool divMode, bool outputRstd, aclTensor* y, aclTensor* rstd,
                                         aclOpExecutor* executor)
{
    // 创建rstd输出Tensor
    aclTensor* rstdOutput = (outputRstd) ?
                                executor->AllocTensor(rstd->GetViewShape(), rstd->GetDataType(),
                                                      rstd->GetViewFormat()) :
                                executor->AllocTensor(op::Shape({0}), op::DataType::DT_FLOAT, op::Format::FORMAT_ND);

    // 调用l0op::RmsNormQuantV3，其中offset对应l0接口中的zero_points
    auto rmsNormQuantV3Outs = l0op::RmsNormQuantV3(x, gamma, scale, nullptr, offset, nullptr, beta, epsilon, divMode,
                                                   yType, outputRstd, rstdOutput, executor);

    auto resultTensor = std::get<IDX_0>(rmsNormQuantV3Outs);
    auto rstdComputeOut = std::get<IDX_1>(rmsNormQuantV3Outs);
    CHECK_RET(resultTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 处理INT4→INT32 packing
    const aclTensor* outTensor = resultTensor;
    if (yType == op::DataType::DT_INT4 && y->GetDataType() == op::DataType::DT_INT32) {
        auto ret = Int42Int32PackedTensor(resultTensor, outTensor, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 将结果拷贝到输出tensor
    auto viewCopyY = l0op::ViewCopy(outTensor, y, executor);
    CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (outputRstd) {
        CHECK_RET(rstdComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyRstd = l0op::ViewCopy(rstdComputeOut, rstd, executor);
        CHECK_RET(viewCopyRstd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnRmsNormQuantV3GetWorkspaceSize(const aclTensor* x, const aclTensor* gamma, const aclTensor* scale,
                                                const aclTensor* offset, const aclTensor* beta, double epsilon,
                                                bool divMode, bool outputRstd, aclTensor* y, aclTensor* rstd,
                                                uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnRmsNormQuantV3GetWorkspaceSize.");
    L2_DFX_PHASE_1(aclnnRmsNormQuantV3, DFX_IN(x, gamma, scale, offset, beta, epsilon, divMode, outputRstd),
                   DFX_OUT(y, rstd));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数校验
    auto paramRet = CheckParams(x, gamma, scale, y, outputRstd, rstd);
    CHECK_RET(paramRet == ACLNN_SUCCESS, paramRet);

    // 从输出y的dtype推导yType，INT32表示实际输出为INT4
    int32_t yType = static_cast<int32_t>(y->GetDataType());
    if (yType == op::DataType::DT_INT32) {
        yType = op::DataType::DT_INT4;
        OP_LOGD("aclnnRmsNormQuantV3 real output is int4.");
    }

    // 预处理gamma和beta的shape（2D→1D）
    const aclTensor* gammaProc = gamma;
    const aclTensor* betaProc = beta;
    auto preDealRet = PreDealData(gammaProc, betaProc, uniqueExecutor.get());
    CHECK_RET(preDealRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // 将输入转换成连续的tensor
    auto xCont = l0op::Contiguous(x, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gammaProc, uniqueExecutor.get());
    auto scaleCont = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(xCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(scaleCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 可选输入不做判空校验
    const aclTensor* offsetCont = (offset == nullptr) ? nullptr : l0op::Contiguous(offset, uniqueExecutor.get());
    const aclTensor* betaCont = (betaProc == nullptr) ? nullptr : l0op::Contiguous(betaProc, uniqueExecutor.get());

    auto ret = ComputeRmsNormQuantV3(xCont, gammaCont, scaleCont, offsetCont, betaCont, epsilon, yType, divMode,
                                     outputRstd, y, rstd, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnRmsNormQuantV3GetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRmsNormQuantV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRmsNormQuantV3);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
