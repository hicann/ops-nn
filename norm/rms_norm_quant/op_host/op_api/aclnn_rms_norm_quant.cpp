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
#include "aclnn_kernels/reshape.h"
#include "rms_norm_quant.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_rms_norm_quant.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const size_t DIMS_ONE_NUMS = 1;
static const size_t DIMS_TWO_NUMS = 2;
static const size_t MAX_DIM_LEN = 8;
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;

static const std::initializer_list<op::DataType> EXTEND_ATB_DTYPE_SUPPORT_LIST_X = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND310_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8, op::DataType::DT_INT4, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_ASCEND310_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8};
static const std::initializer_list<DataType> EMPTY_LIST = {};

struct RmsNormQuantInputTensor {
    const aclTensor* x;
    const aclTensor* gamma;
    const aclTensor* beta;
    const aclTensor* scale;
    const aclTensor* offset;
};

static bool CheckNotNull(RmsNormQuantInputTensor& inputTensor, aclTensor* y)
{
    OP_CHECK_NULL(inputTensor.x, return false);
    OP_CHECK_NULL(inputTensor.gamma, return false);
    OP_CHECK_NULL(inputTensor.beta, return false);
    OP_CHECK_NULL(inputTensor.scale, return false);
    OP_CHECK_NULL(inputTensor.offset, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static const std::initializer_list<DataType>& GetInDtypeSupportList()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910B: {
            return DTYPE_SUPPORT_LIST;
        }
        case SocVersion::ASCEND310P:
        case SocVersion::ASCEND310B: {
            return ASCEND310_DTYPE_SUPPORT_LIST;
        }
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

static const std::initializer_list<DataType>& GetOutDtypeSupportList()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910B: {
            return OUT_DTYPE_SUPPORT_LIST;
        }
        case SocVersion::ASCEND310P:
        case SocVersion::ASCEND310B: {
            return OUT_ASCEND310_DTYPE_SUPPORT_LIST;
        }
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "outputs support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

static bool CheckDtypeValid(RmsNormQuantInputTensor& inputTensor, const aclTensor* y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x, GetInDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensor.gamma, inputTensor.x, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensor.beta, inputTensor.x, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensor.scale, inputTensor.x, return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor.offset, op::DataType::DT_INT8, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, GetOutDtypeSupportList(), return false);
    return true;
}

aclnnStatus PreDealData(RmsNormQuantInputTensor& inputTensor, aclOpExecutor* executor)
{
    if (inputTensor.gamma->GetViewShape().GetDimNum() == DIMS_TWO_NUMS) {
        auto gammaReshape = l0op::Reshape(inputTensor.gamma, {inputTensor.gamma->GetViewShape()[1]}, executor);
        inputTensor.gamma = gammaReshape;
    }
    if (inputTensor.beta != nullptr && inputTensor.beta->GetViewShape().GetDimNum() == DIMS_TWO_NUMS) {
        auto betaReshape = l0op::Reshape(inputTensor.beta, {inputTensor.beta->GetViewShape()[1]}, executor);
        inputTensor.beta = betaReshape;
    }
    return ACLNN_SUCCESS;
}

static bool CheckShapeDim(RmsNormQuantInputTensor& inputTensor, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(inputTensor.x, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(inputTensor.gamma, DIMS_TWO_NUMS, return false);
    OP_CHECK_MAX_DIM(inputTensor.beta, DIMS_TWO_NUMS, return false);
    OP_CHECK_MAX_DIM(y, MAX_DIM_LEN, return false);

    OP_CHECK_MIN_DIM(inputTensor.x, DIMS_ONE_NUMS, return false);
    OP_CHECK_MIN_DIM(inputTensor.gamma, DIMS_ONE_NUMS, return false);
    OP_CHECK_MIN_DIM(inputTensor.beta, DIMS_ONE_NUMS, return false);
    OP_CHECK_MIN_DIM(y, DIMS_ONE_NUMS, return false);

    int64_t scaleDimNum = static_cast<int64_t>(inputTensor.scale->GetViewShape().GetDimNum());
    if (scaleDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DimNum of scale must be 1, but get [%ld]", scaleDimNum);
    }
    int64_t gammaDimNum = static_cast<int64_t>(inputTensor.gamma->GetViewShape().GetDimNum());
    int64_t gammaLastDim = inputTensor.gamma->GetViewShape().GetDim(gammaDimNum - 1);
    int64_t xDimNum = static_cast<int64_t>(inputTensor.x->GetViewShape().GetDimNum());
    int64_t xLastDim = inputTensor.x->GetViewShape().GetDim(xDimNum - 1);
    if (gammaLastDim != xLastDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "the last dim size(%ld) of x must be same as gamma and beta(%ld).", xLastDim,
            gammaLastDim);
        return false;
    }

    // check last dim
    auto outDtype = y->GetDataType();
    int64_t yDimNum = static_cast<int64_t>(y->GetViewShape().GetDimNum());
    int64_t outLastDim = y->GetViewShape().GetDim(yDimNum - 1);
    if (outDtype == op::DataType::DT_INT32) {
        OP_CHECK(xLastDim == outLastDim * INT4_NUMS_IN_INT32_SPACE,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "if outType is int32, the out last dim must be 1/8 of x last dim,"
                " x last dim is (%ld), out last dim is (%ld).",
                xLastDim,
                outLastDim),
            return false);
    } else {
        OP_CHECK(xLastDim == outLastDim,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "if outType is int4 or int8, x last dim must be equal with out last dim. "
                "x last dim is (%ld), out last dim is (%ld).",
                xLastDim,
                outLastDim),
            return false);
    }
    
    OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.gamma, inputTensor.beta, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.scale, inputTensor.offset, return false);
    return true;
}

static aclnnStatus CheckParams(RmsNormQuantInputTensor& inputTensor, aclTensor* y)
{
    // 1. 检查必选输入/输出是否为空指针
    CHECK_RET(CheckNotNull(inputTensor, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入/输出的数据类型是否合法
    CHECK_RET(CheckDtypeValid(inputTensor, y), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入/输出的shape大小
    CHECK_RET(CheckShapeDim(inputTensor, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus Int42Int32PackedTensor(
    const aclTensor *y, const aclTensor *&outTensor, int32_t outDtype, aclOpExecutor *executor)
{
    // if outType is int32, pack output
    auto viewShape = y->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    OP_LOGD("aclnnRmsNormQuant output real dtype is int4, pack to int32 to out.");

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRmsNormQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* scale, const aclTensor* offset,
    double epsilon, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnRmsNormQuant, DFX_IN(x, gamma, beta, scale, offset, epsilon), DFX_OUT(y));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 参数检查
    RmsNormQuantInputTensor inputTensorOri = {x, gamma, beta, scale, offset};
    PreDealData(inputTensorOri, uniqueExecutor.get());
    int32_t yType = y->GetDataType();
    if (yType == op::DataType::DT_INT32) {
        yType = op::DataType::DT_INT4;
        OP_LOGD("AclnnDynamicQuant real output is int4.");
    }
    auto ret = CheckParams(inputTensorOri, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto xCont = l0op::Contiguous(x, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());
    auto betaCont = l0op::Contiguous(beta, uniqueExecutor.get());
    auto offsetCont = l0op::Contiguous(offset, uniqueExecutor.get());
    auto scaleCont = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(xCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(betaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(offsetCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(scaleCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto resultTensor = l0op::RmsNormQuant(x, gamma, beta, scale, offset, epsilon, yType, uniqueExecutor.get());
    const aclTensor *outTensor = resultTensor;
    
    if (yType == op::DataType::DT_INT4) {
        ret = Int42Int32PackedTensor(resultTensor, outTensor, yType, uniqueExecutor.get());
        auto viewCopyY = l0op::ViewCopy(outTensor, y, uniqueExecutor.get());
        CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto viewCopyY = l0op::ViewCopy(resultTensor, y, uniqueExecutor.get());
        CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRmsNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRmsNormQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif