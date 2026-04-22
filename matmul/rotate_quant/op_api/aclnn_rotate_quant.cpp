/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_rotate_quant.h"
#include "rotate_quant.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
namespace {
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr int64_t DIM_MIN = 2;
static constexpr int64_t N_DIM_ALIGNMENT = 8;

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT8};

struct RotateQuantParams {
    const aclTensor* x = nullptr;
    const aclTensor* rot = nullptr;
    int64_t dstType;
    const aclTensor* y = nullptr;
    const aclTensor* scale = nullptr;
};

static aclnnStatus CheckNotNull(const RotateQuantParams& params)
{
    CHECK_COND(params.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(params.rot != nullptr, ACLNN_ERR_PARAM_NULLPTR, "rot must not be nullptr.");
    CHECK_COND(params.y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y must not be nullptr.");
    CHECK_COND(params.scale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const RotateQuantParams& params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.x, INPUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.rot, INPUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(
        params.x->GetDataType() == params.rot->GetDataType(), ACLNN_ERR_PARAM_INVALID,
        "The dtypes of x and rot are not equal.");

    OP_CHECK_DTYPE_NOT_SUPPORT(params.y, OUTPUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_DTYPE_NOT_MATCH(params.scale, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);

    auto yDtype = params.y->GetDataType();
    if (static_cast<int64_t>(yDtype) != params.dstType) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "dstType (%ld) does not match y dtype (%d).", params.dstType,
            static_cast<int64_t>(yDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const RotateQuantParams& params)
{
    auto xShape = params.x->GetViewShape();
    auto rotShape = params.rot->GetViewShape();
    auto yShape = params.y->GetViewShape();
    auto scaleShape = params.scale->GetViewShape();
    auto yDtype = params.y->GetDataType();
    auto xDimNum = xShape.GetDimNum();
    CHECK_COND(xDimNum == DIM_MIN, ACLNN_ERR_PARAM_INVALID, "The dimNum[%lu] of x should be 2.", xDimNum);

    auto rotDimNum = rotShape.GetDimNum();
    CHECK_COND(rotDimNum == DIM_MIN, ACLNN_ERR_PARAM_INVALID, "The dimNum[%lu] of rot should be 2.", rotDimNum);

    int64_t xM = xShape.GetDim(0);     // M
    int64_t xN = xShape.GetDim(1);     // N
    int64_t rotK = rotShape.GetDim(0); // K
    int64_t rotK2 = rotShape.GetDim(1); // K

    CHECK_COND(rotK == rotK2, ACLNN_ERR_PARAM_INVALID, "The rotation matrix must be a square matrix.");

    CHECK_COND(yShape.GetDimNum() == DIM_MIN, ACLNN_ERR_PARAM_INVALID, "The dimNum of y should be 2.");
    CHECK_COND(
        yShape.GetDim(0) == xM, ACLNN_ERR_PARAM_INVALID, "The first dim of y (%ld) should equal M (%ld).",
        yShape.GetDim(0), xM);

    CHECK_COND(rotK != 0, ACLNN_ERR_PARAM_INVALID, "rotK cannot be zero.");
    CHECK_COND(xN % rotK == 0, ACLNN_ERR_PARAM_INVALID, "N must be an integer multiple of K.");
    CHECK_COND(xN % N_DIM_ALIGNMENT == 0, ACLNN_ERR_PARAM_INVALID, "N must be divisible by alignment size.");
    CHECK_COND(scaleShape.GetDimNum() == 1, ACLNN_ERR_PARAM_INVALID, "The dimNum of scale should be 1.");
    CHECK_COND(
        scaleShape.GetDim(0) == xM, ACLNN_ERR_PARAM_INVALID, "The first dim of scale (%ld) should equal M (%ld).",
        scaleShape.GetDim(0), xM);
    if (yDtype == op::DataType::DT_INT32) {
        CHECK_COND(
            yShape.GetDim(1) == xN / N_DIM_ALIGNMENT, ACLNN_ERR_PARAM_INVALID,
            "When yDtype is INT32, the second dim of y (%ld) should equal N / 8 (%ld).", yShape.GetDim(1), xN / N_DIM_ALIGNMENT);
    } else if (yDtype == op::DataType::DT_INT8) {
        CHECK_COND(
            yShape.GetDim(1) == xN, ACLNN_ERR_PARAM_INVALID,
            "When yDtype is INT8, the second dim of y (%ld) should equal N (%ld).", yShape.GetDim(1), xN);
    }

    return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckParams(RotateQuantParams& params)
{
    CHECK_COND(
        CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR, "One of the required inputs is nullptr.");
    CHECK_COND(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dtype.");
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus Int42Int32PackedTensor(const aclTensor* out, const aclTensor*& outTensor, aclOpExecutor* executor)
{
    auto viewShape = out->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(out, viewShape, out->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    return ACLNN_SUCCESS;
}

}; // namespace

aclnnStatus aclnnRotateQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* rot, float alpha, aclTensor* yOut, aclTensor* scaleOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(rot != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(yOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(scaleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto yDtype = yOut->GetDataType();
    auto dstDtype = yOut->GetDataType();
    L2_DFX_PHASE_1(aclnnRotateQuant, DFX_IN(x, rot, alpha), DFX_OUT(yOut, scaleOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    RotateQuantParams params{x, rot, yDtype, yOut, scaleOut};
    aclnnStatus ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || rot->IsEmpty() || yOut->IsEmpty() || scaleOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* rotContiguous = l0op::Contiguous(rot, uniqueExecutor.get());
    CHECK_RET(rotContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* y = nullptr;

    if (yDtype == op::DataType::DT_INT32) {
        yDtype = op::DataType::DT_INT4;
        OP_LOGD("op dynamicquant real output is int4.");
    }

    auto result = l0op::RotateQuant(xContiguous, rotContiguous, yDtype, uniqueExecutor.get());
    y = std::get<0>(result);
    auto scale = std::get<1>(result);
    CHECK_RET(y != nullptr && scale != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (dstDtype == op::DataType::DT_INT32) {
        ret = Int42Int32PackedTensor(y, y, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    auto ret0 = l0op::ViewCopy(y, yOut, uniqueExecutor.get());
    CHECK_RET(ret0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto ret1 = l0op::ViewCopy(scale, scaleOut, uniqueExecutor.get());
    CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRotateQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRotateQuant);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in RotateQuant launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
