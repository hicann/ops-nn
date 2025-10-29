/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_ascend_quant_v3.h"
#include "ascend_quant_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

static constexpr size_t MAX_DIM_LEN = 8;
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr uint64_t INT4_NUMS_IN_INT8_SPACE = 2;
static constexpr int32_t MAX_AXIS_VALUE = 2;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_ASCEND310P = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_INT4};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_ASCEND910_95 = {
    op::DataType::DT_INT8,          op::DataType::DT_INT32,       op::DataType::DT_INT4,
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_ASCEND310P = {op::DataType::DT_INT8};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<DataType> SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType> SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND310P = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType>& GetInDtypeSupportList()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_95: {
            return X_DTYPE_SUPPORT_LIST_ASCEND910B;
        }
        case SocVersion::ASCEND310P:
            return X_DTYPE_SUPPORT_LIST_ASCEND310P;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

static const std::initializer_list<DataType>& GetOutDtypeSupportList()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND310P) {
        return OUT_DTYPE_SUPPORT_LIST_ASCEND310P;
    } else if (socVersion == SocVersion::ASCEND910_95) {
        return OUT_DTYPE_SUPPORT_LIST_ASCEND910_95;
    } else {
        return OUT_DTYPE_SUPPORT_LIST_ASCEND910B;
    }
}

static const std::initializer_list<DataType>& GetScaleOffsetDtypeSupportList()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910_95: {
            return SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND910B;
        }
        case SocVersion::ASCEND310P:
            return SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND310P;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* scale, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeCompatible(const aclTensor* x, const aclTensor* scale)
{
    if (x->GetDataType() == op::DataType::DT_FLOAT) {
        return true;
    } else if (x->GetDataType() == op::DataType::DT_FLOAT16) {
        if (scale->GetDataType() != op::DataType::DT_FLOAT16) {
            return false;
        }
    } else if (x->GetDataType() == op::DataType::DT_BF16) {
        if (scale->GetDataType() != op::DataType::DT_BF16) {
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x, const aclTensor* scale, const aclTensor* offset, const aclTensor* y, int32_t dstType)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, GetInDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, GetOutDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, GetScaleOffsetDtypeSupportList(), return false);
    if (offset != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(offset, GetScaleOffsetDtypeSupportList(), return false);
        OP_CHECK_DTYPE_NOT_SAME(scale, offset, return false);
    }
    if (!CheckDtypeCompatible(x, scale)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "dtype of input x:%s is not compatible with scale:%s.",
            op::ToString(x->GetDataType()).GetString(), op::ToString(scale->GetDataType()).GetString());
        return false;
    }

    if (static_cast<int32_t>(y->GetDataType()) != dstType) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "dstType:%d(%s) must be able to represent the y dtype[%s].", dstType,
            op::ToString(static_cast<op::DataType>(dstType)).GetString(), op::ToString(y->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckDim(const aclTensor* x, const aclTensor* scale, const aclTensor* offset, int32_t axis)
{
    if (offset != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(scale, offset, return false);
    }
    // x and scale shape
    auto xShape = x->GetViewShape();
    auto scaleShape = scale->GetViewShape();
    // x and scale dim num
    int64_t xDimNum = static_cast<int64_t>(xShape.GetDimNum());
    int64_t scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
    // check scale scalar
    if (scaleDimNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale and offset not support scalar.");
        return false;
    }

    if (scaleDimNum != 1 && scaleDimNum != xDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale dimNum must be same as x dimNum or 1.");
        return false;
    }

    // x and scale real axis
    int64_t xAxis = (axis >= 0 ? axis : xDimNum + axis);
    int64_t scaleAxis = (scaleDimNum == 1 ? 0 : xAxis);
    // x and scale axis dim
    int64_t xDim = xShape.GetDim(xAxis);
    int64_t scaleDim = scaleShape.GetDim(scaleAxis);
    if (scaleDim != xDim && scaleDim != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "scale dim(%ld)'s value(%ld) is invalid, should be same as x dim(%ld)'s value(%ld) or 1", scaleAxis,
            scaleDim, xAxis, xDim);
        return false;
    }

    int64_t scaleTotalSize = scaleShape.GetShapeSize();
    if (scaleTotalSize != scaleDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "If scale is not 1-dimensional tensor, all dimensions except the dimension specified by axis should be 1");
        return false;
    }

    return true;
}

static bool CheckInt32OutputShape(const aclTensor* x, const aclTensor* y)
{
    int64_t dimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    if (dimNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x not support scalar.");
        return false;
    }
    int64_t dimInput = x->GetViewShape().GetDim(dimNum - 1);
    int64_t dimOutput = y->GetViewShape().GetDim(dimNum - 1);
    // check last dim
    if (dimInput != (dimOutput * INT4_NUMS_IN_INT32_SPACE)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "if y dtype is int32, the y last dim must be 1/8 of x last dim, x last dim is (%ld), y last dim is (%ld)",
            dimInput, dimOutput);
        return false;
    }
    // check others dim
    for (int64_t i = 0; i < dimNum - 1; ++i) {
        dimInput = x->GetViewShape().GetDim(i);
        dimOutput = y->GetViewShape().GetDim(i);
        if (dimInput != dimOutput) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "the dim(%ld) of x must be same with y, x is (%ld), y is (%ld).", i, dimInput,
                dimOutput);
            return false;
        }
    }

    return true;
}

static bool CheckInt4LastDim(const aclTensor* x)
{
    int64_t dimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    if (dimNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x not support scalar.");
        return false;
    }
    int64_t lastDimInput = x->GetViewShape().GetDim(dimNum - 1);
    if (lastDimInput % INT4_NUMS_IN_INT8_SPACE) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "if y dtype is int4, x last dim must be divisible by 2, last dim is (%ld).",
            lastDimInput);
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* x, const aclTensor* y, const aclTensor* scale, const aclTensor* offset, int32_t axis)
{
    // x和y的shape必须一致
    if (y->GetDataType() != op::DataType::DT_INT32) {
        OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
    }

    // check input int32
    if (y->GetDataType() == op::DataType::DT_INT32 && !CheckInt32OutputShape(x, y)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check int32 output shape failed.");
        return false;
    }

    // check input int4
    if (y->GetDataType() == op::DataType::DT_INT4 && !CheckInt4LastDim(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "check int4 output shape failed.");
        return false;
    }

    // x的数据维度不能超过8
    OP_CHECK_MAX_DIM(x, MAX_DIM_LEN, return false);

    // check x scale offset last dim size
    CHECK_RET(CheckDim(x, scale, offset, axis), false);

    return true;
}

static bool CheckRoundMode(const char* roundMode, int32_t dstType)
{
    if (roundMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roundMode cannot be empty");
        return false;
    }
    const std::string mode = std::string(roundMode);
    if (dstType == op::DataType::DT_HIFLOAT8) {
        if (mode != "round" && mode != "hybrid") {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "check roundMode failed, roundMode[%s] not in ['round','hybrid'] for hifloat8.", mode.c_str());
            return false;
        }
    } else if (dstType == op::DataType::DT_FLOAT8_E4M3FN || dstType == op::DataType::DT_FLOAT8_E5M2) {
        if (mode != "round" && mode != "hybrid") {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "check roundMode failed, roundMode[%s] not in ['round', 'hybrid'] for float8_e5m2/float8_e4m3fn.",
                mode.c_str());
            return false;
        }
    } else {
        if (mode != "round" && mode != "floor" && mode != "ceil" && mode != "trunc") {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "check roundMode failed, roundMode[%s] not in ['round','floor','ceil','trunc'] for int8/int4/int32.",
                mode.c_str());
            return false;
        }
    }
    return true;
}

static bool CheckAxis(const aclTensor* x, int32_t axis)
{
    int32_t xDimNum = static_cast<int32_t>(x->GetViewShape().GetDimNum());
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        if (axis != -1 && axis != xDimNum - 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "axis only support the last dimension in 310P");
            return false;
        }
    }
    if (axis >= xDimNum || axis < -xDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invalid axis: %d, axis can't exceed the dimension of x: %d", axis, xDimNum);
        return false;
    }
    if (xDimNum > MAX_AXIS_VALUE) {
        if (axis < -MAX_AXIS_VALUE || (axis >= 0 && axis < xDimNum - MAX_AXIS_VALUE)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invalid axis: %d, axis must be one of last two dimensions of x", axis);
            return false;
        }
    }
    return true;
}

static aclnnStatus Int42Int32PackedTensor(
    const aclTensor* y, const aclTensor*& outTensor, const int32_t& dstType, aclOpExecutor* executor)
{
    if (dstType != op::DataType::DT_INT32) {
        OP_LOGD("current aclnnAscendQuantV3 output does not need to pack.");
        return ACLNN_SUCCESS;
    }
    // if dstType is int32, pack output
    auto viewShape = y->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    OP_LOGD("aclnnAscendQuantV3 output real dtype is int4, pack to int32 to y.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputScalar(const aclTensor* x)
{
    int64_t dimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    if (dimNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x not support scalar.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclTensor* scale, const aclTensor* offset, const char* roundMode, int32_t dstType,
    int32_t axis, const aclTensor* y)
{
    CHECK_RET(CheckInputScalar(x), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckNotNull(x, scale, y), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(x, scale, offset, y, dstType), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckAxis(x, axis), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(x, y, scale, offset, axis), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckRoundMode(roundMode, dstType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(
    const aclTensor* x, const aclTensor* scale, const aclTensor* offset, bool sqrtMode, const char* roundMode,
    int32_t dstType, int32_t axis, const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAscendQuantV3, DFX_IN(x, scale, offset, roundMode, sqrtMode, dstType), DFX_OUT(y));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, scale, offset, roundMode, dstType, axis, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (x->IsEmpty()) {
        *workspaceSize = 0U;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scaleContiguous = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto scaleCast = l0op::Cast(scaleContiguous, x->GetDataType(), uniqueExecutor.get());
    CHECK_RET(scaleCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* offsetCast = nullptr;
    const aclTensor* offsetContiguous = nullptr;
    if (offset != nullptr) {
        offsetContiguous = l0op::Contiguous(offset, uniqueExecutor.get());
        CHECK_RET(offsetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        offsetCast = l0op::Cast(offsetContiguous, x->GetDataType(), uniqueExecutor.get());
        CHECK_RET(offsetCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        offsetCast = offset;
    }

    // 输出dtype支持int8/int4, int32是int4伪装，实际需要按照int4计算
    auto yDtype = y->GetDataType();
    if (yDtype == op::DataType::DT_INT32) {
        yDtype = op::DataType::DT_INT4;
        OP_LOGD("aclnnAscendQuantV3 real output is int4.");
    }
    auto ascendQuantV2Result = l0op::AscendQuantV2(
        selfContiguous, scaleCast, offsetCast, sqrtMode, roundMode, yDtype, axis, uniqueExecutor.get());
    CHECK_RET(ascendQuantV2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果输出是Int4，需要转成Int32，8个Int4输出拼成1个Int32
    const aclTensor* outTensor = ascendQuantV2Result;
    ret = Int42Int32PackedTensor(ascendQuantV2Result, outTensor, dstType, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(outTensor, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAscendQuantV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAscendQuantV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}