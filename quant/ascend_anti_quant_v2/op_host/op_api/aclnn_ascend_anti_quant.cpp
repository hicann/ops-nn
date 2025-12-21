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
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "ascend_anti_quant_v2.h"
#include "aclnn_ascend_anti_quant.h"

using namespace op;

static constexpr size_t MAX_DIM_LEN = 8;
static constexpr int64_t INT4_NUMS_IN_INT32 = 8;
static constexpr int64_t EVEN_FACTOR = 2;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_INT4, op::DataType::DT_INT8, op::DataType::DT_INT32};

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_ASCEND310P = {op::DataType::DT_INT8};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_ASCEND310P = {op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<DataType> SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<DataType> SCALE_OFFSET_DTYPE_SUPPORT_LIST_ASCEND310P = {op::DataType::DT_FLOAT};

static const std::initializer_list<DataType>& GetXDtypeSupportList()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910_95:
        case SocVersion::ASCEND910B: {
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
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910_95:
        case SocVersion::ASCEND910B: {
            return OUT_DTYPE_SUPPORT_LIST_ASCEND910B;
        }
        case SocVersion::ASCEND310P:
            return OUT_DTYPE_SUPPORT_LIST_ASCEND310P;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

static const std::initializer_list<DataType>& GetScaleOffsetDtypeSupportList()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_95:
        case SocVersion::ASCEND910_93: {
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

static bool CheckDtypeValid(
    const aclTensor* x, const aclTensor* scale, const aclTensor* offset, const aclTensor* y, int64_t dstType)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, GetXDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, GetOutDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, GetScaleOffsetDtypeSupportList(), return false);
    if (offset != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(offset, GetScaleOffsetDtypeSupportList(), return false);
        OP_CHECK_DTYPE_NOT_SAME(scale, offset, return false);
    }

    if (static_cast<int64_t>(y->GetDataType()) != dstType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out(y) data type must be the same as dstType.");
        return false;
    }

    return true;
}

static bool CheckDim(const aclTensor* y, const aclTensor* scale, const aclTensor* offset)
{
    int64_t scaleDim = scale->GetViewShape().GetDim(0);
    if (offset != nullptr) {
        int64_t offsetDim = offset->GetViewShape().GetDim(0);
        if (scaleDim != offsetDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim of scale must be the same as dim of offset");
            return false;
        }
    }

    // per-tensor
    if (scaleDim == 1) {
        return true;
    }

    int64_t dimNum = static_cast<int64_t>(y->GetViewShape().GetDimNum());
    if (dimNum == 0) {
        dimNum = 1;
    }

    int64_t lastDim = y->GetViewShape().GetDim(dimNum - 1);
    if (scaleDim != lastDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "the last dim size(%ld) of out(y) must be same as scale and offset(%ld).", lastDim,
            scaleDim);
        return false;
    }
    return true;
}

static bool CheckShapeForInt32(const aclTensor* x, const aclTensor* y)
{
    OP_LOGD("CheckShapeForInt32 begin");

    size_t dimNum = static_cast<size_t>(x->GetViewShape().GetDimNum());
    size_t yDimNum = static_cast<size_t>(y->GetViewShape().GetDimNum());
    if (dimNum == 0) {
        if (yDimNum == 1 && y->GetViewShape().GetDim(0) == INT4_NUMS_IN_INT32) {
            return true;
        } else {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "x dtype is int32. And x is scalar, y should be (8,). x shape:%s, y shape:%s",
                op::ToString(x->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
            return false;
        }
    }

    if (dimNum != yDimNum) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "dim num of x and y should be same. x shape:%s, y shape:%s",
            op::ToString(x->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
        return false;
    }
    for (size_t axis = 0; axis < dimNum - 1; ++axis) {
        if (x->GetViewShape().GetDim(axis) != y->GetViewShape().GetDim(axis)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "x dtype is int32. The shape of x and y should be same, expect for the size of the last axis."
                "x shape:%s, y shape:%s, axis:%zu",
                op::ToString(x->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString(), axis);
            return false;
        }
    }
    if (x->GetViewShape().GetDim(dimNum - 1) * INT4_NUMS_IN_INT32 != y->GetViewShape().GetDim(dimNum - 1)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "x dtype is int32. The last dim of y should be 8 times the last dim of x. x shape:%s, y shape:%s",
            op::ToString(x->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
        return false;
    }

    OP_LOGD("CheckShapeForInt32 end");
    return true;
}

static bool CheckShapeForInt4(const aclTensor* x, const aclTensor* y)
{
    OP_LOGD("CheckShapeForInt4 begin");

    OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);

    size_t dimNum = static_cast<size_t>(x->GetViewShape().GetDimNum());
    if (dimNum == 0 || x->GetViewShape().GetDim(dimNum - 1) % EVEN_FACTOR != 0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "x dtype is int4, the size of last dim must be an even number. x shape:%s, y shape:%s",
            op::ToString(x->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
        return false;
    }

    OP_LOGD("CheckShapeForInt4 end");
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y, const aclTensor* scale, const aclTensor* offset)
{
    OP_LOGD("CheckShape begin");

    if (x->GetDataType() == op::DataType::DT_INT4) {
        CHECK_RET(CheckShapeForInt4(x, y), false);
    } else if (x->GetDataType() == op::DataType::DT_INT32) {
        CHECK_RET(CheckShapeForInt32(x, y), false);
    } else {
        // x和y的shape必须一致
        OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
    }

    // x的数据维度不能超过8
    OP_CHECK_MAX_DIM(x, MAX_DIM_LEN, return false);
    if (scale->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale dim num must be 1. scale:%zu", scale->GetViewShape().GetDimNum());
        return false;
    }
    if (offset != nullptr) {
        if (offset->GetViewShape().GetDimNum() != 1) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "offset dim num must be 1. offset:%zu", offset->GetViewShape().GetDimNum());
            return false;
        }
        OP_CHECK_BROADCAST_WITH_SHAPE(offset, y->GetViewShape(), return false);
    }

    CHECK_RET(CheckDim(y, scale, offset), false);

    OP_CHECK_BROADCAST_WITH_SHAPE(scale, y->GetViewShape(), return false);

    OP_LOGD("CheckShape end");
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclTensor* scale, const aclTensor* offset, int64_t dstType, const aclTensor* y)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, scale, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x, scale, offset, y, dstType), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(x, y, scale, offset), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* TensorPreProcess(const aclTensor* x, aclOpExecutor* executor)
{
    OP_LOGD("TensorPreProcess begin");

    if (x->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("x is not int32. TensorPreProcess end");
        return x;
    }

    auto viewShape = x->GetViewShape();
    size_t dimNum = viewShape.GetDimNum();
    if (dimNum == 0) {
        dimNum = 1; // for x is scalar, unpack x to vector with shape is {8,}
        viewShape.SetDimNum(1);
        viewShape.SetDim(0, INT4_NUMS_IN_INT32);
    } else {
        viewShape[dimNum - 1] = viewShape[dimNum - 1] * INT4_NUMS_IN_INT32;
    }

    auto xTemp = executor->CreateView(x, viewShape, x->GetViewOffset());
    CHECK_RET(xTemp != nullptr, nullptr);

    xTemp->SetDataType(DataType::DT_INT4);

    OP_LOGD("TensorPreProcess end");
    return xTemp;
}

aclnnStatus aclnnAscendAntiQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scale, const aclTensor* offset, int64_t dstType, bool sqrtMode,
    const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAscendAntiQuant, DFX_IN(x, scale, offset, dstType, sqrtMode), DFX_OUT(y));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, scale, offset, dstType, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // input参数如果非连续，需要转连续
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scaleContiguous = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // int32 fix to int4
    const aclTensor* tensorX = TensorPreProcess(xContiguous, uniqueExecutor.get());
    CHECK_RET(tensorX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (offset != nullptr) {
        auto offsetContiguous = l0op::Contiguous(offset, uniqueExecutor.get());
        CHECK_RET(offsetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用l0算子AscendAntiQuantV2进行计算
        auto antiQuantV2Result = l0op::AscendAntiQuantV2(
            tensorX, scaleContiguous, offsetContiguous, dstType, sqrtMode, uniqueExecutor.get());
        CHECK_RET(antiQuantV2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto viewCopyResult = l0op::ViewCopy(antiQuantV2Result, y, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 调用l0算子AscendAntiQuantV2进行计算
        auto antiQuantV2Result =
            l0op::AscendAntiQuantV2(tensorX, scaleContiguous, offset, dstType, sqrtMode, uniqueExecutor.get());
        CHECK_RET(antiQuantV2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto viewCopyResult = l0op::ViewCopy(antiQuantV2Result, y, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAscendAntiQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAscendAntiQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}