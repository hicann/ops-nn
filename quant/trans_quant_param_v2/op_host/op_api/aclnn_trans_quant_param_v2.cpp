/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>
#include <dlfcn.h>

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_trans_quant_param_v3.h"
#include "aclnn_trans_quant_param_v2.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerTransQuantParamV2GetWorkspaceSize(
    const aclTensor* scale, const aclTensor* offset, int64_t roundMode, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerTransQuantParamV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64};

namespace {
static inline bool CheckNotNull(const aclTensor* scale, const aclTensor* out)
{
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* scale, const aclTensor* offset, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, SCALE_TYPE_SUPPORT_LIST, return false);
    if (offset != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(offset, op::DataType::DT_FLOAT, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_TYPE_SUPPORT_LIST, return false);
    return true;
}
static inline bool CheckFormat(const aclTensor* scale, const aclTensor* offset, const aclTensor* out)
{
    bool formatValid =
        scale->GetStorageFormat() == op::Format::FORMAT_ND && out->GetStorageFormat() == op::Format::FORMAT_ND;
    if (offset != nullptr) {
        formatValid = formatValid && offset->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    return formatValid;
}

static inline bool CheckShape(const aclTensor* scale, const aclTensor* offset, int64_t roundMode)
{
    bool isScaleVaild = (scale->GetViewShape().GetDimNum() == 1 && scale->GetViewShape().GetDim(0) > 0);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        // 2是scale的shape维度，此时shape为(g, n), (1, n), g为gmm的分组数，n为matmul的右矩阵的n
        if (!(isScaleVaild || (scale->GetViewShape().GetDimNum() == 2 && scale->GetViewShape().GetDim(0) > 0 &&
                               scale->GetViewShape().GetDim(1) > 0))) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Dim value of scale should be large than 0, but current is %s",
                op::ToString(scale->GetViewShape()).GetString());
            return false;
        }
    } else {
        // 2是scale的shape维度，此时shape为(1, n)，n为matmul的右矩阵的n
        if (!(isScaleVaild || (scale->GetViewShape().GetDimNum() == 2 && scale->GetViewShape().GetDim(0) == 1 &&
                               scale->GetViewShape().GetDim(1) > 0))) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "shape of scale only support (n, ) and (1, n), but current is %s",
                op::ToString(scale->GetViewShape()).GetString());
            return false;
        }
    }
    if (offset != nullptr) {
        bool isOffsetVaild = (offset->GetViewShape().GetDimNum() == 1 && offset->GetViewShape().GetDim(0) > 0) ||
                             (offset->GetViewShape().GetDimNum() == 2 && offset->GetViewShape().GetDim(0) == 1 &&
                              offset->GetViewShape().GetDim(1) > 0);
        if (isOffsetVaild == false) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "shape of offset only support (n, ) and (1, n), but current is %s",
                op::ToString(offset->GetViewShape()).GetString());
            return false;
        }
        if (offset->GetViewShape().GetDimNum() == 1) {
            int64_t scaleDimValue = scale->GetViewShape().GetDim(0);
            int64_t offsetDimValue = offset->GetViewShape().GetDim(0);
            if (scaleDimValue > 1 && offsetDimValue > 1 && scaleDimValue != offsetDimValue) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "scale and offset should be same when both greater than 1, but scale is %ld, offset is %ld",
                    scaleDimValue, offsetDimValue);
                return false;
            }
        }
    }
    if (roundMode != 0 && roundMode != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roundMode only support 0 and 1, but current is %ld", roundMode);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* scale, const aclTensor* offset, int64_t roundMode, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(scale, out), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(scale, offset, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查format是否符合要求
    CHECK_RET(CheckFormat(scale, offset, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否符合要求
    CHECK_RET(CheckShape(scale, offset, roundMode), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(
    const aclTensor* scale, const aclTensor* offset, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    auto ret = CheckParams(scale, offset, 0, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto outX = const_cast<aclTensor*>(out);
    outX->SetDataType(op::DataType::DT_UINT64);
    return aclnnInnerTransQuantParamV2GetWorkspaceSize(scale, offset, 0, outX, workspaceSize, executor);
}

aclnnStatus aclnnTransQuantParamV3GetWorkspaceSize(
    const aclTensor* scale, const aclTensor* offset, int64_t roundMode, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    auto ret = CheckParams(scale, offset, roundMode, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto outX = const_cast<aclTensor*>(out);
    outX->SetDataType(op::DataType::DT_UINT64);
    return aclnnInnerTransQuantParamV2GetWorkspaceSize(scale, offset, roundMode, outX, workspaceSize, executor);
}

aclnnStatus aclnnTransQuantParamV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    return aclnnInnerTransQuantParamV2(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnTransQuantParamV3(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    return aclnnInnerTransQuantParamV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif