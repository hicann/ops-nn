/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_situ_mx_quant.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"
#include "aclnnInner_situ_mx_quant.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t MIN_DIM_NUM = 1;
constexpr size_t MAX_DIM_NUM = 8;
constexpr int64_t SPLIT_NUM = 2;
// y_scale: 每(align_num * block_size)=64个y元素共享一组E8M0 scale
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t SCALE_ALIGN_NUM = 2;
constexpr int64_t SCALE_GROUP_SIZE = MX_BLOCK_SIZE * SCALE_ALIGN_NUM;
constexpr int64_t DTYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DTYPE_FLOAT8_E4M3FN = 36;
constexpr int64_t AXIS_LAST_DIM = -1;

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> Y_SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};

static bool CheckNotNull(const aclTensor* x, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(yScaleOut, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, int64_t dstType, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(yScaleOut, Y_SCALE_DTYPE_SUPPORT_LIST, return false);

    // dstType的取值范围已在CheckScalarParams中校验，此处仅校验yOut与dstType的匹配关系
    const op::DataType expectedYDtype = (dstType == DTYPE_FLOAT8_E5M2) ? op::DataType::DT_FLOAT8_E5M2 :
                                                                         op::DataType::DT_FLOAT8_E4M3FN;
    OP_CHECK_DTYPE_NOT_MATCH(yOut, expectedYDtype, return false);
    return true;
}

static bool CheckScalarParams(int64_t axis, int64_t dstType, double beta, const char* roundModeOptional)
{
    if (axis != AXIS_LAST_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "axis only supports -1 currently, but got %ld.", axis);
        return false;
    }
    if (dstType != DTYPE_FLOAT8_E5M2 && dstType != DTYPE_FLOAT8_E4M3FN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dstType only supports 35(FLOAT8_E5M2) or 36(FLOAT8_E4M3FN), but got %ld.",
                dstType);
        return false;
    }
    if (beta <= 0.0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "beta must be greater than 0, but got %f.", beta);
        return false;
    }
    // nullptr按默认值"rint"处理
    if (roundModeOptional != nullptr && std::string(roundModeOptional) != "rint") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roundModeOptional only supports \"rint\" for FP8 output, but got %s.",
                roundModeOptional);
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    // torch_npu对3/4/5维连续Tensor会推导NCL/NCHW/NCDHW格式标签，其连续存储与ND字节布局一致，
    // 属于等效线性格式，均需放行；分形等非线性存储格式仍拦截
    const auto isLinearFormat = [](op::Format format) {
        return format == op::Format::FORMAT_ND || format == op::Format::FORMAT_NCL ||
               format == op::Format::FORMAT_NCHW || format == op::Format::FORMAT_NCDHW;
    };
    if (!isLinearFormat(x->GetStorageFormat()) || !isLinearFormat(yOut->GetStorageFormat()) ||
        !isLinearFormat(yScaleOut->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x, yOut and yScaleOut only support linear formats (ND/NCL/NCHW/NCDHW), but got x: %s, yOut: %s, "
                "yScaleOut: %s.",
                op::ToString(x->GetStorageFormat()).GetString(), op::ToString(yOut->GetStorageFormat()).GetString(),
                op::ToString(yScaleOut->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool HasUnknownDim(const op::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

static bool CheckShape(const aclTensor* x, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    if (x->IsEmpty() || yOut->IsEmpty() || yScaleOut->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x, yOut and yScaleOut do not support empty tensor, x: %s, yOut: %s, "
                "yScaleOut: %s.",
                op::ToString(x->GetViewShape()).GetString(), op::ToString(yOut->GetViewShape()).GetString(),
                op::ToString(yScaleOut->GetViewShape()).GetString());
        return false;
    }

    OP_CHECK_MIN_DIM(x, MIN_DIM_NUM, return false);
    OP_CHECK_MAX_DIM(x, MAX_DIM_NUM, return false);

    const op::Shape& xShape = x->GetViewShape();
    const int64_t lastDim = xShape.GetDim(xShape.GetDimNum() - 1);
    if (lastDim % SPLIT_NUM != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dimension of x must be divisible by 2, but got %ld.", lastDim);
        return false;
    }

    const int64_t hDim = lastDim / SPLIT_NUM;
    op::Shape expectedYShape;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        expectedYShape.AppendDim(xShape.GetDim(i));
    }
    expectedYShape.AppendDim(hDim);

    op::Shape expectedYScaleShape = expectedYShape;
    const int64_t scaleNum = (hDim + SCALE_GROUP_SIZE - 1) / SCALE_GROUP_SIZE;
    expectedYScaleShape.SetDim(expectedYScaleShape.GetDimNum() - 1, scaleNum);
    expectedYScaleShape.AppendDim(SCALE_ALIGN_NUM);

    // 含未知维(-1/-2)的动态shape无法精确比较，交由执行阶段推导校验
    if (!HasUnknownDim(xShape)) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(yOut, expectedYShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(yScaleOut, expectedYScaleShape, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, double beta, int64_t axis, int64_t dstType, char* roundModeOptional,
                               const aclTensor* yOut, const aclTensor* yScaleOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, yOut, yScaleOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查host侧标量参数：axis/dstType/beta/roundModeOptional
    CHECK_RET(CheckScalarParams(axis, dstType, beta, roundModeOptional), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入/输出的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, dstType, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查format是否符合要求
    CHECK_RET(CheckFormat(x, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查shape/空tensor是否符合约束
    CHECK_RET(CheckShape(x, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnSituMxQuantGetWorkspaceSize(const aclTensor* x, double beta, double linearBeta, bool activateLeft,
                                             int64_t axis, int64_t dstType, char* roundModeOptional,
                                             const aclTensor* yOut, const aclTensor* yScaleOut, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto ret = CheckParams(x, beta, axis, dstType, roundModeOptional, yOut, yScaleOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto yOutX = const_cast<aclTensor*>(yOut);
    auto yScaleOutX = const_cast<aclTensor*>(yScaleOut);
    return aclnnInnerSituMxQuantGetWorkspaceSize(x, beta, linearBeta, activateLeft, axis, dstType, roundModeOptional,
                                                 yOutX, yScaleOutX, workspaceSize, executor);
}

aclnnStatus aclnnSituMxQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerSituMxQuant(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
