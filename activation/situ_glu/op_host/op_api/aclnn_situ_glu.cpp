/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_situ_glu.cpp
 * \brief
 */
#include "aclnn_situ_glu.h"

#include "situglu.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include <cmath>
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* SituGlu 算子的完整计算流程如下:
 *       x                    dim/beta/linearBeta/activateLeft
 *        |                         |
 *         \                       /
 *   Contiguous(workspace_0)      /
 *                  \           /
 *              SituGlu(workspace_1)
 *                     |
 *                 ViewCopy
 *                     |
 *                    out
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* x, const aclTensor* out, const uint64_t* workspaceSize,
                         aclOpExecutor** executor)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    // 检查x的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

    // 检查out的数据类型是否与x一致
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* x, int64_t dim, const aclTensor* out)
{
    // x的维度必须大于0
    OP_CHECK_MIN_DIM(x, 1, return false);

    // 入参dim超出了x的shape可选维度范围[-x.dim，x.dim-1]
    int64_t xDimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    if (dim < -xDimNum || dim >= xDimNum) {
        const std::string reason = (("Input dimension out of range (expected to be in range of [" +
                                     std::to_string(-xDimNum) + ", " + std::to_string((xDimNum - 1)) + "], but got " +
                                     std::to_string(dim) + ")."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0026", (std::vector<const char*>{"param_name", "op_name", "incorrect_value", "reason"}),
            (std::vector<const char*>{"dim", "aclnnSituGlu", (std::to_string(dim)).c_str(), reason.c_str()}));
        return false;
    }

    // 获取dim的非负数维度值，例如xDimNum=3时(-3,-2,-1) -> (0,1,2)
    int64_t positiveDim = dim;
    if (dim < 0) {
        positiveDim += xDimNum;
    }

    // 入参x根据指定的dim所对应的维度不能整除2
    int64_t sliceDimSize = x->GetViewShape().GetDim(static_cast<size_t>(positiveDim));
    const int64_t SLICE_NUM = 2;
    if (sliceDimSize % SLICE_NUM != 0) {
        const std::string reason = (("Split dimension must be even, but dimension " + std::to_string(dim) +
                                     " is size " + std::to_string(sliceDimSize) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0015", (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
            (std::vector<const char*>{"x", "aclnnSituGlu", (std::to_string(sliceDimSize)).c_str(), reason.c_str()}));
        return false;
    }

    // out的shape不等于x根据dim拆分后的shape
    op::Shape outShapeExpect = x->GetViewShape();
    int64_t outShapeExpectForDim = outShapeExpect.GetDim(static_cast<size_t>(positiveDim)) / SLICE_NUM;
    outShapeExpect.SetDim(static_cast<size_t>(positiveDim), outShapeExpectForDim);
    if (outShapeExpect != out->GetViewShape()) {
        const std::string reason = (("The shape of out must be " +
                                     std::string(op::ToString(outShapeExpect).GetString()) + ", but get " +
                                     std::string(op::ToString(out->GetViewShape()).GetString()) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0009", (std::vector<const char*>{"param_name", "op_name", "incorrect_shape", "reason"}),
            (std::vector<const char*>{"out", "aclnnSituGlu",
                                      (std::string(op::ToString(out->GetViewShape()).GetString())).c_str(),
                                      reason.c_str()}));
        return false;
    }

    return true;
}

static bool CheckAttrDomain(double beta, double linearBeta)
{
    // 算子内部按float32语义计算，属性域检查与kernel口径保持一致
    float betaFloat = static_cast<float>(beta);
    float linearBetaFloat = static_cast<float>(linearBeta);
    if (!std::isfinite(betaFloat) || betaFloat == 0.0f) {
        const std::string reason = (("beta must be a finite non-zero value, but got " + std::to_string(beta) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0026", (std::vector<const char*>{"param_name", "op_name", "incorrect_value", "reason"}),
            (std::vector<const char*>{"beta", "aclnnSituGlu", (std::to_string(beta)).c_str(), reason.c_str()}));
        return false;
    }
    if (!std::isfinite(linearBetaFloat)) {
        const std::string reason = (("linearBeta must be a finite value, but got " + std::to_string(linearBeta) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ0026",
                                  (std::vector<const char*>{"param_name", "op_name", "incorrect_value", "reason"}),
                                  (std::vector<const char*>{"linearBeta", "aclnnSituGlu",
                                                            (std::to_string(linearBeta)).c_str(), reason.c_str()}));
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, int64_t dim, double beta, double linearBeta, bool activateLeft,
                               const aclTensor* out, const uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, out, workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(x, dim, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查属性取值是否满足约束
    CHECK_RET(CheckAttrDomain(beta, linearBeta), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecSituGluGetWorkspaceSize(const aclTensor* x, int64_t dim, double beta, double linearBeta,
                                               bool activateLeft, const aclTensor* out, uint64_t* workspaceSize,
                                               aclOpExecutor** executor)
{
    // 固定写法，参数检查
    auto ret = CheckParams(x, dim, beta, linearBeta, activateLeft, out, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 如果为空tensor，则直接返回空
    if (x->IsEmpty()) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x如果非连续，需要转连续
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子SituGlu进行计算
    auto situGluResult = l0op::SituGlu(xContiguous, dim, static_cast<float>(beta), static_cast<float>(linearBeta),
                                       activateLeft, uniqueExecutor.get());
    CHECK_RET(situGluResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(situGluResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSituGluGetWorkspaceSize(const aclTensor* x, int64_t dim, double beta, double linearBeta,
                                         bool activateLeft, const aclTensor* out, uint64_t* workspaceSize,
                                         aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSituGlu, DFX_IN(x, dim, beta, linearBeta, activateLeft), DFX_OUT(out));
    return ExecSituGluGetWorkspaceSize(x, dim, beta, linearBeta, activateLeft, out, workspaceSize, executor);
}

aclnnStatus aclnnSituGlu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSituGlu);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
