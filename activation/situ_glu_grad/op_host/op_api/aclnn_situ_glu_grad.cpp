/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_situ_glu_grad.h"
#include "situ_glu_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"
#include "base/err_msg.h"
#include <string>
#include <vector>

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;
constexpr int64_t SPLIT_NUM = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                   DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* gradY, const aclTensor* x, const aclTensor* out)
{
    // gradY、x、out不能为空指针
    OP_CHECK_NULL(gradY, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* gradY, const aclTensor* x, const aclTensor* out)
{
    // 检查gradY的数据类型是否在SituGluGrad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradY, DTYPE_SUPPORT_LIST, return false);

    // x和out的数据类型必须与gradY一致
    OP_CHECK_DTYPE_NOT_SAME(gradY, x, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradY, out, return false);

    return true;
}

static inline bool CheckDimInRange(int64_t dim, int64_t xDimNum)
{
    if (dim >= xDimNum || dim < -xDimNum) {
        const std::string reason = ("expected dim to be in range of [" + std::to_string(-xDimNum) + ", " +
                                    std::to_string(xDimNum - 1) + "], but got " + std::to_string(dim) + ".");
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0026", (std::vector<const char*>{"param_name", "op_name", "incorrect_value", "reason"}),
            (std::vector<const char*>{"dim", "aclnnSituGluGrad", (std::to_string(dim)).c_str(), reason.c_str()}));
        return false;
    }
    return true;
}

static inline bool CheckShape(const aclTensor* gradY, const aclTensor* x, const aclTensor* out, int64_t dim)
{
    OP_CHECK_MAX_DIM(gradY, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(x, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    OP_CHECK_MIN_DIM(x, 1, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(out, x, return false);

    int64_t xDimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    if (!CheckDimInRange(dim, xDimNum)) {
        return false;
    }
    int64_t positiveDim = dim < 0 ? dim + xDimNum : dim;

    int64_t splitShape = x->GetViewShape().GetDim(static_cast<size_t>(positiveDim));
    if (splitShape % SPLIT_NUM != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Halving dimension must be even, but dimension %ld is size %ld.", dim,
                splitShape);
        return false;
    }

    // gradY的shape必须为x在dim维度减半、其余维度与x一致
    op::Shape gradYShapeExpect = x->GetViewShape();
    gradYShapeExpect.SetDim(static_cast<size_t>(positiveDim), splitShape / SPLIT_NUM);
    if (gradYShapeExpect != gradY->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of gradY must be %s, but got %s.",
                op::ToString(gradYShapeExpect).GetString(), op::ToString(gradY->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradY, const aclTensor* x, const aclTensor* out, int64_t dim)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradY, x, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否准确
    CHECK_RET(CheckDtypeValid(gradY, x, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape与dim取值是否准确
    CHECK_RET(CheckShape(gradY, x, out, dim), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecSituGluGradGetWorkspaceSize(const aclTensor* gradY, const aclTensor* x, int64_t dim, double beta,
                                                   double linearBeta, bool activateLeft, const aclTensor* out,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradY, x, out, dim);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果x是空tensor，则out也是空tensor，直接返回
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradY转换成连续的Tensor
    auto contiguousGradY = l0op::Contiguous(gradY, uniqueExecutor.get());
    CHECK_RET(contiguousGradY != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入x转换成连续的Tensor
    auto contiguousX = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(contiguousX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用SituGluGrad算子kernel，结果写入连续临时tensor
    auto situGluGradOut = l0op::SituGluGrad(contiguousGradY, contiguousX, dim, static_cast<float>(beta),
                                            static_cast<float>(linearBeta), activateLeft, uniqueExecutor.get());
    CHECK_RET(situGluGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(situGluGradOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSituGluGradGetWorkspaceSize(const aclTensor* gradY, const aclTensor* x, int64_t dim, double beta,
                                             double linearBeta, bool activateLeft, const aclTensor* out,
                                             uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnSituGluGrad, DFX_IN(gradY, x, dim, beta, linearBeta, activateLeft), DFX_OUT(out));
    return ExecSituGluGradGetWorkspaceSize(gradY, x, dim, beta, linearBeta, activateLeft, out, workspaceSize, executor);
}

aclnnStatus aclnnSituGluGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSituGluGrad);

    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
