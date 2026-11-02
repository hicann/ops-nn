/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shrink.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_shrink.h"

#include <cmath>

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static bool CheckDtypeValid(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    // lambd/bias 参数表声明为 FLOAT 标量，非 FLOAT dtype（如 int32）按文档须返回 161002，
    // 不可隐式转换参与计算（CanCast 会放行整型，不适用于此处的强类型约束）。
    if (lambd->GetDataType() != DataType::DT_FLOAT || bias->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "lambd and bias dtype must be float32, but lambd is [%s], bias is [%s].",
                ToString(lambd->GetDataType()).GetString(), ToString(bias->GetDataType()).GetString());
        return false;
    }
    // 公开契约（aclnnShrink.md/README）：out 的 dtype 必须与 self 一致，不一致返回 161002；
    // 不做隐式 Cast 降精度输出。
    if (self->GetDataType() != out->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dtype of self and out must be the same, but self is [%s], out is [%s].",
                ToString(self->GetDataType()).GetString(), ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckNotNull(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias,
                                const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(lambd, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}
static inline bool CheckLambdValue(const aclScalar* lambd)
{
    // 声明域为非负数（NaN 不是数，域外）；对齐仓内 shrink 族算子 soft_shrink_grad
    // 的 NaN+负值合并校验，NaN 或负值均返回 161002（原 NaN<0 为假会静默放行，
    // kernel Compares 恒假导致全死区输出全 0）。
    float lambdValue = lambd->ToFloat();
    if (std::isnan(lambdValue) || lambdValue < 0.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "lambd must not be NaN and should be greater or equal to 0, but found [%f].",
                lambdValue);
        return false;
    }
    return true;
}

static inline bool CheckBiasValue(const aclScalar* bias)
{
    // 声明域为任意实数（NaN 不是实数，域外），按 161002 拒绝。
    float biasValue = bias->ToFloat();
    if (std::isnan(biasValue)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias must not be NaN, but found [%f].", biasValue);
        return false;
    }
    return true;
}

inline static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    // 公开契约（DESIGN.md/REQUIREMENTS.md/README/aclnnShrink.md）声明数据格式仅 ND；
    // 私有存储格式（FRACTAL_NZ 等 blocked 布局）的物理元素序与线性序不同，而 Kernel
    // 按线性 ND 寻址（TilingData 仅 totalNum/blockFactor/ubFactor），接受即静默错排。
    // 对齐仓内同类元素级算子（celu_v2/hard_shrink）的 Regbase 架构行为：self/out
    // 存储格式任一私有即按文档返回 161002。
    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format is not supported, self [%s], out [%s].",
                op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    OP_CHECK(self->GetViewFormat() == out->GetViewFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], output [%s].",
                     op::ToString(self->GetViewFormat()).GetString(), op::ToString(out->GetViewFormat()).GetString()),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias,
                               const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, lambd, bias, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, lambd, bias, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckLambdValue(lambd), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckBiasValue(bias), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecShrinkGetWorkspaceSize(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias,
                                              aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, lambd, bias, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto shrinkOutRet = l0op::Shrink(selfContiguous, lambd->ToFloat(), bias->ToFloat(), uniqueExecutor.get());
    CHECK_RET(shrinkOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // CheckDtypeValid 已强制 self/out dtype 一致（161002），l0op::Shrink 按输入 dtype
    // 分配输出，shrinkOutRet 与 out 的 dtype 恒同；此处原有的 l0op::Cast 为恒等转换，
    // 只会额外下发一次无意义 Cast 并误导维护者，已删除。
    auto viewCopyResult = l0op::ViewCopy(shrinkOutRet, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnShrinkGetWorkspaceSize(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias,
                                        aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnShrink, DFX_IN(self, lambd, bias), DFX_OUT(out));
    return ExecShrinkGetWorkspaceSize(self, lambd, bias, out, workspaceSize, executor);
}
aclnnStatus aclnnShrink(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnShrink);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
} // namespace op;
#endif
