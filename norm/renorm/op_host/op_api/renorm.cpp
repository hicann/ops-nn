/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file renorm.cpp
 * \brief
 */
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_dfx.h"
#include "renorm.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(Renorm);

// AiCore支持的Renorm类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 对Renorm的输入输出进行infershape
static bool RenormInferShape(const op::Shape& selfShape, op::Shape& outShape, const int64_t dim)
{
    size_t real_dim_num = selfShape.GetDimNum();
    if (dim < 0 || static_cast<size_t>(dim) >= real_dim_num ) {
        return false;
    }
    outShape.SetDimNum(real_dim_num);
    for (size_t i = 0; i < real_dim_num; ++i) {
        outShape.SetDim(i, 1);
    }
    // 仅dim指定的维度大小与self的保持一致，其余维度的大小为1。
    outShape.SetDim(dim, selfShape.GetDim(dim));
    return true;
}

// 判断走AiCore还是AiCPU
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (!CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    return true;
}

// AiCore的执行逻辑
static aclTensor* RenormAiCore(
    const aclTensor* self, const float normType, const int64_t dim, const float maxNorm, aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(RenormAiCore, self, normType, dim, maxNorm, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Renorm, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(normType, dim, maxNorm));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RenormAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}

// Renorm的实现
const aclTensor* Renorm(
    const aclTensor* self, const float normType, const int64_t dim, const float maxNorm, aclOpExecutor* executor)
{
    // 目前Renorm无AiCPU,仅支持AiCore
    if (!IsAiCoreSupport(self)) {
        return nullptr;
    }
    op::Shape outShape;
    if (!RenormInferShape(self->GetViewShape(), outShape, dim)) {
        OP_LOGE(
            ACL_ERROR_INVALID_PARAM, "infer shape failed. input shorage shape: [%s], view shape: [%s]",
            op::ToString(self->GetStorageShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return nullptr;
    }
    auto out = executor->AllocTensor(outShape, self->GetDataType(), op::Format::FORMAT_ND);
    if (!RenormAiCore(self, normType, dim, maxNorm, out, executor)) {
        return nullptr;
    }
    return out;
}
} // namespace l0op
