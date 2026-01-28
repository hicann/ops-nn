/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nonzero.cpp
 * \brief
 */
#include "nonzero.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;

namespace l0op {
constexpr uint32_t NONZERO_TENSOR_DIMS_MAX = 8;
OP_TYPE_REGISTER(NonZero);

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL,  op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_FLOAT, op::DataType::DT_INT8,    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> REGBASE_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL,  op::DataType::DT_FLOAT16, op::DataType::DT_BF16,  op::DataType::DT_FLOAT,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_INT32, op::DataType::DT_UINT32, op::DataType::DT_INT64};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        if (self->GetViewShape().GetDimNum() == 0) {
            return false;
        }
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    } else if (Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return CheckType(self->GetDataType(), REGBASE_AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

// AICORE算子kernel
static aclTensor* NonzeroAiCore(const aclTensor* self, aclTensor* out, bool transpose, aclOpExecutor* executor)
{
    L0_DFX(NonzeroAiCore, self, out, transpose);
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        Shape outShapeShape{NONZERO_TENSOR_DIMS_MAX + 1};
        auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
            NonZero, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(transpose, out->GetDataType()),
            OP_OUTSHAPE({outShapeTensor, 0}));
        CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
        return out;
    }
    // outShapeTensor用于执行期放置实际shape，并用于刷新out的大小。
    Shape outShapeShape{static_cast<int64_t>(out->GetViewShape().GetDimNum()) + 1};
    auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT32, Format::FORMAT_ND);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将NonZero算子加入任务队列
    // NonZero是算子的OpType，self是算子的输入，out是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        NonZero, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(transpose, out->GetDataType()),
        OP_OUTSHAPE({outShapeTensor, 0}));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

// AICPU算子kernel
static aclTensor* NonzeroAiCpu(const aclTensor* self, aclTensor* out, bool transpose, aclOpExecutor* executor)
{
    L0_DFX(NonzeroAiCpu, self, out, transpose);
    static internal::AicpuTaskSpace space("NonZero", ge::DEPEND_SHAPE_RANGE);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        NonZero, OP_ATTR_NAMES({"transpose", "dtype"}), OP_INPUT(self), OP_OUTPUT(out),
        OP_ATTR(transpose, out->GetDataType()));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

aclTensor* Nonzero(const aclTensor* self, aclTensor* out, aclOpExecutor* executor, bool transpose)
{
    if (IsAiCoreSupport(self)) {
        return NonzeroAiCore(self, out, transpose, executor);
    } else {
        return NonzeroAiCpu(self, out, transpose, executor);
    }
}
} // namespace l0op
