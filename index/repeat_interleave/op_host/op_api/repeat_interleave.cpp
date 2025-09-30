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
 * \file repeat_interleave.cpp
 * \brief
 */
#include "repeat_interleave.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

static const int64_t AXIS_LIMIT = 8;  // L0算子最大支持8维
static const int64_t INT32_MAX_NUM = 2147483647;
static const int64_t MTE_NUM = 256;

namespace l0op {
OP_TYPE_REGISTER(RepeatInterleave);
OP_TYPE_REGISTER(RepeatInterleaveV2);
static const std::initializer_list<op::DataType> AICORE910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool IsRepeatInterleaveV2Support(const aclTensor* x, const aclTensor* repeats, int64_t axis, int64_t outputSize) {
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
        return false;
    }

    if(outputSize >= INT32_MAX_NUM) {
        return false;
    }

    // 这种情况是V1特殊优化场景，走V1
    int64_t xDimNum = x->GetViewShape().GetDimNum();
    if(xDimNum == 1 && repeats->GetViewShape().GetDimNum() == 1 &&
       repeats->GetViewShape().GetDim(0) == x->GetViewShape().GetDim(0)) {
        // repeats and z one-to-one，and rowLength = 1 and not is int64
        if(x->GetDataType() != op::DataType::DT_INT64) {
            return false;
        }
    }

    if (xDimNum == 0) {
        return false; //V2 不支持标量tensor
    }

    auto xDimSize = x->GetViewShape().GetDimNum();
    axis = axis >= 0 ? axis : axis + xDimSize; // axis保证为非负数
    int64_t rows = 1;
    for (int64_t i = 0; i < axis; i++) {
        rows *= (x->GetViewShape())[i];
    }

    if (rows * outputSize < MTE_NUM) {
        return false; // V2始终是满核运行，搬运次数小于256时走V1，防止小shape因为V2启动的核多导致劣化
    }

    return CheckType(x->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
}

static op::Shape RepeatInterleaveInferShape(const aclTensor* x, int64_t axis, int64_t outputSize)
{
    // 运行repeatInterleave的轴一定是dim。对应的dim维度大小必须为outputSize
    op::Shape initialShape = x->GetViewShape();
    initialShape.SetDim(axis, outputSize);
    return initialShape;
}

// 只支持AICORE
const aclTensor *RepeatInterleave(const aclTensor* x, const aclTensor* repeats, int64_t axis, int64_t outputSize,
    aclOpExecutor *executor)
{
    L0_DFX(RepeatInterleave, x, repeats, axis, outputSize);
    op::Shape outShape = RepeatInterleaveInferShape(x, axis, outputSize);
    auto out = executor->AllocTensor(outShape, x->GetDataType(), x->GetViewFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RepeatInterleave, OP_INPUT(x, repeats), OP_OUTPUT(out), OP_ATTR(axis));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RepeatInterleaveAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// 只支持AICORE
const aclTensor *RepeatInterleaveV2(const aclTensor* x, const aclTensor* repeats, int64_t axis, int64_t outputSize,
    aclOpExecutor *executor)
{
    L0_DFX(RepeatInterleaveV2, x, repeats, axis, outputSize);
    op::Shape outShape = RepeatInterleaveInferShape(x, axis, outputSize);
    auto out = executor->AllocTensor(outShape, x->GetDataType(), x->GetViewFormat());
    if(IsRepeatInterleaveV2Support(x, repeats, axis, outputSize)) {
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RepeatInterleaveV2, OP_INPUT(x, repeats), OP_OUTPUT(out), OP_ATTR(axis));
        OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RepeatInterleaveV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
    } else {
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RepeatInterleave, OP_INPUT(x, repeats), OP_OUTPUT(out), OP_ATTR(axis));
        OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RepeatInterleaveAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
    }
    return out;
}
}