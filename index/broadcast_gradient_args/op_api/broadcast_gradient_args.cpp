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
 * \file broadcast_gradient_args.cpp
 * \brief
 */
#include "broadcast_gradient_args.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(BroadcastGradientArgs);

// 每个动态输出占用9个int64槽位，broadcast_gradient_args有2个动态输出(y1/y2)，共需18
constexpr int64_t OUT_SHAPE_SIZE = 18;

aclnnStatus BroadcastGradientArgs(const aclTensor* x1, const aclTensor* x2, aclTensor* y1, aclTensor* y2,
                                  aclOpExecutor* executor)
{
    L0_DFX(BroadcastGradientArgs, x1, x2, y1, y2);

    // 分配outShapeTensor，用于运行时存储y1/y2的实际shape
    // 框架在kernel执行后读取outShapeTensor刷新y1/y2的StorageShape
    Shape outShapeShape{OUT_SHAPE_SIZE};
    auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);
    CHECK_RET(outShapeTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将BroadcastGradientArgs算子加入AICORE任务队列
    // OP_OUTSHAPE({outShapeTensor, 0}) -> y1的shape从outShapeTensor[0]开始读
    // OP_OUTSHAPE({outShapeTensor, 1}) -> y2的shape从outShapeTensor[9]开始读
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BroadcastGradientArgs, OP_INPUT(x1, x2), OP_OUTPUT(y1, y2),
                                           OP_OUTSHAPE({outShapeTensor, 0}), OP_OUTSHAPE({outShapeTensor, 1}));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BroadcastGradientArgs ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace l0op
