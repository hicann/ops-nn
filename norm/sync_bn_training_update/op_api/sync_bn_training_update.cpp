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
 * \file sync_bn_training_update.cpp
 * \brief
 */
#include "sync_bn_training_update.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SyncBNTrainingUpdate);
const aclTensor* SyncBNTrainingUpdate(
    const aclTensor* meanAll, const aclTensor* runningMean, const float momentum, aclOpExecutor* executor)
{
    L0_DFX(SyncBNTrainingUpdate, meanAll, runningMean, momentum);
    auto out = executor->AllocTensor(meanAll->GetViewShape(), meanAll->GetDataType(), meanAll->GetViewFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SyncBNTrainingUpdate, OP_INPUT(meanAll, runningMean), OP_OUTPUT(out), OP_ATTR(momentum));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SyncBNTrainingUpdateAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}
} // namespace l0op