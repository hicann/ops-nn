/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_fused.cpp
 * \brief
 */

#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "sync_batch_norm_gather_stats_fused.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SyncBatchNormGatherStatsFused);

const std::array<const aclTensor*, 4> SyncBatchNormGatherStatsFused(const aclTensor* mean, const aclTensor* invstd,
                                                                    const aclTensor* counts,
                                                                    const aclTensor* runningMean,
                                                                    const aclTensor* runningVar, float momentum,
                                                                    float eps, aclOpExecutor* executor)
{
    L0_DFX(SyncBatchNormGatherStatsFused, mean, invstd, counts, runningMean, runningVar, momentum, eps);
    auto shape = runningVar->GetViewShape();
    auto dtype = mean->GetDataType();

    auto meanAllOut = executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
    auto invstdAllOut = executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
    auto runningMeanOut = executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
    auto runningVarOut = executor->AllocTensor(shape, dtype, Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SyncBatchNormGatherStatsFused, OP_INPUT(mean, invstd, counts, runningMean, runningVar),
        OP_OUTPUT(meanAllOut, invstdAllOut, runningMeanOut, runningVarOut), OP_ATTR(momentum, eps));

    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SyncBatchNormGatherStatsFused kernel launch failed.");
    }

    return {meanAllOut, invstdAllOut, runningMeanOut, runningVarOut};
}

} // namespace l0op