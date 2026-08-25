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
 * \file apply_came_part1.cpp
 * \brief
 */

#include "arch35/apply_came_part1_tiling_data.h"
#include "arch35/apply_came_part1_tiling_key.h"
#include "arch35/apply_came_part1_fp32.h"
#include "arch35/apply_came_part1_fp16.h"
#include "arch35/apply_came_part1_post.h"

using namespace ApplyCamePart1;

template <typename D_T>
__global__ __aicore__ void apply_came_part1(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c,
                                            GM_ADDR sum_grad_rc, GM_ADDR workspace, GM_ADDR tiling)
{
    // Keep the batch loop in the entry point so header-only kernel changes are rebuilt with the entry source.
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    ENABLE_DETERMINISTIC();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ApplyCamePart1TilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyCamePart1TilingData, tilingData, tiling);

    for (int64_t batchIdx = 0; batchIdx < tilingData.batchCount; ++batchIdx) {
        GM_ADDR batchGrad = grad;
        GM_ADDR batchSumGradR = sum_grad_r;
        GM_ADDR batchSumGradC = sum_grad_c;
        GM_ADDR batchSumGradRC = sum_grad_rc;
        if constexpr (sizeof(D_T) == sizeof(float)) {
            ApplyCamePart1::ApplyCamePart1FP32<float> op;
            op.Init(batchGrad, eps, batchSumGradR, batchSumGradC, batchSumGradRC, userWS, &tilingData, batchIdx);
            op.Process();
            ApplyCamePart1::ApplyCamePart1Post<float> opPost;
            opPost.Init(batchGrad, eps, batchSumGradR, batchSumGradC, batchSumGradRC, userWS, &tilingData, batchIdx);
            opPost.Process();
        } else {
            ApplyCamePart1::ApplyCamePart1FP16<D_T> op;
            op.Init(batchGrad, eps, batchSumGradR, batchSumGradC, batchSumGradRC, userWS, &tilingData, batchIdx);
            op.Process();
            ApplyCamePart1::ApplyCamePart1Post<float> opPost;
            opPost.Init(batchGrad, eps, batchSumGradR, batchSumGradC, batchSumGradRC, userWS, &tilingData, batchIdx);
            opPost.Process();
        }
    }
}
