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

#include "kernel_operator.h"
#include "sync_batch_norm_gather_stats_fused_common.h"
#include "sync_batch_norm_gather_stats_fused_workspace.h"
#include "sync_batch_norm_gather_stats_fused_first_axis_common.h"
#include "sync_batch_norm_gather_stats_fused_first_axis_workspace.h"

using namespace SyncBatchNormGatherStatsFused;

#define KEY_COMMON_FP32 101
#define KEY_COMMON_FP16 102
#define KEY_COMMON_BF16 103
#define KEY_WORKSPACE_FP32 201
#define KEY_WORKSPACE_FP16 202
#define KEY_WORKSPACE_BF16 203
#define KEY_FIRST_AXIS_COMMON_FP32 301
#define KEY_FIRST_AXIS_COMMON_FP16 302
#define KEY_FIRST_AXIS_COMMON_BF16 303
#define KEY_FIRST_AXIS_WORKSPACE_FP32 401
#define KEY_FIRST_AXIS_WORKSPACE_FP16 402
#define KEY_FIRST_AXIS_WORKSPACE_BF16 403

#define INVOKE_COMMON_IMPL(T)                                                                            \
    do {                                                                                                 \
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsFusedTilingDataCommon, tilingData, tiling);  \
        SyncBatchNormGatherStatsFusedCommon<T> op;                                                       \
        op.Init(mean, invstd, counts, runningMean, runningVar, meanAllOut, invstdAllOut, runningMeanOut, \
                runningVarOut, usrWorkspace, &tilingData, pipe);                                         \
        op.Process(&tilingData);                                                                         \
    } while (0)

#define INVOKE_WORKSPACE_IMPL(T)                                                                           \
    do {                                                                                                   \
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsFusedTilingDataWorkspace, tilingData, tiling); \
        SyncBatchNormGatherStatsFusedWorkspace<T> op;                                                      \
        op.Init(mean, invstd, counts, runningMean, runningVar, meanAllOut, invstdAllOut, runningMeanOut,   \
                runningVarOut, usrWorkspace, &tilingData, pipe);                                           \
        op.Process();                                                                                      \
    } while (0)

#define INVOKE_FIRST_AXIS_COMMON_IMPL(T)                                                                         \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon, tilingData, tiling); \
        SyncBatchNormGatherStatsFusedFirstAxisCommon<T> op;                                                      \
        op.Init(mean, invstd, counts, runningMean, runningVar, meanAllOut, invstdAllOut, runningMeanOut,         \
                runningVarOut, usrWorkspace, &tilingData, pipe);                                                 \
        op.Process(&tilingData);                                                                                 \
    } while (0)

#define INVOKE_FIRST_AXIS_WORKSPACE_IMPL(T)                                                                         \
    do {                                                                                                            \
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace, tilingData, tiling); \
        SyncBatchNormGatherStatsFusedFirstAxisWorkspace<T> op;                                                      \
        op.Init(mean, invstd, counts, runningMean, runningVar, meanAllOut, invstdAllOut, runningMeanOut,            \
                runningVarOut, usrWorkspace, &tilingData, pipe);                                                    \
        op.Process();                                                                                               \
    } while (0)

extern "C" __global__ __aicore__ void sync_batch_norm_gather_stats_fused(GM_ADDR mean, GM_ADDR invstd, GM_ADDR counts,
                                                                         GM_ADDR runningMean, GM_ADDR runningVar,
                                                                         GM_ADDR meanAllOut, GM_ADDR invstdAllOut,
                                                                         GM_ADDR runningMeanOut, GM_ADDR runningVarOut,
                                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
    if (g_coreType == AIC) {
        return;
    }
    if (TILING_KEY_IS(KEY_COMMON_FP32)) {
        INVOKE_COMMON_IMPL(float);
    } else if (TILING_KEY_IS(KEY_COMMON_FP16)) {
        INVOKE_COMMON_IMPL(half);
    } else if (TILING_KEY_IS(KEY_COMMON_BF16)) {
        INVOKE_COMMON_IMPL(bfloat16_t);
    } else if (TILING_KEY_IS(KEY_WORKSPACE_FP32)) {
        INVOKE_WORKSPACE_IMPL(float);
    } else if (TILING_KEY_IS(KEY_WORKSPACE_FP16)) {
        INVOKE_WORKSPACE_IMPL(half);
    } else if (TILING_KEY_IS(KEY_WORKSPACE_BF16)) {
        INVOKE_WORKSPACE_IMPL(bfloat16_t);
    } else if (TILING_KEY_IS(KEY_FIRST_AXIS_COMMON_FP32)) {
        INVOKE_FIRST_AXIS_COMMON_IMPL(float);
    } else if (TILING_KEY_IS(KEY_FIRST_AXIS_COMMON_FP16)) {
        INVOKE_FIRST_AXIS_COMMON_IMPL(half);
    } else if (TILING_KEY_IS(KEY_FIRST_AXIS_COMMON_BF16)) {
        INVOKE_FIRST_AXIS_COMMON_IMPL(bfloat16_t);
    } else if (TILING_KEY_IS(KEY_FIRST_AXIS_WORKSPACE_FP32)) {
        INVOKE_FIRST_AXIS_WORKSPACE_IMPL(float);
    } else if (TILING_KEY_IS(KEY_FIRST_AXIS_WORKSPACE_FP16)) {
        INVOKE_FIRST_AXIS_WORKSPACE_IMPL(half);
    } else if (TILING_KEY_IS(KEY_FIRST_AXIS_WORKSPACE_BF16)) {
        INVOKE_FIRST_AXIS_WORKSPACE_IMPL(bfloat16_t);
    }
}