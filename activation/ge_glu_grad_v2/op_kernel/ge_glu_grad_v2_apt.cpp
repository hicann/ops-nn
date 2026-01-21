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
 * \file ge_glu_grad_v2_apt.cpp
 * \brief
 */

#include "arch35/ge_glu_grad_v2_tanh_bfp16.h"
#include "arch35/ge_glu_grad_v2_tanh_fp16.h"
#include "arch35/ge_glu_grad_v2_tanh_fp32.h"
#include "arch35/ge_glu_grad_v2_erf_bfp16.h"
#include "arch35/ge_glu_grad_v2_erf_fp16.h"
#include "arch35/ge_glu_grad_v2_erf_fp32.h"
#include "../inc/platform.h"
#include "kernel_operator.h"

using namespace AscendC;

#define TILING_KEY_201 201
#define TILING_KEY_202 202
#define TILING_KEY_203 203
#define TILING_KEY_301 301
#define TILING_KEY_302 302
#define TILING_KEY_303 303
#define TILING_KEY_101 101
#define TILING_KEY_102 102
#define TILING_KEY_103 103
#define TILING_KEY_701 701
#define TILING_KEY_702 702
#define TILING_KEY_703 703
#define TILING_KEY_801 801
#define TILING_KEY_802 802
#define TILING_KEY_803 803
#define TILING_KEY_901 901
#define TILING_KEY_902 902
#define TILING_KEY_903 903

KERNEL_API void ge_glu_grad_v2(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    TPipe tPipe;
    /* Tanh */
    if (TILING_KEY_IS(TILING_KEY_101)) {
        GeGluGradV2Tanh::GeGluGradV2TanhBFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_102)) {
        GeGluGradV2Tanh::GeGluGradV2TanhBFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_201)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_202)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_301)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP32 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_302)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP32 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_103)) {
        GeGluGradV2Tanh::GeGluGradV2TanhBFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process(true);
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_203)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process(true);
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_303)) {
        GeGluGradV2Tanh::GeGluGradV2TanhFP32 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process(true);
        return;
    }

    /* Erf */
    if (TILING_KEY_IS(TILING_KEY_701)) {
        GeGluGradV2Erf::GeGluGradV2ErfBFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_702)) {
        GeGluGradV2Erf::GeGluGradV2ErfBFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_801)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_802)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_901)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP32 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_902)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP32 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process();
    }
    if (TILING_KEY_IS(TILING_KEY_703)) {
        GeGluGradV2Erf::GeGluGradV2ErfBFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process(true);
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_803)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP16 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process(true);
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_903)) {
        GeGluGradV2Erf::GeGluGradV2ErfFP32 op(dy, x, gelu, dx, &tilingData, &tPipe);
        op.Init();
        op.Process(true);
        return;
    }
}
