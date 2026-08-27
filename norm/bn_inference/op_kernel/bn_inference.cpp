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
 * \file bn_inference.cpp
 * \brief BNInference kernel entry for Ascend 950.
 */

#include "kernel_operator.h"
#include "arch35/bn_inference_generic.h"
#include "arch35/bn_inference_packed.h"
#include "arch35/bn_inference_tiling_data.h"
#include "arch35/bn_inference_tiling_key.h"

using namespace AscendC;
using namespace BNInferenceOps;

// Optional inputs do not have DTYPE_* macros when they are absent in GE/TTK compilation.
// The fallback types are only used to instantiate branches guarded by the HAS_* template flags.
#ifndef DTYPE_SCALE
#define DTYPE_SCALE DTYPE_X
#endif

#ifndef DTYPE_OFFSET
#define DTYPE_OFFSET DTYPE_X
#endif

#define BN_INFERENCE_RUN_GENERIC(HAS_SCALE, HAS_OFFSET, CHANNEL_LAST, PRE_FOLDED)                                     \
    do {                                                                                                              \
        GET_TILING_DATA_WITH_STRUCT(BNInferenceTilingData, bnInferenceTilingData, tiling);                            \
        TPipe bnInferencePipe;                                                                                        \
        BNInferenceGeneric<DTYPE_X, DTYPE_MEAN, DTYPE_VARIANCE, DTYPE_MOMENTUM, DTYPE_SCALE, DTYPE_OFFSET, HAS_SCALE, \
                           HAS_OFFSET, CHANNEL_LAST, PRE_FOLDED>                                                      \
            bnInferenceOp;                                                                                            \
        bnInferenceOp.Init(x, mean, variance, momentum, scale, offset, y, &bnInferenceTilingData, &bnInferencePipe);  \
        bnInferenceOp.Process();                                                                                      \
    } while (0)

#define BN_INFERENCE_RUN_PACKED(HAS_SCALE, HAS_OFFSET, CHANNEL_LAST, PRE_FOLDED)                                     \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(BNInferenceTilingData, bnInferenceTilingData, tiling);                           \
        TPipe bnInferencePipe;                                                                                       \
        BNInferencePacked<DTYPE_X, DTYPE_MEAN, DTYPE_VARIANCE, DTYPE_MOMENTUM, DTYPE_SCALE, DTYPE_OFFSET, HAS_SCALE, \
                          HAS_OFFSET, CHANNEL_LAST, PRE_FOLDED>                                                      \
            bnInferenceOp;                                                                                           \
        bnInferenceOp.Init(x, mean, variance, momentum, scale, offset, y, &bnInferenceTilingData, &bnInferencePipe); \
        bnInferenceOp.Process();                                                                                     \
    } while (0)

extern "C" __global__ __aicore__ void bn_inference(GM_ADDR x, GM_ADDR mean, GM_ADDR variance, GM_ADDR momentum,
                                                   GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    if (TILING_KEY_IS(BN_INFERENCE_KEY_EMPTY)) {
        return;
    }

    REGISTER_TILING_DEFAULT(BNInferenceTilingData);
    if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_GENERIC_0)) {
        BN_INFERENCE_RUN_GENERIC(false, false, false, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_GENERIC_1)) {
        BN_INFERENCE_RUN_GENERIC(true, false, false, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_GENERIC_2)) {
        BN_INFERENCE_RUN_GENERIC(false, true, false, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_GENERIC_3)) {
        BN_INFERENCE_RUN_GENERIC(true, true, false, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_PACKED_0)) {
        BN_INFERENCE_RUN_PACKED(false, false, false, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_PACKED_1)) {
        BN_INFERENCE_RUN_PACKED(true, false, false, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_PACKED_2)) {
        BN_INFERENCE_RUN_PACKED(false, true, false, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_PACKED_3)) {
        BN_INFERENCE_RUN_PACKED(true, true, false, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_GENERIC_0)) {
        BN_INFERENCE_RUN_GENERIC(false, false, true, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_GENERIC_1)) {
        BN_INFERENCE_RUN_GENERIC(true, false, true, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_GENERIC_2)) {
        BN_INFERENCE_RUN_GENERIC(false, true, true, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_GENERIC_3)) {
        BN_INFERENCE_RUN_GENERIC(true, true, true, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_PACKED_0)) {
        BN_INFERENCE_RUN_PACKED(false, false, true, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_PACKED_1)) {
        BN_INFERENCE_RUN_PACKED(true, false, true, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_PACKED_2)) {
        BN_INFERENCE_RUN_PACKED(false, true, true, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_PACKED_3)) {
        BN_INFERENCE_RUN_PACKED(true, true, true, false);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_GENERIC_PRE_FOLDED_0)) {
        BN_INFERENCE_RUN_GENERIC(false, false, false, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_GENERIC_PRE_FOLDED_1)) {
        BN_INFERENCE_RUN_GENERIC(true, false, false, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_GENERIC_PRE_FOLDED_3)) {
        BN_INFERENCE_RUN_GENERIC(true, true, false, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_PACKED_PRE_FOLDED_0)) {
        BN_INFERENCE_RUN_PACKED(false, false, false, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_PACKED_PRE_FOLDED_1)) {
        BN_INFERENCE_RUN_PACKED(true, false, false, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CF_PACKED_PRE_FOLDED_3)) {
        BN_INFERENCE_RUN_PACKED(true, true, false, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_GENERIC_PRE_FOLDED_0)) {
        BN_INFERENCE_RUN_GENERIC(false, false, true, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_GENERIC_PRE_FOLDED_1)) {
        BN_INFERENCE_RUN_GENERIC(true, false, true, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_GENERIC_PRE_FOLDED_3)) {
        BN_INFERENCE_RUN_GENERIC(true, true, true, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_PACKED_PRE_FOLDED_0)) {
        BN_INFERENCE_RUN_PACKED(false, false, true, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_PACKED_PRE_FOLDED_1)) {
        BN_INFERENCE_RUN_PACKED(true, false, true, true);
    } else if (TILING_KEY_IS(BN_INFERENCE_KEY_CL_PACKED_PRE_FOLDED_3)) {
        BN_INFERENCE_RUN_PACKED(true, true, true, true);
    }
}

#undef BN_INFERENCE_RUN_GENERIC
#undef BN_INFERENCE_RUN_PACKED
