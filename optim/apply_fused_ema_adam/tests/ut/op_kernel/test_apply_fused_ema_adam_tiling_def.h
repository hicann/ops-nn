/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _APPLY_FUSED_EMA_ADAM_H_
#define _APPLY_FUSED_EMA_ADAM_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct ApplyFusedEmaAdamUtTilingData {
    float lr = 1e-3;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    float emaDecay = 0.5;
    float weightDecay = 0.5;
    uint64_t mode;
    uint64_t biasCorrection;

    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t coreCalcNum;
    uint64_t loopNum;
    uint64_t coreCalcMax;
    uint64_t frontCalcExtra;
    uint64_t tailCalcExtra;
};

inline void InitApplyFusedEmaAdamTilingData(uint8_t* tiling, ApplyFusedEmaAdamUtTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ApplyFusedEmaAdamUtTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    ApplyFusedEmaAdamUtTilingData tilingData;      \
    InitApplyFusedEmaAdamTilingData(tilingPointer, &tilingData)
#endif
