/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _MSE_LOSS_GRAD_TILING_H
#define _MSE_LOSS_GRAD_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct MseLossGradV2TilingData {
    float cof = 1.0;
    uint64_t totalLength = 1;
    uint64_t tileNum = 1;
    uint64_t padLength = 1;
    uint64_t blockLength = 1;
    uint32_t usedDb = 1;
};

inline void InitMseLossGradV2TilingData(uint8_t* tiling, MseLossGradV2TilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MseLossGradV2TilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    MseLossGradV2TilingData tilingData;            \
    InitMseLossGradV2TilingData(tilingPointer, &tilingData)
#endif
