/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_util.h"

#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const uint64_t kDeqScaleMul = 0xFFFFE000;
static const uint64_t kQuantScale = 0x1ULL << 46;
static const uint64_t kFixpipeQuantMask0 = 0x1FFULL;
static const uint64_t kFixpipeQuantMask1 = 0x4000FFFFFFFFULL;
static const int32_t kK0 = 32;
static const int offsetDeviation = 37;

static int Float32ToInt9(float value)
{
    int intValue = static_cast<int>(round(value));
    int int9Value = std::max(-256, std::min(255, intValue));
    return int9Value;
}
}; // namespace

aclnnStatus TransQuantScaleToM1(const float* scaleArray, uint64_t scaleSize, uint64_t** quantParam)
{
    union {
        float scaleFloat;
        uint32_t scaleUint32;
    } quantScale;
    for (int64_t idx = 0; idx < static_cast<int64_t>(scaleSize); idx++) {
        quantScale.scaleFloat = scaleArray[idx];
        (*quantParam)[idx] = static_cast<uint64_t>(quantScale.scaleUint32) | kQuantScale;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus TransQuantOffsetToToM1(const float* offsetArray, uint64_t offsetSize, uint64_t** quantParam)
{
    int int9Value;
    uint64_t int9bits;
    for (uint64_t idx = 0; idx < offsetSize; idx++) {
        int9Value = Float32ToInt9(offsetArray[idx]);
        int9bits = (static_cast<uint64_t>(int9Value) & kFixpipeQuantMask0) << offsetDeviation;
        (*quantParam)[idx] &= kFixpipeQuantMask1;
        (*quantParam)[idx] |= int9bits;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif