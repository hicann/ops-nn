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
 * \file quantize_tiling_def.h
 * \brief Kernel-UT tiling struct + GET_TILING_DATA (mirrors op_host/quantize_tiling.h QuantizeTilingData layout).
 */
#ifndef QUANTIZE_TILING_DEF_H
#define QUANTIZE_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__

struct QuantizeTilingData {
    uint32_t numCore = 1;
    uint32_t hasZeroPoint = 0;
    int64_t channelNum = 1;
    int64_t rowLen = 1;
    int64_t totalRows = 1;
    int64_t totalElems = 0;
    int64_t blockFactor = 0;
    int64_t blockTailFactor = 0;
    int64_t baseLen = 256;
    uint32_t zpDtype = 0;
};

inline void IQuantizeTilingData(uint8_t* tiling, QuantizeTilingData* constData)
{
    memcpy(constData, tiling, sizeof(QuantizeTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    QuantizeTilingData tilingData;                 \
    IQuantizeTilingData(tilingPointer, &tilingData)

#endif // QUANTIZE_TILING_DEF_H
