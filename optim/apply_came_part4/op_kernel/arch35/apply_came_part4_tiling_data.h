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
 * \file apply_came_part4_tiling_data.h
 * \brief ApplyCamePart4 TilingData structure (arch35)
 *
 * Aligned to canndev ApplyCamePart4TilingData, with the ConfusionTranspose
 * tiling struct removed (v1 always computes r*c via per-row Muls, see kernel).
 */

#ifndef _APPLY_CAME_PART4_TILING_DATA_H_
#define _APPLY_CAME_PART4_TILING_DATA_H_

#include <cstdint>

struct ApplyCamePart4TilingData {
    int64_t n = 0;            // rows of param (length of r)
    int64_t m = 0;            // columns of param (length of c)
    int64_t totalCoreNum = 0; // total AIV core number
    int64_t handleMax = 0;    // max elements handled per loop in R/C phase

    // R phase split (over n)
    int64_t rNumPerCore = 0; // elements per non-tail core
    int64_t rCoreNumToUse = 0;
    int64_t rNumPerLoop = 0; // elements per UB loop, non-tail core
    int64_t rLoopCount = 0;
    int64_t rNumTailPerLoop = 0; // elements per UB loop, tail core
    int64_t rLoopCountTailCore = 0;
    int64_t rNumTailLoopLast = 0; // remaining unaligned elements after tail-core loops

    // C phase split (over m), same layout as R
    int64_t cNumPerCore = 0;
    int64_t cCoreNumToUse = 0;
    int64_t cNumPerLoop = 0;
    int64_t cLoopCount = 0;
    int64_t cNumTailPerLoop = 0;
    int64_t cLoopCountTailCore = 0;
    int64_t cNumTailLoopLast = 0;

    // Param phase split (cores over n; inner 2D tile loops)
    int64_t rRcNumPerCore = 0;
    int64_t rRcCoreNumToUse = 0;
    int64_t rRcNumOnTailCore = 0;
    int64_t rRcNumPerLoop = 0; // rows per tile
    int64_t rRcLoopCount = 0;
    int64_t rRcNumTailLoop = 0;
    int64_t rRcLoopCountTailCore = 0;
    int64_t rRcNumTailLoopTailCore = 0;
    int64_t cRcNumPerLoop = 0; // columns per tile
    int64_t cRcLoopCount = 0;
    int64_t cRcNumTailLoop = 0;
};

#endif // _APPLY_CAME_PART4_TILING_DATA_H_
