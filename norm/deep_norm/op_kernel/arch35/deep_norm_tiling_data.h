/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file deep_norm_tiling_data.h
 * \brief DeepNorm arch35 (Ascend950) plain tiling-data struct.
 */

#ifndef _DEEP_NORM_TILING_DATA_H_
#define _DEEP_NORM_TILING_DATA_H_

#include <cstdint>

struct DeepNormTilingData {
    uint32_t numCore = 0;      // used core number
    uint32_t numCol = 0;       // last dim length D (reduce axis)
    uint32_t numRow = 0;       // product of leading dims N
    uint32_t rowPerCore = 0;   // rows handled by a non-tail core
    uint32_t numColAlign = 0;  // D aligned up to the fp32 vector length
    uint32_t powerSplit = 0;   // largest power-of-two <= D (>= vl), reduce fold point
    float eps = 0.0f;          // epsilon
    float alpha = 0.0f;        // alpha scale for x
    float avgFactor = 0.0f;    // 1 / D
};

#endif // _DEEP_NORM_TILING_DATA_H_
