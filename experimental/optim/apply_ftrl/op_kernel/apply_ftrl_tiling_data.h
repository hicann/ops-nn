/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_ftrl_tiling_data.h
 * \brief ApplyFtrl TilingData structure (ascend910b / DAV_2201).
 *
 * Standard C++ struct form (per DESIGN s3.2: "禁用 BEGIN_TILING_DATA_DEF").
 * Shared by host tiling and device kernel; contains no ASC keywords.
 */

#ifndef APPLY_FTRL_TILING_DATA_H
#define APPLY_FTRL_TILING_DATA_H

#include <cstdint>

struct ApplyFtrlTilingData {
    // Total number of elements in var/accum/linear/grad (all share the same shape).
    int64_t totalElements = 0;
    // Number of elements per core (aligned up to ubBlockSize to keep DMA safe and
    // neighbouring cores' GM regions non-overlapping).
    int64_t blockFactor = 0;
    // Number of elements processed per UB iteration (256B / 64-element aligned, so the
    // CompareScalar / Select count satisfies the 256B alignment rule).
    int64_t ubFactor = 0;
};

#endif  // APPLY_FTRL_TILING_DATA_H
