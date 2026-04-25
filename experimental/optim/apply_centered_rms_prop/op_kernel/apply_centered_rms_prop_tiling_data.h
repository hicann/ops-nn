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

/**
 * \file apply_centered_rms_prop_tiling_data.h
 * \brief ApplyCenteredRMSProp TilingData structure (arch35).
 *
 * Standard C++ struct form (per DESIGN.md §5.3 "not using BEGIN_TILING_DATA_DEF").
 *
 * Iteration-1 skeleton: fp16 path only. The struct itself is dtype-agnostic;
 * fp32 path will be added in iteration-2 and reuse the same TilingData layout.
 */

#ifndef _APPLY_CENTERED_RMS_PROP_TILING_DATA_H_
#define _APPLY_CENTERED_RMS_PROP_TILING_DATA_H_

#include <cstdint>

struct ApplyCenteredRMSPropTilingData {
    // Total number of elements in var/mg/ms/mom/grad (all share the same shape).
    int64_t totalElements = 0;
    // Number of elements per-core (aligned up to ubBlockSize to keep DMA safe).
    int64_t blockFactor = 0;
    // Number of elements processed per UB iteration.
    int64_t ubFactor = 0;
};

#endif // _APPLY_CENTERED_RMS_PROP_TILING_DATA_H_
