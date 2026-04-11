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
 * \file softplus_v2_tiling_data.h
 * \brief SoftplusV2 TilingData structure definition
 */

#ifndef _SOFTPLUS_V2_TILING_DATA_H_
#define _SOFTPLUS_V2_TILING_DATA_H_

struct SoftplusV2TilingData {
    int64_t totalLength;      // total number of elements
    int64_t blockFactor;      // elements per core
    int64_t tileLength;       // aligned tile length per iteration
    float beta;               // scaling factor
    float threshold;          // linearization threshold
    float invBeta;            // 1.0 / beta, precomputed on Host
};

#endif // _SOFTPLUS_V2_TILING_DATA_H_
