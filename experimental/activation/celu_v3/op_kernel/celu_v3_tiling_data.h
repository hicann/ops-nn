/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file celu_v3_tiling_data.h
 * \brief CeluV3 TilingData structure definition
 *
 * Standard C++ struct (no BEGIN_TILING_DATA_DEF macro).
 */

#ifndef _CELU_V3_TILING_DATA_H_
#define _CELU_V3_TILING_DATA_H_

struct CeluV3TilingData {
    int64_t totalElements = 0;   // Total number of elements
    int64_t blockFactor = 0;     // Number of elements per AI Core
    int64_t ubFactor = 0;        // Number of elements per UB iteration
    float alphaVal = 1.0f;       // alpha value
    float invAlpha = 1.0f;       // 1.0 / alpha
};

#endif // _CELU_V3_TILING_DATA_H_
