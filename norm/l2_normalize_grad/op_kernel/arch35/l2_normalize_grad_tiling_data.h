/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_tiling_data.h
 * \brief L2NormalizeGrad arch35 (Ascend950) plain tiling-data struct.
 *
 * One definition shared by host tiling (context->GetTilingData<L2NormalizeGradTilingData>())
 * and kernel (GET_TILING_DATA_WITH_STRUCT). [outer, D, inner] reduction model; cores split by outer.
 */
#ifndef _L2_NORMALIZE_GRAD_TILING_DATA_H_
#define _L2_NORMALIZE_GRAD_TILING_DATA_H_

#include <cstdint>

struct L2NormalizeGradTilingData {
    int64_t outer = 0;       // product of dims before the reduce axis
    int64_t dimLen = 0;      // D = length of the reduce axis
    int64_t inner = 0;       // product of dims after the reduce axis
    int64_t blockFactor = 0; // outer groups per (non-tail) core
    int64_t usedCoreNum = 0; // cores actually used
    int64_t colFactor = 0;   // inner columns per tile (strided/7020 only)
    float eps = 0.0f;        // denominator floor
};

#endif // _L2_NORMALIZE_GRAD_TILING_DATA_H_
