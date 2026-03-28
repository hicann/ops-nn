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
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/**
 * \file foreach_div_scalar_tiling_data.h
 * \brief ForeachDivScalar TilingData structure definition
 */

#ifndef _FOREACH_DIV_SCALAR_TILING_DATA_H_
#define _FOREACH_DIV_SCALAR_TILING_DATA_H_

// Maximum supported TensorList length
constexpr uint32_t MAX_TENSOR_NUM = 128;

struct ForeachDivScalarTilingData {
    // ---- Global info ----
    uint32_t tensorNum = 0;           // Number of tensors in TensorList

    // ---- Scalar info ----
    uint32_t scalarDtype = 0;         // 0=float, 1=float16, 2=double (kernel reads scalar from GM directly)

    // ---- Multi-core split info ----
    int64_t totalElements = 0;        // Total elements across all tensors
    int64_t blockFactor = 0;          // Elements per core (element-granularity multi-core splitting)

    // ---- UB split info ----
    int64_t ubFactor = 0;             // Elements per UB loop iteration

    // ---- Per-tensor descriptors ----
    // Use int32_t to keep struct size compact (each tensor up to 2B elements is sufficient)
    int32_t tensorLengths[MAX_TENSOR_NUM] = {0};  // Element count of each tensor
};

#endif
