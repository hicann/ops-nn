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
 * \file nonzero_tiling_data.h
 * \brief Tiling data struct for NonZero operator
 */

#ifndef __NONZERO_TILING_DATA_H__
#define __NONZERO_TILING_DATA_H__

#include <cstdint>

struct NonzeroTilingData {
    int64_t totalRows;   // number of rows in 2D-flattened input
    int64_t cols;        // number of columns (product of dims beyond first)
    int64_t rowsPerCore; // rows assigned per core
    int64_t rowStride;   // logical stride of input tensor (== cols for contiguous)
    int64_t wsStride;    // workspace slots per core (count header at blockIdx*wsStride)
};

#endif // __NONZERO_TILING_DATA_H__
