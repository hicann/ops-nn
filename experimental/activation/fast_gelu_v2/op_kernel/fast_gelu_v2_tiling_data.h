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
 * \file fast_gelu_v2_tiling_data.h
 * \brief FastGeluV2 TilingData structure definition
 *
 * This structure is filled by the host-side Tiling function and read by the
 * device-side Kernel via GET_TILING_DATA_WITH_STRUCT. All fields must be
 * int64_t for cross-platform compatibility.
 */

#ifndef _FAST_GELU_V2_TILING_DATA_H_
#define _FAST_GELU_V2_TILING_DATA_H_

struct FastGeluV2TilingData {
    int64_t totalNum = 0;     // Total number of input elements across all dimensions
    int64_t blockFactor = 0;  // Number of elements assigned to each AI Core (aligned to ubBlockSize)
    int64_t ubFactor = 0;     // Number of elements processed per UB iteration (aligned to ubBlockSize)
};
#endif
