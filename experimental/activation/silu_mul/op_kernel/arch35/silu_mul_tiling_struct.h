/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file silu_mul_tiling_struct.h
 * \brief TilingData struct for SiluMul Arch35 (Ascend 950)
 *
 * Plain POD struct used with REGISTER_TILING_DEFAULT / GET_TILING_DATA_WITH_STRUCT,
 * decoupled from the 910b BEGIN_TILING_DATA_DEF-based SiluMulTilingData so both
 * platforms can coexist without duplicate registration.
 */

#ifndef SILU_MUL_TILING_STRUCT_H
#define SILU_MUL_TILING_STRUCT_H

#include "kernel_tiling/kernel_tiling.h"

struct SiluMulArch35TilingData {
    int32_t lastDimSize;  // last dim (tail axis) element count d
    int32_t batchSize;    // total elements / d
    int32_t PPMaxCalNum;  // max elements processable per UB iteration (platform-derived)
    uint32_t needCoreNum; // number of cores actually launched
    uint32_t maxUbSize;   // platform UB size in bytes, used for kernel UB layout
};

#endif // SILU_MUL_TILING_STRUCT_H
