/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_FUSED_MAT_MUL_TILING_H_
#define _TEST_FUSED_MAT_MUL_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
#include <string>

inline void InitFusedMatMulTilingData(void* tiling, void* const_data, size_t size)
{
    memcpy(const_data, tiling, size);
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitFusedMatMulTilingData(tiling_arg, &tiling_data, sizeof(tiling_struct));

#define X_LOG(format, ...)

#define KERNEL_LOG_KERNEL_EORROR(format, ...)

#endif  // _TEST_FUSED_MAT_MUL_TILING_H_

