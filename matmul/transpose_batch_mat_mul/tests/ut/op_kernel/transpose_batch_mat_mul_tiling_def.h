/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TRANSPOSE_BATCH_MAT_MUL_TILING_DEF_H_
#define _TRANSPOSE_BATCH_MAT_MUL_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#include "transpose_batch_mat_mul_tiling_data.h"

inline void InitTbmmTilingData(void* tiling, void* const_data, size_t size)
{
    memcpy(const_data, tiling, size);
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct_type, tiling_var_name, tiling_arg) \
    tiling_struct_type tiling_var_name;                                           \
    InitTbmmTilingData((tiling_arg), &tiling_var_name, sizeof(tiling_struct_type))
#endif