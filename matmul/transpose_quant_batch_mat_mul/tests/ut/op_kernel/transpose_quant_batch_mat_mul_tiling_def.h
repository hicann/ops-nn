/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TRANSPOSE_QUANT_BATCH_MAT_MUL_TILING_DEF_H_
#define _TRANSPOSE_QUANT_BATCH_MAT_MUL_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../mat_mul_v3/arch35/mat_mul_tiling_data.h"

inline void InitTqbmmTilingData(void* tiling, void* const_data)
{
    memcpy(const_data, tiling, sizeof(BatchMatMulV3TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    BatchMatMulV3TilingData tiling_data;         \
    InitTqbmmTilingData(tiling_arg, &tiling_data);
#endif

#define TQBMM_IMPL_CLASS_COMMON_TRNAS(transposeX1, transposeX2, templateClass, ...)                         \
    do {                                                                                                    \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_BIAS, DTYPE_X1_SCALE, DTYPE_Y, transposeX1, \
                      transposeX2, DTYPE_LOC_LOCAL, __VA_ARGS__>                                            \
            op;                                                                                             \
        op.Init(aGM, bGM, x2_scaleGM, x1_scaleGM, cGM, user, &tilingData, &pipe);                           \
        op.Process();                                                                                       \
    } while (0)