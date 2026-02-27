/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_quant_batch_mat_mul_tiling_key_public.h
 * \brief
 */

#ifndef OP_KERNEL_TQBMM_TILING_KEY_PUBLIC_H
#define OP_KERNEL_TQBMM_TILING_KEY_PUBLIC_H
#include <cstdint>

#define TRANSPOSE_QUANT_BATCH_MAT_MUL_BATCH_SPLIT_FALSE 0
#define TRANSPOSE_QUANT_BATCH_MAT_MUL_BATCH_SPLIT_TRUE 1

#define TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X1_0_1_2 0
#define TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X1_1_0_2 1

#define TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X2_0_1_2 0
#define TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X2_0_2_1 1

enum class TQBMMPermX1 : std::uint8_t {
    PERM_X1_0_1_2 = TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X1_0_1_2,
    PERM_X1_1_0_2 = TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X1_1_0_2,
};

enum class TQBMMPermX2 : std::uint8_t {
    PERM_X2_0_1_2 = TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X2_0_1_2,
    PERM_X2_0_2_1 = TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X2_0_2_1,
};

enum class TQBMMBatchSplit : std::uint8_t {
    BATCH_SPLIT_FALSE = TRANSPOSE_QUANT_BATCH_MAT_MUL_BATCH_SPLIT_FALSE,
    BATCH_SPLIT_TRUE = TRANSPOSE_QUANT_BATCH_MAT_MUL_BATCH_SPLIT_TRUE
};

#endif
