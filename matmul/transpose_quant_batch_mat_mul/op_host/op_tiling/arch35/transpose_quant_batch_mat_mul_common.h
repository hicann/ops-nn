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
 * \file transpose_quant_batch_mat_mul_common.h
 * \brief
 */
#ifndef __OP_HOST_TRANSPOSE_QUANT_BATCH_MAT_MUL_COMMON_H__
#define __OP_HOST_TRANSPOSE_QUANT_BATCH_MAT_MUL_COMMON_H__

namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {

constexpr size_t BATCH_IDX = 0;
constexpr size_t X1_IDX = 0;
constexpr size_t X2_IDX = 1;
constexpr size_t M_IDX = 1;
constexpr size_t KA_IDX = 2;
constexpr size_t KB_IDX = 1;
constexpr size_t N_IDX = 2;
constexpr size_t BIAS_IDX = 2;
constexpr size_t SCALE_X1_IDX = 3;
constexpr size_t SCALE_X2_IDX = 4;
constexpr size_t PERM_X1_IDX = 2;
constexpr size_t PERM_X2_IDX = 3;
constexpr size_t PERM_Y_IDX = 4;
constexpr size_t ATTR_NUM = 6;
constexpr size_t ALLOW_DIM = 3;
constexpr size_t EXPECTED_SCALE_DIM = 1;
constexpr size_t TQBMM_VALID_K = 512;
constexpr size_t TQBMM_VALID_N = 128;
constexpr uint64_t  L1_ALIGN_SIZE = 32;
constexpr uint64_t  L2_ALIGN_SIZE = 128;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32UL;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32UL;
constexpr uint64_t NUM_HALF = 2;
constexpr uint64_t BASEM_BASEN_RATIO = 2;
constexpr uint32_t DOUBLE_CORE_NUM = 2;
constexpr uint64_t BASEK_LIMIT = 4095;
} // namespace transpose_quant_batch_mat_mul_advanced
} // namespace optiling
#endif // __OP_HOST_TRANSPOSE_QUANT_BATCH_MAT_MUL_COMMON_H__