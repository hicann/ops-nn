/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file transpose_batch_mat_mul_common.h
 * \brief
 */
#ifndef __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_COMMON_H__
#define __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_COMMON_H__

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"

namespace optiling {
namespace transpose_batch_mat_mul_advanced {

constexpr size_t HF32_ATTR_NUM = 4;
constexpr size_t HF32_ATTR_INDEX = 3;
constexpr size_t ALLOW_DIM = 3;
constexpr size_t BATCH_IDX = 0;
constexpr size_t M_IDX = 1;
constexpr size_t KA_IDX = 2;
constexpr size_t KB_IDX = 1;
constexpr size_t N_IDX = 2;
constexpr size_t BIAS_IDX = 2;
constexpr size_t SCALE_IDX = 3;

}
} // namespace
#endif // __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_COMMON_H__