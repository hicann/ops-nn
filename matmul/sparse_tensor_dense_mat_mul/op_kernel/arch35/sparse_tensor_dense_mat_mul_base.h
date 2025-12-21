/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_tensor_dense_mat_mul_base.h
 * \brief
 */
#ifndef SPARSE_TENSOR_DENSE_MAT_MUL_BASE_H
#define SPARSE_TENSOR_DENSE_MAT_MUL_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sparse_tensor_dense_mat_mul_tiling_def.h"

namespace SparseTensorDenseMatMul {

constexpr int64_t SIMT_MAX_THREAD_NUM = 2048;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t INDICES_DIM_1 = 2;

} // namespace SparseTensorDenseMatMul

#endif // SPARSE_TENSOR_DENSE_MAT_MUL_BASE_H
