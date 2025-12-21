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
 * \file sparse_tensor_dense_mat_mul_apt.cpp
 * \brief
 */
#include "arch35/sparse_tensor_dense_mat_mul_b16.h"
#include "arch35/sparse_tensor_dense_mat_mul_b32.h"
#include "arch35/sparse_tensor_dense_mat_mul_zeroing.h"

#define TILING_KEY_B16_ADJA_ADJB 10001
#define TILING_KEY_B16_ADJA_NOT_ADJB 10002
#define TILING_KEY_B16_NOT_ADJA_ADJB 10003
#define TILING_KEY_B16_NOT_ADJA_NOT_ADJB 10004
#define TILING_KEY_B32_ADJA_ADJB 20001
#define TILING_KEY_B32_ADJA_NOT_ADJB 20002
#define TILING_KEY_B32_NOT_ADJA_ADJB 20003
#define TILING_KEY_B32_NOT_ADJA_NOT_ADJB 20004
#define TILING_KEY_EMPTY_Y 30000
#define TILING_KEY_ZEROING_Y 40000

using namespace SparseTensorDenseMatMul;

extern "C" __global__ __aicore__ void sparse_tensor_dense_mat_mul(
    GM_ADDR x1_indices, GM_ADDR x1_values, GM_ADDR x1_shape, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(SparseTensorDenseMatMulTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseTensorDenseMatMulTilingData, tilingData, tiling);

    SetSysWorkspace(workspace);
    GM_ADDR usrWorkSpace = AscendC::GetUserWorkspace(workspace);

    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_B16_ADJA_ADJB)) {
        SparseTensorDenseMatMulB16<DTYPE_X1_INDICES, DTYPE_X1_VALUES, float, true, true> matmulOp(&pipe, &tilingData);
        matmulOp.Init(x1_indices, x1_values, x2, y, usrWorkSpace);
        matmulOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B16_ADJA_NOT_ADJB)) {
        SparseTensorDenseMatMulB16<DTYPE_X1_INDICES, DTYPE_X1_VALUES, float, true, false> matmulOp(&pipe, &tilingData);
        matmulOp.Init(x1_indices, x1_values, x2, y, usrWorkSpace);
        matmulOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B16_NOT_ADJA_ADJB)) {
        SparseTensorDenseMatMulB16<DTYPE_X1_INDICES, DTYPE_X1_VALUES, float, false, true> matmulOp(&pipe, &tilingData);
        matmulOp.Init(x1_indices, x1_values, x2, y, usrWorkSpace);
        matmulOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B16_NOT_ADJA_NOT_ADJB)) {
        SparseTensorDenseMatMulB16<DTYPE_X1_INDICES, DTYPE_X1_VALUES, float, false, false> matmulOp(&pipe, &tilingData);
        matmulOp.Init(x1_indices, x1_values, x2, y, usrWorkSpace);
        matmulOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B32_ADJA_ADJB)) {
        SparseTensorDenseMatMulB32<DTYPE_X1_INDICES, DTYPE_X1_VALUES, true, true> matmulOp(&pipe, &tilingData);
        matmulOp.Init(x1_indices, x1_values, x2, y);
        matmulOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B32_ADJA_NOT_ADJB)) {
        SparseTensorDenseMatMulB32<DTYPE_X1_INDICES, DTYPE_X1_VALUES, true, false> matmulOp(&pipe, &tilingData);
        matmulOp.Init(x1_indices, x1_values, x2, y);
        matmulOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B32_NOT_ADJA_ADJB)) {
        SparseTensorDenseMatMulB32<DTYPE_X1_INDICES, DTYPE_X1_VALUES, false, true> matmulOp(&pipe, &tilingData);
        matmulOp.Init(x1_indices, x1_values, x2, y);
        matmulOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B32_NOT_ADJA_NOT_ADJB)) {
        SparseTensorDenseMatMulB32<DTYPE_X1_INDICES, DTYPE_X1_VALUES, false, false> matmulOp(&pipe, &tilingData);
        matmulOp.Init(x1_indices, x1_values, x2, y);
        matmulOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_EMPTY_Y)){
        // 对于输出y为空Tensor的情况，什么也不用做
    } else if (TILING_KEY_IS(TILING_KEY_ZEROING_Y)){
        SparseTensorDenseMatMulZeroing<DTYPE_X1_VALUES> matmulOp(&pipe, &tilingData);
        matmulOp.InitAndProcessZeroing(y);
    }
    pipe.Reset();
}
