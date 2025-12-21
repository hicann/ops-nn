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
 * \file sparse_to_dense.cpp
 * \brief
 */
#include "arch35/sparse_to_dense_simt.h"

using namespace AscendC;
using namespace SparseToDense;

#define TILING_KEY_DATA_NUM_UINT32_TENSOR 1000000
#define TILING_KEY_DATA_NUM_UINT32_SCALAR 1000001
#define TILING_KEY_DATA_NUM_UINT64_TENSOR 1000010
#define TILING_KEY_DATA_NUM_UINT64_SCALAR 1000011

extern "C" __global__ __aicore__ void sparse_to_dense(GM_ADDR indices, GM_ADDR output_shape, GM_ADDR values, 
                                            GM_ADDR default_value, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(SparseToDenseTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_DATA_NUM_UINT32_TENSOR)) {
        SparseToDenseSimt<DTYPE_INDICES, DTYPE_Y, uint32_t, false> op(pipe, tilingData);
        op.Init(indices, output_shape, values, default_value, y);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DATA_NUM_UINT32_SCALAR)) {
        SparseToDenseSimt<DTYPE_INDICES, DTYPE_Y, uint32_t, true> op(pipe, tilingData);
        op.Init(indices, output_shape, values, default_value, y);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DATA_NUM_UINT64_TENSOR)) {
        SparseToDenseSimt<DTYPE_INDICES, DTYPE_Y, uint64_t, false> op(pipe, tilingData);
        op.Init(indices, output_shape, values, default_value, y);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DATA_NUM_UINT64_SCALAR)) {
        SparseToDenseSimt<DTYPE_INDICES, DTYPE_Y, uint64_t, true> op(pipe, tilingData);
        op.Init(indices, output_shape, values, default_value, y);
        op.Process();
    }
}