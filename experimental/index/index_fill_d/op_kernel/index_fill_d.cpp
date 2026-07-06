/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_fill_d.cpp
 * \brief
 */

#include "index_fill_d.h"

#define DOUBLE_BUFFER_NUM 2
#define SINGLE_BUFFER_NUM 1

enum class IndexFillDTilingKey : uint32_t {
    TILING_KEY_DB = 0,
    TILING_KEY_NDB = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void index_fill_d(GM_ADDR x, GM_ADDR assist1, GM_ADDR assist2, GM_ADDR y, GM_ADDR workspace,
                                        GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(IndexFillDTilingData);
    GET_TILING_DATA_WITH_STRUCT(IndexFillDTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (schMode == static_cast<uint32_t>(IndexFillDTilingKey::TILING_KEY_DB)) {
        if constexpr (AscendC::IsSameType<DTYPE_X, bool>::value) {
            MyIndexFillD::KernelIndexFillD<int8_t, int8_t, DOUBLE_BUFFER_NUM> op;
            op.Init(x, assist1, assist2, y, &tilingData, &pipe); // 算子kernel实例初始化
            op.Process();
        } else {
            MyIndexFillD::KernelIndexFillD<DTYPE_X, DTYPE_Y, DOUBLE_BUFFER_NUM> op;
            op.Init(x, assist1, assist2, y, &tilingData, &pipe); // 算子kernel实例初始化
            op.Process();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(IndexFillDTilingKey::TILING_KEY_NDB)) {
        if constexpr (AscendC::IsSameType<DTYPE_X, bool>::value) {
            MyIndexFillD::KernelIndexFillD<int8_t, int8_t, SINGLE_BUFFER_NUM> op;
            op.Init(x, assist1, assist2, y, &tilingData, &pipe); // 算子kernel实例初始化
            op.Process();
        } else {
            MyIndexFillD::KernelIndexFillD<DTYPE_X, DTYPE_Y, SINGLE_BUFFER_NUM> op;
            op.Init(x, assist1, assist2, y, &tilingData, &pipe); // 算子kernel实例初始化
            op.Process();
        }
    }
}