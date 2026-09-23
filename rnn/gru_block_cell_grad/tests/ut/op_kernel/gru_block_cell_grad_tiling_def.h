/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRU_BLOCK_CELL_GRAD_TILING_DEF_H_
#define GRU_BLOCK_CELL_GRAD_TILING_DEF_H_

#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/gru_block_cell_grad_tiling_struct.h"

#define __CCE_UT_TEST__

inline void InitGruBlockCellGradTilingData(const uint8_t* tiling, GruBlockCellGradTilingData* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(GruBlockCellGradTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingPointer) \
    tilingStruct tilingData;                                                 \
    InitGruBlockCellGradTilingData(tilingPointer, &(tilingData))

#define REGISTER_TILING_DEFAULT(tilingStruct)

#endif // GRU_BLOCK_CELL_GRAD_TILING_DEF_H_
