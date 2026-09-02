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
 * \file gather_elements_tiling.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_TILING_H
#define GATHER_ELEMENTS_TILING_H
#include <cstdint>
#include "register/tilingdata_base.h"
#include "gather_elements_v2_tiling_defs.h"

namespace optiling {
struct GatherElementsCompileInfo {
    int32_t core_num;
    int32_t ub_size;
    uint64_t sysWorkspaceSize;
};

BEGIN_TILING_DATA_DEF(GatherElementsTilingData)
TILING_DATA_FIELD_DEF(int64_t, tilingMode)
// parameters of params
TILING_DATA_FIELD_DEF(int64_t, axis)
TILING_DATA_FIELD_DEF(int64_t, params_pre)
TILING_DATA_FIELD_DEF(int64_t, params_axis)
TILING_DATA_FIELD_DEF(int64_t, params_row)
TILING_DATA_FIELD_DEF(int64_t, params_total)

// parameters of indices
TILING_DATA_FIELD_DEF(int64_t, need_core_num)
TILING_DATA_FIELD_DEF(int64_t, indices_num)
TILING_DATA_FIELD_DEF(int64_t, indices_axis)
TILING_DATA_FIELD_DEF(int64_t, indices_num_each_core)
TILING_DATA_FIELD_DEF(int64_t, indices_num_remaining)
TILING_DATA_FIELD_DEF(int64_t, indices_loop_num)
TILING_DATA_FIELD_DEF(int64_t, indices_row_num_once)
TILING_DATA_FIELD_DEF(int64_t, indices_row_num_last)
TILING_DATA_FIELD_DEF(int64_t, remaining_block_remain)
TILING_DATA_FIELD_DEF(int64_t, remaining_block_num)

// parameters of x slices and indices slices
TILING_DATA_FIELD_DEF(int64_t, slice_thickness_once)
TILING_DATA_FIELD_DEF(int64_t, slice_num)
TILING_DATA_FIELD_DEF(int64_t, slice_thickness_last)

// parameters of indices slices
TILING_DATA_FIELD_DEF(int64_t, indices_slice_thickness_dim1)
TILING_DATA_FIELD_DEF(int64_t, indices_slice_thickness_dim1_last)
TILING_DATA_FIELD_DEF(int64_t, indices_slice_num_dim1)

// shape of params
TILING_DATA_FIELD_DEF(int64_t, params_shape_0)
TILING_DATA_FIELD_DEF(int64_t, params_shape_1)
TILING_DATA_FIELD_DEF(int64_t, params_shape_2)
TILING_DATA_FIELD_DEF(int64_t, params_shape_3)
TILING_DATA_FIELD_DEF(int64_t, params_shape_4)
TILING_DATA_FIELD_DEF(int64_t, params_shape_5)
TILING_DATA_FIELD_DEF(int64_t, params_shape_6)
TILING_DATA_FIELD_DEF(int64_t, params_shape_7)

// shape of indices
TILING_DATA_FIELD_DEF(int64_t, indices_shape_0)
TILING_DATA_FIELD_DEF(int64_t, indices_shape_1)
TILING_DATA_FIELD_DEF(int64_t, indices_shape_2)
TILING_DATA_FIELD_DEF(int64_t, indices_shape_3)
TILING_DATA_FIELD_DEF(int64_t, indices_shape_4)
TILING_DATA_FIELD_DEF(int64_t, indices_shape_5)
TILING_DATA_FIELD_DEF(int64_t, indices_shape_6)
TILING_DATA_FIELD_DEF(int64_t, indices_shape_7)

// binary
TILING_DATA_FIELD_DEF(int64_t, dims)

TILING_DATA_FIELD_DEF(int64_t, repeat_per_core)
TILING_DATA_FIELD_DEF(int64_t, rounds)
TILING_DATA_FIELD_DEF(int64_t, rounds_tail)

TILING_DATA_FIELD_DEF(int64_t, dbFlag)

// v2 dispatch fields
TILING_DATA_FIELD_DEF(int64_t, useV2)
TILING_DATA_FIELD_DEF(int64_t, v2Mode)
TILING_DATA_FIELD_DEF_STRUCT(GatherElementsV2TilingData, v2Data)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GatherElements, GatherElementsTilingData)

// common information
struct CommonInformation {
    int64_t indices_pre;
    int64_t params_except_pre_size;
    int32_t params_block_num;
    int32_t indices_block_num;
    int32_t large_num_per_block;
    int32_t indices_block_num_large;
    int64_t params_total_ceil;
    int64_t params_total_ceil_size;
    int32_t params_dsize;
    int32_t indices_dsize;
    int64_t task_num;
};
} // namespace optiling
#endif // GATHER_ELEMENTS_TILING_H
