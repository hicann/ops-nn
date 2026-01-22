/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_ELEMENTS_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_ELEMENTS_TILING_H_
#include <cstdint>
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
struct GatherElementsCompileInfo {
  int32_t core_num;
  int32_t ub_size;
  int32_t params_dsize;
  int32_t indices_dsize;
  int32_t support_vgather;
  int32_t infer_vgather;
};

struct GatherElementsTilingParams {
  int64_t tilingMode;
  // parameters of params
  int64_t axis;
  int64_t params_pre;
  int64_t params_axis;
  int64_t params_row;
  int64_t params_total;

  // parameters of indices
  int64_t need_core_num;
  int64_t indices_num;
  int64_t indices_axis;
  int64_t indices_num_each_core;
  int64_t indices_num_remaining;
  int64_t indices_loop_num;
  int64_t indices_row_num_once;
  int64_t indices_row_num_last;
  int64_t remaining_block_remain;
  int64_t remaining_block_num;

  // parameters of x slices and indices slices
  int64_t slice_thickness_once;
  int64_t slice_num;
  int64_t slice_thickness_last;

  // parameters of indices slices
  int64_t indices_slice_thickness_dim1;
  int64_t indices_slice_thickness_dim1_last;
  int64_t indices_slice_num_dim1;

  // shape of params
  int64_t params_shape_0;
  int64_t params_shape_1;
  int64_t params_shape_2;
  int64_t params_shape_3;
  int64_t params_shape_4;
  int64_t params_shape_5;
  int64_t params_shape_6;
  int64_t params_shape_7;

  // shape of indices
  int64_t indices_shape_0;
  int64_t indices_shape_1;
  int64_t indices_shape_2;
  int64_t indices_shape_3;
  int64_t indices_shape_4;
  int64_t indices_shape_5;
  int64_t indices_shape_6;
  int64_t indices_shape_7;
  
  // binary
  int64_t dims;

  int64_t repeat_per_core;
  int64_t rounds;
  int64_t rounds_tail;

  int64_t dbFlag=0;
};

// common information
struct CommonInformation{
  int64_t indices_pre;
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
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GATHER_ELEMENTS_TILING_H_
