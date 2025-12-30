/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_tiling.h
 * \brief
 */
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_RUNTIME2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_RUNTIME2_H

namespace optiling {
struct ScatterElementsCompileInfo {
  int32_t ub_size_bytes;
  int32_t available_aicore_num;
  int64_t coreNum{0};
  int64_t ubSize{0};
};

struct ScatterElementsTilingParams {
  int32_t dims_indices;
  int32_t dims_data;
  int32_t shape_acc_data;
  int32_t shape_acc_indices;
  int32_t shape_indices_0;
  int32_t shape_indices_1;
  int32_t shape_indices_2;
  int32_t shape_indices_3;
  int32_t shape_indices_4;
  int32_t shape_indices_5;
  int32_t shape_indices_6;
  int32_t shape_indices_7;
  int32_t shape_data_0;
  int32_t shape_data_1;
  int32_t shape_data_2;
  int32_t shape_data_3;
  int32_t shape_data_4;
  int32_t shape_data_5;
  int32_t shape_data_6;
  int32_t shape_data_7;
  int32_t data_block;
  int32_t repeat;
  int32_t used_aicore_num;
  int32_t batch_num_per_aicore;
  int32_t batch_tail;
  int32_t indices_block;
  int32_t indices_repeat;
  int32_t rounds_indices;
  int32_t tail_indices;
  int32_t tail_indices_block;
  int32_t tail_indices_repeat;
  int32_t updates_block;
  int32_t updates_repeat;
  int32_t tail_updates_block;
  int32_t tail_updates_repeat;
  int32_t tiling_mode;
  int32_t params_row_data;
  int32_t params_row_indices;
  int32_t params_axis_data;
  int32_t params_axis_indices;
  int32_t axis;
  int32_t rounds;
  int32_t data_tail_repeat;
  int32_t repeat_per_core;
  int32_t rounds_tail;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_RUNTIME2_H