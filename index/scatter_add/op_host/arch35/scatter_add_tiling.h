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
 * \file scatter_add_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_H_

#include <sstream>
#include <cctype>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"

namespace optiling {
const int64_t BLOCK_SIZE = 32;
const int64_t OUT_SPECIAL_DIM_0 = 163623;
const int64_t OUT_SPECIAL_DIM_1 = 1;
const int64_t OUT_SPECIAL_DIM_2 = 80;
const int64_t OUT_SPECIAL_DIM_3 = 21340;
const int64_t OUT_SPECIAL_DIM_4 = 12828;
const int64_t DATA_TWO = 2;
// 32b aligned, ub can store all updatesNum, float32 atomic
const int64_t TILING_MODE_1 = 1;
// 32b aligned, ub can't store all updatesNum, float32 atomic
const int64_t TILING_MODE_2 = 2;
// updateDataNum is less than 1 block, ub can store all updatesNum, float32 atomic
const int64_t TILING_MODE_3 = 3;
// updateDataNum is less than 1 block, ub can't store all updatesNum, float32 atomic
const int64_t TILING_MODE_4 = 4;
// updateDataNum is more than 1 block, float32 atomic
const int64_t TILING_MODE_5 = 5;
// 32b aligned, ub can store all var and updates, not atomic
const int64_t TILING_MODE_6 = 6;
// 32b aligned, ub can store all var, not atomic
const int64_t TILING_MODE_7 = 7;
// 32b aligned, ub can store all updates, not atomic
const int64_t TILING_MODE_8 = 8;
// 32b aligned, ub can't store all var and updates, not atomic
const int64_t TILING_MODE_9 = 9;
// updateDataNum is less than 1 block, ub can store all var and updates, not atomic
const int64_t TILING_MODE_10 = 10;
// updateDataNum is less than 1 block, ub can store all var, not atomic
const int64_t TILING_MODE_11 = 11;
// updateDataNum is less than 1 block, ub can store all updates, not atomic
const int64_t TILING_MODE_12 = 12;
// updateDataNum is less than 1 block, ub can't store all var and updates, not atomic
const int64_t TILING_MODE_13 = 13;
// updateDataNum is more than 1 block, not atomic, and less than updateubnum
const int64_t TILING_MODE_14 = 14;
// updateDataNum is more than 1 block, not atomic, and more than updateubnum
const int64_t TILING_MODE_15 = 15;
// high perf branch, updateDataNum is less than 1 block
const int64_t TILING_MODE_16 = 16;
// high perf branch, updateDataNum is 32b aligned
const int64_t TILING_MODE_17 = 17;
// div 0 check
const int64_t ZERO = 0;

struct ScatterAddCompileInfo {
  int64_t core_num{1};
  int64_t ub_size{1};
  int64_t var_size{1};
  int64_t indices_size{1};
  int64_t support_atomic{1};
};

struct ScatterAddCalcShapeInfo {
  gert::Shape var_shape;
  gert::Shape indices_shape;
  gert::Shape updates_shape;
  gert::Shape out_shape;
};

struct ScatterAddTilingParams {
  int64_t tiling_mode;
  int64_t indice_step;
  int64_t core_num;
  int64_t updates_data_num;
  int64_t indices_loop_num;
  int64_t indices_last_num;
  int64_t updates_num;
  int64_t updates_loop_num;
  int64_t updates_last_num;
  int64_t var_num;
  int64_t var_loop_num;
  int64_t var_last_num;
  int64_t var_each_core_burst_len;
  int64_t var_last_core_burst_len;
  int64_t max_indice;
  int64_t var_each_core_data;
  int64_t indices_each_core_data;
  int64_t indices_last_core_data;
  int64_t each_core_indices_loop_num;
  int64_t each_core_indices_last_num;
  int64_t last_core_indices_loop_num;
  int64_t last_core_indices_last_num;
  int64_t tiling_core_num;
  void Init() {
    tiling_mode = TILING_MODE_1;
    indice_step = 0;
    core_num = 0;
    updates_data_num = 0;
    indices_loop_num = 0;
    indices_last_num = 0;
    updates_num = 0;
    updates_loop_num = 0;
    updates_last_num = 0;
    var_num = 0;
    var_loop_num = 0;
    var_last_num = 0;
    var_each_core_burst_len = 0;
    var_last_core_burst_len = 0;
    max_indice = 0;
    var_each_core_data = 0;
    indices_each_core_data = 0;
    indices_last_core_data = 0;
    each_core_indices_loop_num = 0;
    each_core_indices_last_num = 0;
    last_core_indices_loop_num = 0;
    last_core_indices_last_num = 0;
    tiling_core_num = 0;
  }
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_H_