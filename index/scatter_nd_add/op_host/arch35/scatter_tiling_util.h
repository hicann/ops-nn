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
 * \file scatter_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_UTIL_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_UTIL_H_

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace optiling {
const int64_t BLOCK_SIZE = 32;
// addDataNum is 32b aligned, ub can store all adds_num
const int64_t TILING_MODE_1 = 1;
// addDataNum is 32b aligned, ub can't store all adds_num
const int64_t TILING_MODE_2 = 2;
// addDataNum isn't 32b aligned and less than 1 block
// ub can store all adds_num
const int64_t TILING_MODE_3 = 3;
// addDataNum isn't 32b aligned and less than 1 block
// ub can't store all adds_num
const int64_t TILING_MODE_4 = 4;
// addDataNum isn't 32b aligned and more than 1 block
const int64_t TILING_MODE_5 = 5;
// highperm branch
const int64_t TILING_MODE_6 = 6;
const int64_t TILING_MODE_7 = 7;
const int64_t UPDATE_ROW_THRESHOLD = 2048;

struct ScatterCompileInfo {
  int64_t core_num;
  int64_t ub_size;
  int64_t var_size;
  int64_t indices_size;
  int64_t support_atomic;
};

struct CalcShapeInfo {
  gert::Shape var_shape;
  gert::Shape indices_shape;
  gert::Shape adds_shape;
  gert::Shape out_shape;
};

/*
 * @brief: the compile info parse func
 * @param [in] context: TilingParaseContext
 * @return ge::graphStatus: ge::graphStatus
 */
ge::graphStatus TilingPrepare4Scatter(gert::TilingParseContext* context);

/*
 * @brief: cehck scatter ops tensor shape
 * @param [in] context: TilingParaseContext
 * @param [in] varShape: varShape
 * @param [in] indicesShape: indicesShape
 * @param [in] addsShape: addsShape
 * @param [in] outShape: outShape
 * @return bool: bool
 */
bool CheckScatterOpsTensorShape(const gert::TilingContext* context, const CalcShapeInfo& calc_shape_info);

struct ScatterNdAddTilingParams {
  int64_t var_offset[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t tiling_mode;
  int64_t indice_step;
  int64_t core_num;
  int64_t adds_data_num;
  int64_t indices_loop_num;
  int64_t indices_last_num;
  int64_t adds_num;
  int64_t adds_loop_num;
  int64_t adds_last_num;
  int64_t indices_last_dim;
  int64_t indices_front_dim;
  int64_t tiling_core_num;

  void Init() {
    tiling_mode = TILING_MODE_1;
    indice_step = 0;
    core_num = 0;
    adds_data_num = 0;
    indices_loop_num = 0;
    indices_last_num = 0;
    adds_num = 0;
    adds_loop_num = 0;
    adds_last_num = 0;
    indices_last_dim = 0;
    indices_front_dim = 0;
    tiling_core_num = 0;
  }
};

struct ScatterNdAddTilingParamsHP {
  int64_t var_offset[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t tiling_mode;
  int64_t core_num;
  int64_t indices_last_dim;
  int64_t updates_data_num;
  int64_t indices_each_core_data;
  int64_t indices_last_core_data;
  int64_t each_core_indices_loop_num;
  int64_t each_core_indices_last_num;
  int64_t last_core_indices_loop_num;
  int64_t last_core_indices_last_num;
  int64_t updates_loop_num;
  int64_t updates_last_num;
  int64_t tiling_core_num;

  void Init() {
    tiling_mode = TILING_MODE_6;
    core_num = 0;
    indices_last_dim = 0;
    updates_data_num = 0;
    indices_each_core_data = 0;
    indices_last_core_data = 0;
    each_core_indices_loop_num = 0;
    each_core_indices_last_num = 0;
    last_core_indices_loop_num = 0;
    last_core_indices_last_num = 0;
    updates_loop_num = 0;
    updates_last_num = 0;
    tiling_core_num = 0;
  }
};

struct ScatterNonAliasingAddTilingParams {
  int64_t tiling_mode;
  int64_t indice_step;
  int64_t core_num;
  int64_t adds_data_num;
  int64_t indices_loop_num;
  int64_t indices_last_num;
  int64_t adds_num;
  int64_t adds_loop_num;
  int64_t adds_last_num;
  int64_t var_offset[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t indices_last_dim;
  int64_t indices_front_dim;
  int64_t var_num;
  int64_t tiling_core_num;

  void Init() {
    tiling_mode = TILING_MODE_1;
    indices_last_num = 0;
    adds_num = 0;
    adds_loop_num = 0;
    adds_last_num = 0;
    var_num = 0;
    tiling_core_num = 0;
    indice_step = 0;
    core_num = 0;
    adds_data_num = 0;
    indices_loop_num = 0;
  }
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_UTIL_H_