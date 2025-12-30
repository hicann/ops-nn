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
 * \file scatter_add_tiling.cpp
 * \brief
 */
#include "scatter_add_tiling.h"
#include "scatter_add_tiling_base.h"

namespace optiling {
static constexpr size_t SC_IN_IDX_VAR = 0;
static constexpr size_t SC_IN_IDX_INDICES = 1;
static constexpr size_t SC_IN_IDX_UPDATES = 2;
static constexpr size_t SC_OUT_IDX_OUT = 0;
static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;

struct CalParams {
  int64_t indices_num;
  int64_t updates_num;
  int64_t updates_data_num;
  int64_t ub_size;
  int64_t var_size;
  int64_t indices_size;
  int64_t var_data_each_block;
  int64_t var_num;
  int64_t core_num;
  int64_t max_indice;
  int64_t add_data_num;
  int64_t var_dim_num;
  int64_t support_atomic;
};

// -----------------ScatterAdd Util START------------------
static ge::graphStatus TilingPrepare4ScatterAdd(gert::TilingParseContext* context) {
  auto compile_info = context->GetCompiledInfo<ScatterAddCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4ScatterAdd GRAPH_SUCESS.");
  return ge::GRAPH_SUCCESS;
}

// -----------------ScatterAdd Util END------------------

void CalAtomicBranchRunningParams(gert::TilingContext* context, ScatterAddTilingParams* tiling_data_ptr,
                                  CalParams& cal_params) {
  int64_t update_size_byte = cal_params.var_size * cal_params.updates_num;
  int64_t half_ub_size = cal_params.ub_size / 2;
  OP_CHECK_IF(half_ub_size == 0,
                  OP_LOGE(context->GetNodeName(), "half_ub_size = 0 is not support"), return);
  OP_CHECK_IF(cal_params.var_size == 0,
                  OP_LOGE(context->GetNodeName(), "var_size = 0 is not support"), return);
  OP_CHECK_IF(cal_params.indices_size == 0,
                  OP_LOGE(context->GetNodeName(), "indices_size = 0 is not support"), return);
  tiling_data_ptr->updates_loop_num = cal_params.updates_data_num / (half_ub_size / cal_params.var_size);
  tiling_data_ptr->updates_last_num = cal_params.updates_data_num % (half_ub_size / cal_params.var_size);
  tiling_data_ptr->indices_loop_num = cal_params.indices_num / (half_ub_size / cal_params.indices_size);
  tiling_data_ptr->indices_last_num = cal_params.indices_num % (half_ub_size / cal_params.indices_size);
  tiling_data_ptr->updates_data_num = cal_params.updates_data_num;
  tiling_data_ptr->updates_num = cal_params.updates_num;

  if (cal_params.updates_data_num % cal_params.var_data_each_block == 0) {
    tiling_data_ptr->tiling_mode = TILING_MODE_1;
    if (update_size_byte > half_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_2;
    }
  } else {
    tiling_data_ptr->tiling_mode = TILING_MODE_5;
    if (cal_params.updates_data_num < cal_params.var_data_each_block) {
      tiling_data_ptr->tiling_mode = TILING_MODE_4;
      if (update_size_byte <= half_ub_size) {
        tiling_data_ptr->updates_loop_num = cal_params.updates_num / (half_ub_size / cal_params.var_size);
        tiling_data_ptr->updates_last_num = cal_params.updates_num % (half_ub_size / cal_params.var_size);
      }
    }
  }
}

void CalNotAtomicBranchRunningParams(gert::TilingContext* context, ScatterAddTilingParams* tiling_data_ptr,
                                     CalParams& cal_params) {
  int64_t var_all_size_byte = cal_params.var_size * cal_params.var_num;
  int64_t var_size_byte = cal_params.var_size * tiling_data_ptr->indice_step * cal_params.updates_data_num;
  int64_t update_size_byte = cal_params.var_size * cal_params.updates_num;
  int64_t var_ub_size = cal_params.ub_size / 8 * 3;
  int64_t indices_ub_size = cal_params.ub_size / 8 * 2;
  OP_CHECK_IF(cal_params.var_size == 0,
                  OP_LOGE(context->GetNodeName(), "var_size = 0 is not support"), return);
  OP_CHECK_IF(cal_params.indices_size == 0,
                  OP_LOGE(context->GetNodeName(), "indices_size = 0 is not support"), return);
  OP_CHECK_IF(var_ub_size == 0,
                  OP_LOGE(context->GetNodeName(), "var_ub_size = 0 is not support"), return);
  OP_CHECK_IF(indices_ub_size == 0,
                  OP_LOGE(context->GetNodeName(), "indices_ub_size = 0 is not support"),
                  return);
  OP_CHECK_IF(cal_params.var_data_each_block == 0,
                  OP_LOGE(context->GetNodeName(), "var_data_each_block = 0 is not support"),
                  return);
  tiling_data_ptr->var_loop_num = cal_params.var_num / (var_ub_size / cal_params.var_size);
  tiling_data_ptr->var_last_num = cal_params.var_num % (var_ub_size / cal_params.var_size);
  tiling_data_ptr->updates_loop_num = cal_params.updates_data_num / (var_ub_size / cal_params.var_size);
  tiling_data_ptr->updates_last_num = cal_params.updates_data_num % (var_ub_size / cal_params.var_size);
  tiling_data_ptr->indices_loop_num = cal_params.indices_num / (indices_ub_size / cal_params.indices_size);
  tiling_data_ptr->indices_last_num = cal_params.indices_num % (indices_ub_size / cal_params.indices_size);
  tiling_data_ptr->updates_data_num = cal_params.updates_data_num;
  tiling_data_ptr->updates_num = cal_params.updates_num;
  tiling_data_ptr->var_num = cal_params.var_num;

  if (cal_params.updates_data_num % cal_params.var_data_each_block == 0) {
    if (update_size_byte <= var_ub_size && var_size_byte <= var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_6;
    } else if (update_size_byte > var_ub_size && var_size_byte <= var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_7;
    } else if (update_size_byte <= var_ub_size && var_size_byte > var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_8;
    } else {
      tiling_data_ptr->tiling_mode = TILING_MODE_9;
    }
  } else if (cal_params.updates_data_num < cal_params.var_data_each_block) {
    if (update_size_byte <= var_ub_size && var_all_size_byte <= var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_10;
    } else if (update_size_byte > var_ub_size && var_all_size_byte <= var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_11;
    } else if (update_size_byte <= var_ub_size && var_all_size_byte > var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_12;
    } else {
      tiling_data_ptr->tiling_mode = TILING_MODE_13;
    }
  } else {
    if (cal_params.updates_data_num / (var_ub_size / cal_params.var_size) == 0) {
      tiling_data_ptr->tiling_mode = TILING_MODE_14;
    } else {
      tiling_data_ptr->tiling_mode = TILING_MODE_15;
    }
  }

  if (tiling_data_ptr->tiling_mode == TILING_MODE_6 || tiling_data_ptr->tiling_mode == TILING_MODE_7) {
    tiling_data_ptr->var_each_core_data = tiling_data_ptr->indice_step * tiling_data_ptr->updates_data_num;
    int64_t var_last_core_data = cal_params.var_num - tiling_data_ptr->var_each_core_data * (cal_params.core_num - 1);
    tiling_data_ptr->var_each_core_burst_len = tiling_data_ptr->var_each_core_data / cal_params.var_data_each_block;
    tiling_data_ptr->var_last_core_burst_len = var_last_core_data / cal_params.var_data_each_block;
  }
  if (tiling_data_ptr->tiling_mode == TILING_MODE_9 || tiling_data_ptr->tiling_mode == TILING_MODE_14 ||
      tiling_data_ptr->tiling_mode == TILING_MODE_15) {
    tiling_data_ptr->var_loop_num = cal_params.updates_data_num / (var_ub_size / cal_params.var_size);
    tiling_data_ptr->var_last_num = cal_params.updates_data_num % (var_ub_size / cal_params.var_size);
  }
}

void CalScatterAddHighPerfBranchParams(gert::TilingContext* context, ScatterAddTilingParams* tiling_data_ptr,
                                       CalParams& cal_params) {
  int64_t half_ub_size = cal_params.ub_size / DATA_TWO;
  OP_CHECK_IF(half_ub_size == 0,
                  OP_LOGE(context->GetNodeName(), "half_ub_size = 0 is not support"), return);
  OP_CHECK_IF(cal_params.core_num == 0,
                  OP_LOGE(context->GetNodeName(), "core_num = 0 is not support"), return);
  OP_CHECK_IF(cal_params.indices_size == 0,
                  OP_LOGE(context->GetNodeName(), "indices_size = 0 is not support"), return);
  OP_CHECK_IF(cal_params.var_data_each_block == 0,
                  OP_LOGE(context->GetNodeName(), "var_data_each_block = 0 is not support"),
                  return);
  tiling_data_ptr->tiling_mode = TILING_MODE_16;
  tiling_data_ptr->updates_data_num = cal_params.updates_data_num;
  tiling_data_ptr->indices_each_core_data = ceil(float(cal_params.indices_num) / cal_params.core_num);
  tiling_data_ptr->core_num = ceil(float(cal_params.indices_num) / tiling_data_ptr->indices_each_core_data);
  tiling_data_ptr->indices_last_core_data =
      cal_params.indices_num - tiling_data_ptr->indices_each_core_data * (tiling_data_ptr->core_num - 1);
  tiling_data_ptr->each_core_indices_loop_num =
      tiling_data_ptr->indices_each_core_data / (half_ub_size / cal_params.indices_size);
  tiling_data_ptr->each_core_indices_last_num =
      tiling_data_ptr->indices_each_core_data % (half_ub_size / cal_params.indices_size);
  tiling_data_ptr->last_core_indices_loop_num =
      tiling_data_ptr->indices_last_core_data / (half_ub_size / cal_params.indices_size);
  tiling_data_ptr->last_core_indices_last_num =
      tiling_data_ptr->indices_last_core_data % (half_ub_size / cal_params.indices_size);

  if (cal_params.updates_data_num % cal_params.var_data_each_block == 0) {
    tiling_data_ptr->tiling_mode = TILING_MODE_17;
  }
}

bool CheckScatterAddShape(const gert::TilingContext* context, const ScatterAddCalcShapeInfo& calc_shape_info) {
  if (!(calc_shape_info.var_shape == calc_shape_info.out_shape)) {
    OP_LOGE(context->GetNodeName(), "the length of var must be same as the length of output.");
    return false;
  }

  int64_t var_size = calc_shape_info.var_shape.GetDimNum();
  int64_t indices_size = calc_shape_info.indices_shape.GetDimNum();
  int64_t updates_size = calc_shape_info.updates_shape.GetDimNum();
  if (indices_size == 1 && calc_shape_info.indices_shape.GetDim(0) == 1 && var_size - updates_size == 1) {
    OP_LOGI(context->GetNodeName(), "Input indices is a scalar.");
    indices_size = 0;
  }

  if (var_size + indices_size - 1 != updates_size) {
    OP_LOGE(
        context->GetNodeName(),
        "var_size and indices_size does not satisfy the relation expression with updates_size.");
    return false;
  }
  for (int64_t i = 0; i < indices_size; i++) {
    if (calc_shape_info.indices_shape.GetDim(i) != calc_shape_info.updates_shape.GetDim(i)) {
      OP_LOGE(context->GetNodeName(), "indicesShapeDim does not equal to updatesShapeDim.");
      return false;
    }
  }
  for (int64_t i = indices_size; i < updates_size; i++) {
    if (calc_shape_info.var_shape.GetDim(i - indices_size + 1) != calc_shape_info.updates_shape.GetDim(i)) {
      OP_LOGE(context->GetNodeName(), "varShapeDim does not equal to updatesShapeDim.");
      return false;
    }
  }
  return true;
}

static bool CheckScatterAddHighPerfShape(const gert::Shape& var_shape, const gert::Shape& indices_shape) {
  if ((indices_shape.GetDimNum() == DIM_NUM_1 && var_shape.GetDimNum() == DIM_NUM_2) &&
      ((var_shape.GetDim(0) == OUT_SPECIAL_DIM_0 && var_shape.GetDim(1) == OUT_SPECIAL_DIM_1) ||
       (var_shape.GetDim(0) == OUT_SPECIAL_DIM_0 && var_shape.GetDim(1) == OUT_SPECIAL_DIM_2) ||
       (var_shape.GetDim(0) == OUT_SPECIAL_DIM_3 && var_shape.GetDim(1) == OUT_SPECIAL_DIM_1) ||
       (var_shape.GetDim(0) == OUT_SPECIAL_DIM_3 && var_shape.GetDim(1) == OUT_SPECIAL_DIM_2))) {
    return true;
  }
  if ((indices_shape.GetDimNum() == DIM_NUM_1 && var_shape.GetDimNum() == DIM_NUM_1) && (var_shape.GetDim(0) == OUT_SPECIAL_DIM_4)) {
    return true;
  }
  return false;
}

bool GetScatterAddCompileParams(const gert::TilingContext* context, const ScatterAddCompileInfo* compile_info,
                                CalParams& cal_params) {
  cal_params.core_num = compile_info->core_num;
  cal_params.ub_size = compile_info->ub_size;
  cal_params.var_size = compile_info->var_size;
  cal_params.indices_size = compile_info->indices_size;
  cal_params.support_atomic = compile_info->support_atomic;
  OP_CHECK_IF(compile_info->core_num <= 0 || compile_info->ub_size <= 0 || compile_info->var_size <= 0 ||
                      compile_info->indices_size <= 0 || cal_params.support_atomic < 0,
                  OP_LOGE(
                      context->GetNodeName(),
                      "core_num, ub_size, var_size, indices_size must be greater to 0, but got %ld, %ld, %ld, %ld, %ld",
                      cal_params.core_num, cal_params.ub_size, cal_params.var_size, cal_params.indices_size,
                      cal_params.support_atomic),
                  return false);
  return true;
}

static ge::graphStatus Tiling4ScatterAdd(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "ScatterAddTiling running begin");
  auto compile_info = reinterpret_cast<const ScatterAddCompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  OP_LOGD(context->GetNodeName(), "ScatterAddTiling is ascendc. runing ascendc tiling.");
  return ScatterAddTilingForAscendC(context); 

}

// register tiling interface of the ScatterAddTiling op.
IMPL_OP_OPTILING(ScatterAdd).Tiling(Tiling4ScatterAdd).TilingParse<ScatterAddCompileInfo>(TilingPrepare4ScatterAdd);
}  // namespace optiling
