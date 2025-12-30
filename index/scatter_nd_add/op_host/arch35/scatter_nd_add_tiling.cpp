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
 * \file scatter_nd_add_tiling.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include <sstream>
#include <cctype>
#include "./scatter_tiling_util.h"
#include "scatter_nd_add_tiling_base.h"

namespace optiling {
static constexpr size_t SC_IN_VAR_IDX = 0;
static constexpr size_t SC_IN_INDICES_IDX = 1;
static constexpr size_t SC_IN_UPDATES_IDX = 2;
static constexpr size_t SC_OUT_OUT_IDX = 0;
static constexpr int64_t VAR_OFFSET_NUM = 7;

struct CalParams {
  int64_t indices_num;
  int64_t adds_num;
  int64_t add_data_num;
  int64_t max_indice;
  int64_t ub_size;
  int64_t core_num;
  int64_t var_size;
  int64_t indices_size;
  int64_t var_data_each_block;
  int64_t indices_last_dim;
  int64_t var_dim_num;
  int64_t atomic_flag;
};

static bool CheckCalParams(const gert::TilingContext* context, const CalParams& cal_params) {
  OP_CHECK_IF(cal_params.indices_size == 0,
                  OP_LOGE(context->GetNodeName(), "indices_size = 0 is not support"),
                  return false);
  OP_CHECK_IF(cal_params.core_num == 0,
                  OP_LOGE(context->GetNodeName(), "core_num = 0 is not support"), return false);
  OP_CHECK_IF(cal_params.var_size == 0,
                  OP_LOGE(context->GetNodeName(), "var_size = 0 is not support"), return false);
  OP_CHECK_IF(cal_params.var_data_each_block == 0,
                  OP_LOGE(context->GetNodeName(), "var_data_each_block = 0 is not support"),
                  return false);

  return true;
}

static bool CalRunningParams(const gert::TilingContext* context, ScatterNdAddTilingParams* tiling_data_ptr,
                             CalParams& cal_params) {
  int64_t add_size_byte = cal_params.var_size * cal_params.adds_num;
  int64_t half_ub_size = cal_params.ub_size / 2;
  OP_CHECK_IF(half_ub_size == 0,
                  OP_LOGE(context->GetNodeName(), "half_ub_size = 0 is not support"),
                  return true);
  OP_CHECK_IF(!CheckCalParams(context, cal_params),
                  OP_LOGE(context->GetNodeName(), "calculation parameters valid!"),
                  return false);
  OP_CHECK_IF(
      tiling_data_ptr->indices_last_dim == 0,
      OP_LOGE(context->GetNodeName(), "tiling_data_ptr->indices_last_dim = 0 is not support"),
      return true);
  OP_CHECK_IF(
      tiling_data_ptr->indices_last_dim == 0,
      OP_LOGE(context->GetNodeName(), "tiling_data_ptr->indices_last_dim = 0 is not support"),
      return true);
  int64_t half_ub_indices_num = half_ub_size / cal_params.indices_size;
  OP_CHECK_IF(half_ub_indices_num == 0,
                  OP_LOGE(context->GetNodeName(), "half_ub_indices_num = 0 is not support"),
                  return true);
  tiling_data_ptr->adds_loop_num = cal_params.add_data_num / (half_ub_size / cal_params.var_size);
  tiling_data_ptr->adds_last_num = cal_params.add_data_num % (half_ub_size / cal_params.var_size);
  tiling_data_ptr->indices_loop_num = (cal_params.indices_num / tiling_data_ptr->indices_last_dim) /
                                      (half_ub_indices_num / tiling_data_ptr->indices_last_dim);
  tiling_data_ptr->indices_last_num =
      cal_params.indices_num -
      tiling_data_ptr->indices_loop_num *
          (half_ub_indices_num / tiling_data_ptr->indices_last_dim * tiling_data_ptr->indices_last_dim);
  tiling_data_ptr->adds_data_num = cal_params.add_data_num;
  tiling_data_ptr->adds_num = cal_params.adds_num;

  if (cal_params.add_data_num % cal_params.var_data_each_block == 0) {
    if (add_size_byte <= half_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_1;
    } else {
      tiling_data_ptr->tiling_mode = TILING_MODE_2;
    }
  } else {
    if (cal_params.add_data_num < cal_params.var_data_each_block) {
      if (add_size_byte <= half_ub_size) {
        tiling_data_ptr->tiling_mode = TILING_MODE_3;
        tiling_data_ptr->adds_loop_num = cal_params.adds_num / (half_ub_size / cal_params.var_size);
        tiling_data_ptr->adds_last_num = cal_params.adds_num % (half_ub_size / cal_params.var_size);
      } else {
        tiling_data_ptr->tiling_mode = TILING_MODE_4;
      }
    } else {
      tiling_data_ptr->tiling_mode = TILING_MODE_5;
    }
  }
  if (cal_params.add_data_num < cal_params.var_data_each_block) {
    tiling_data_ptr->core_num = 1;
    tiling_data_ptr->indice_step = ceil(float(cal_params.max_indice) / cal_params.core_num);
    tiling_data_ptr->indice_step =
        ceil(float(tiling_data_ptr->indice_step) / cal_params.var_data_each_block) * cal_params.var_data_each_block;
  } else {
    tiling_data_ptr->indice_step = ceil(float(cal_params.max_indice) / cal_params.core_num);
    tiling_data_ptr->indice_step =
        ceil(float(tiling_data_ptr->indice_step) / cal_params.var_data_each_block) * cal_params.var_data_each_block;
    tiling_data_ptr->core_num = ceil(float(cal_params.max_indice) / tiling_data_ptr->indice_step);
  }

  return true;
}

static bool CalScatterNdHighPerfBranchParams(const gert::TilingContext* context,
                                             ScatterNdAddTilingParamsHP* tiling_data_ptr, const CalParams& cal_params,
                                             int64_t need_cast) {
  const int64_t UB_NUM = 2;
  const int64_t UB_NEEDCAST_NUM = 4;

  int64_t alloc_indice_ubsize = cal_params.ub_size / UB_NUM;
  if (need_cast == 1) {
    alloc_indice_ubsize = cal_params.ub_size / UB_NEEDCAST_NUM;
  }

  OP_CHECK_IF(alloc_indice_ubsize == 0,
                  OP_LOGE(context->GetNodeName(), "alloc_indice_ubsize = 0 is not support"),
                  return false);
  OP_CHECK_IF(!CheckCalParams(context, cal_params),
                  OP_LOGE(context->GetNodeName(), "calculation parameters valid!"),
                  return false);

  int64_t alloc_ub_indicesnum = 1;
  if (cal_params.add_data_num < UPDATE_ROW_THRESHOLD) {
    tiling_data_ptr->tiling_mode = TILING_MODE_7;
    int64_t update_row_align = (cal_params.add_data_num + cal_params.var_data_each_block - 1) /
                               cal_params.var_data_each_block * cal_params.var_data_each_block;
    alloc_ub_indicesnum = alloc_indice_ubsize / cal_params.indices_size / update_row_align /
                          tiling_data_ptr->indices_last_dim * tiling_data_ptr->indices_last_dim;
  } else {
    tiling_data_ptr->tiling_mode = TILING_MODE_6;
    alloc_ub_indicesnum = alloc_indice_ubsize / cal_params.indices_size / tiling_data_ptr->indices_last_dim *
                          tiling_data_ptr->indices_last_dim;
  }
  tiling_data_ptr->updates_data_num = cal_params.add_data_num;
  tiling_data_ptr->indices_each_core_data = ceil(float(cal_params.indices_num) / cal_params.core_num);
  tiling_data_ptr->indices_each_core_data =
      (tiling_data_ptr->indices_each_core_data + tiling_data_ptr->indices_last_dim - 1) /
      tiling_data_ptr->indices_last_dim * tiling_data_ptr->indices_last_dim;
  tiling_data_ptr->core_num = ceil(float(cal_params.indices_num) / tiling_data_ptr->indices_each_core_data);
  tiling_data_ptr->indices_last_core_data =
      cal_params.indices_num - tiling_data_ptr->indices_each_core_data * (tiling_data_ptr->core_num - 1);
  tiling_data_ptr->each_core_indices_loop_num = tiling_data_ptr->indices_each_core_data / alloc_ub_indicesnum;
  tiling_data_ptr->each_core_indices_last_num = tiling_data_ptr->indices_each_core_data % alloc_ub_indicesnum;
  tiling_data_ptr->last_core_indices_loop_num = tiling_data_ptr->indices_last_core_data / alloc_ub_indicesnum;
  tiling_data_ptr->last_core_indices_last_num = tiling_data_ptr->indices_last_core_data % alloc_ub_indicesnum;
  tiling_data_ptr->updates_loop_num = cal_params.add_data_num / (alloc_indice_ubsize / cal_params.var_size);
  tiling_data_ptr->updates_last_num = cal_params.add_data_num % (alloc_indice_ubsize / cal_params.var_size);

  return true;
}

static bool TilingCalc4UnAtomicAdd(gert::TilingContext* context, ScatterNdAddTilingParams* tiling_data_ptr,
                                   CalParams& cal_params, const CalcShapeInfo& calc_shape_info) {
  tiling_data_ptr->indices_last_dim = cal_params.indices_last_dim;
  if (cal_params.indices_last_dim > 0) {
    tiling_data_ptr->indices_front_dim = cal_params.indices_num / cal_params.indices_last_dim;
  } else {
    tiling_data_ptr->indices_front_dim = 1;
    for (int64_t i = calc_shape_info.indices_shape.GetDimNum() - 1; i >= 0; i--) {
      tiling_data_ptr->indices_front_dim *= calc_shape_info.indices_shape.GetDim(i);
    }
  }

  OP_CHECK_IF(!CalRunningParams(context, tiling_data_ptr, cal_params),
                  OP_LOGE(context->GetNodeName(), "calculation running parameters failed!"),
                  return false);

  for (int64_t i = 0; i < VAR_OFFSET_NUM; i++) {
    tiling_data_ptr->var_offset[i] = 0;
  }

  for (int64_t i = 0; i < cal_params.indices_last_dim; i++) {
    tiling_data_ptr->var_offset[i] = 1;
    for (int64_t j = i + 1; j < cal_params.var_dim_num; j++)
      tiling_data_ptr->var_offset[i] *= calc_shape_info.var_shape.GetDim(j);
  }

  context->SetBlockDim(tiling_data_ptr->core_num);

  return true;
}

static bool TilingCalc4AtomicAdd(gert::TilingContext* context, ScatterNdAddTilingParamsHP* tiling_data_ptr,
                                 CalParams& cal_params, const CalcShapeInfo& calc_shape_info) {
  tiling_data_ptr->indices_last_dim = cal_params.indices_last_dim;
  CalScatterNdHighPerfBranchParams(context, tiling_data_ptr, cal_params, 0);
  OP_CHECK_IF(!CalScatterNdHighPerfBranchParams(context, tiling_data_ptr, cal_params, 0),
                  OP_LOGE(context->GetNodeName(),
                                                  "calculate scatter nd high performance branch parameters failed!"),
                  return false);

  for (int64_t i = 0; i < cal_params.indices_last_dim; i++) {
    tiling_data_ptr->var_offset[i] = 1;
    for (int64_t j = i + 1; j < cal_params.var_dim_num; j++)
      tiling_data_ptr->var_offset[i] *= calc_shape_info.var_shape.GetDim(j);
  }

  context->SetBlockDim(tiling_data_ptr->core_num);

  return true;
}

static bool PrepareCalcData(const gert::TilingContext* context, CalParams& cal_params,
                            const ScatterCompileInfo* compile_info, const CalcShapeInfo& calc_shape_info) {
  cal_params.core_num = compile_info->core_num;
  cal_params.ub_size = compile_info->ub_size;
  cal_params.var_size = compile_info->var_size;
  cal_params.indices_size = compile_info->indices_size;
  // scatter_nd_add need key atomicflag in compile info
  cal_params.atomic_flag = compile_info->support_atomic;
  OP_CHECK_IF(compile_info->core_num <= 0 || compile_info->ub_size <= 0 || compile_info->var_size <= 0 ||
                      compile_info->indices_size <= 0,
                  OP_LOGE(
                      context->GetNodeName(),
                      "core_num, ub_size, var_size, indices_size must be greater to 0, but got %ld, %ld, %ld, %ld",
                      cal_params.core_num, cal_params.ub_size, cal_params.var_size, cal_params.indices_size),
                  return false);

  cal_params.indices_num = calc_shape_info.indices_shape.GetShapeSize();
  cal_params.adds_num = calc_shape_info.adds_shape.GetShapeSize();
  cal_params.indices_last_dim =
      (calc_shape_info.indices_shape.GetDimNum() > 0)
          ? calc_shape_info.indices_shape.GetDim(calc_shape_info.indices_shape.GetDimNum() - 1)
          : 0;
  cal_params.max_indice = calc_shape_info.var_shape.GetShapeSize();
  cal_params.add_data_num = 1;
  cal_params.var_dim_num = calc_shape_info.var_shape.GetDimNum();
  if (cal_params.var_dim_num > 1) {
    for (int64_t i = cal_params.indices_last_dim; i < cal_params.var_dim_num; i++) {
      cal_params.add_data_num *= calc_shape_info.var_shape.GetDim(i);
    }
  }

  cal_params.var_data_each_block = BLOCK_SIZE / cal_params.var_size;
  OP_LOGD(context->GetNodeName(), "BLOCK_SIZE=%ld.", BLOCK_SIZE);
  OP_LOGD(context->GetNodeName(), "var_size=%ld.", cal_params.var_size);

  OP_LOGD(context->GetNodeName(), "indices_num=%ld.", cal_params.indices_num);
  OP_LOGD(context->GetNodeName(), "adds_num=%ld.", cal_params.adds_num);
  OP_LOGD(context->GetNodeName(), "add_data_num=%ld.", cal_params.add_data_num);
  OP_LOGD(context->GetNodeName(), "max_indice=%ld.", cal_params.max_indice);

  return true;
}

static ge::graphStatus Tiling4ScatterNdAdd(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "ScatterAddTiling running begin");
  auto compile_info = reinterpret_cast<const ScatterCompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  OP_LOGD(context->GetNodeName(), "Tiling4ScatterNdAdd running Simt tiling.");
  ScatterNdAddSimtTiling tilingObj(context);
  tilingObj.coreNum_ = compile_info->core_num;
  tilingObj.ubSize_ = compile_info->ub_size;
  return tilingObj.DoTiling();
}

// register tiling interface of the ScatterNdSubTiling op.
IMPL_OP_OPTILING(ScatterNdAdd).Tiling(Tiling4ScatterNdAdd).TilingParse<ScatterCompileInfo>(TilingPrepare4Scatter);
IMPL_OP_OPTILING(ScatterNdSub).Tiling(Tiling4ScatterNdAdd).TilingParse<ScatterCompileInfo>(TilingPrepare4Scatter);
}  // namespace optiling
