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
 * \file scatter_nd_tiling.cpp
 * \brief
 */
#include <sstream>
#include <cctype>
#include "scatter_nd_tiling_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
static constexpr size_t SC_IN_INDICES_IDX = 0;
static constexpr size_t SC_IN_X_IDX = 1;
static constexpr size_t SC_IN_SHAPE_IDX = 2;
static constexpr size_t SC_OUT_Y_IDX = 0;

static const int64_t OUT_SPECIAL_DIM_0 = 640000;
static const int64_t OUT_SPECIAL_DIM_1 = 1;
static const int64_t OUT_SPECIAL_DIM_2 = 80;
static const int64_t OUT_SPECIAL_DIM_3 = 300000;
static const int64_t OUT_SPECIAL_DIM_4 = 256;
static const int64_t OUT_SPECIAL_DIM_5 = 279424;

// 32b aligned, ub can store all updates, not atomic
static const int64_t TILING_MODE_8 = 8;
// 32b aligned, ub can't store all var and updates, not atomic
static const int64_t TILING_MODE_9 = 9;
// update_data_num is less than 1 block, ub can store all var and updates, not atomic
static const int64_t TILING_MODE_10 = 10;
// update_data_num is less than 1 block, ub can store all var, not atomic
static const int64_t TILING_MODE_11 = 11;
// update_data_num is less than 1 block, ub can store all updates, not atomic
static const int64_t TILING_MODE_12 = 12;
// update_data_num is less than 1 block, ub can't store all var and updates, not atomic
static const int64_t TILING_MODE_13 = 13;
// update_data_num is more than 1 block, not atomic, and less than updateubnum
static const int64_t TILING_MODE_14 = 14;
// update_data_num is more than 1 block, not atomic, and more than updateubnum
static const int64_t TILING_MODE_15 = 15;
// high perf branch, update_data_num is less than 1 block
static const int64_t TILING_MODE_16 = 16;
// high perf branch, update_data_num is 32b aligned
static const int64_t TILING_MODE_17 = 17;

struct CalParams {
  int64_t indices_back;

  int64_t core_num;
  int64_t ub_size;
  int64_t update_size;
  int64_t indices_size;
  int64_t support_atomic;
  int64_t need_cast;

  int64_t var_num;
  int64_t indices_num;
  int64_t updates_num;
  int64_t update_data_num;
  int64_t max_indice;
  int64_t updates_data_each_block;
};

static bool CheckScatterNdTensorShape(const gert::TilingContext* context, const CalcShapeInfo& calc_shape_info,
                                      const int64_t indices_last_dim) {
  const int64_t indices_dims = calc_shape_info.indices_shape.GetDimNum();
  const int64_t updates_dims = calc_shape_info.var_shape.GetDimNum();
  const int64_t output_dims = calc_shape_info.out_shape.GetDimNum();

  OP_CHECK_IF(indices_dims <= 1,
                  OP_LOGE(
                      context->GetNodeName(), "the ndim of indices is less than 1 or equal to 1, indices_dims = %ld.",
                      indices_dims),
                  return false);

  OP_CHECK_IF(
      indices_dims - 1 + output_dims - indices_last_dim != updates_dims,
      OP_LOGE(context->GetNodeName(),
                                      "output's shape and updates'shape are not equal in some dimensions, indices_dims "
                                      "- 1 + output_dims - indices_last_dim = %ld, updates_dims = %ld.",
                                      indices_dims - 1 + output_dims - indices_last_dim, updates_dims),
      return false);

  for (int64_t i = 0; i < indices_dims - 1; i++) {
    OP_CHECK_IF(calc_shape_info.indices_shape.GetDim(i) != calc_shape_info.var_shape.GetDim(i),
                    OP_LOGE(
                        context->GetNodeName(),
                        "indices's shape and updates'shape are not equal in some dimensions, "
                        "calc_shape_info.indices_shape.GetDim(i) = %ld, calc_shape_info.var_shape.GetDim(i) = %ld.",
                        calc_shape_info.indices_shape.GetDim(i), calc_shape_info.var_shape.GetDim(i)),
                    return false);
  }

  for (int64_t i = 0; i < updates_dims - indices_dims + 1; i++) {
    OP_CHECK_IF(
        calc_shape_info.var_shape.GetDim(indices_dims - 1 + i) != calc_shape_info.out_shape[indices_last_dim + i],
        OP_LOGE(context->GetNodeName(),
                                        "output's shape and updates'shape are not equal in some dimensions, "
                                        "calc_shape_info.var_shape.GetDim(indices_dims - 1 + i) = %ld, "
                                        "calc_shape_info.out_shape[indices_last_dim + i] = %ld.",
                                        calc_shape_info.var_shape.GetDim(indices_dims - 1 + i),
                                        calc_shape_info.out_shape[indices_last_dim + i]),
        return false);
  }
  return true;
}

static void SetCalParamsHelper1(CalParams& cal_params, const ScatterNdCompileInfo* compile_info) {
  cal_params.core_num = compile_info->core_num;
  cal_params.ub_size = compile_info->ub_size;
  cal_params.update_size = compile_info->updates_size;
  cal_params.indices_size = compile_info->indices_size;
  cal_params.support_atomic = compile_info->support_atomic;
  cal_params.need_cast = compile_info->need_cast;
}

static void SetCalParamsHelper2(CalParams& cal_params, const CalcShapeInfo& calc_shape_info) {
  cal_params.var_num = calc_shape_info.out_shape.GetShapeSize();
  cal_params.indices_num = calc_shape_info.indices_shape.GetShapeSize();
  cal_params.updates_num = calc_shape_info.var_shape.GetShapeSize();
  cal_params.update_data_num = 1;
  int64_t out_dim_num = calc_shape_info.out_shape.GetDimNum();
  if (out_dim_num != cal_params.indices_back) {
    for (int64_t i = cal_params.indices_back; i < out_dim_num; i++) {
      cal_params.update_data_num *= calc_shape_info.out_shape.GetDim(i);
    }
  }

  cal_params.max_indice = 1;
  for (int64_t i = 0; i < cal_params.indices_back; i++) {
    cal_params.max_indice *= calc_shape_info.out_shape.GetDim(i);
  }
}

static bool PrepareCalcData(const gert::TilingContext* context, ScatterNdTilingParams* tiling_data_ptr,
                            CalParams& cal_params, const ScatterNdCompileInfo* compile_info,
                            const CalcShapeInfo& calc_shape_info) {
  SetCalParamsHelper1(cal_params, compile_info);

  OP_CHECK_IF(cal_params.core_num <= ZERO || cal_params.ub_size <= ZERO || cal_params.update_size <= ZERO ||
                      cal_params.indices_size <= ZERO,
                  OP_LOGE(
                      context->GetNodeName(),
                      "core_num, ub_size, update_size, indices_size must be greater to 0, but got %ld, %ld, %ld, %ld",
                      cal_params.core_num, cal_params.ub_size, cal_params.update_size, cal_params.indices_size),
                  return false);

  SetCalParamsHelper2(cal_params, calc_shape_info);

  tiling_data_ptr->max_indice = cal_params.max_indice;
  tiling_data_ptr->indices_last_dim = cal_params.indices_back;
  OP_CHECK_IF(cal_params.update_size == 0,
                  OP_LOGE(context->GetNodeName(), "PrepareCalcData, update_size cannot be 0."),
                  return false);
  cal_params.updates_data_each_block = BLOCK_SIZE / cal_params.update_size;

  for (int64_t i = 0; i < cal_params.indices_back; i++) {
    tiling_data_ptr->var_offset[i] = 1;
    for (int64_t j = i + 1; j < cal_params.indices_back; j++) {
      tiling_data_ptr->var_offset[i] *= calc_shape_info.out_shape.GetDim(j);
    }
  }

  OP_LOGD(context->GetNodeName(), "op [Tiling4ScatterNd] : var_num=%ld.", cal_params.var_num);
  OP_LOGD(context->GetNodeName(), "op [Tiling4ScatterNd] : indices_num=%ld.", cal_params.indices_num);
  OP_LOGD(context->GetNodeName(), "op [Tiling4ScatterNd] : updates_num=%ld.", cal_params.updates_num);

  if (cal_params.update_data_num < cal_params.updates_data_each_block) {
    tiling_data_ptr->core_num = 1;
  } else {
    OP_CHECK_IF(
        cal_params.core_num == 0,
        OP_LOGE(context->GetNodeName(), "PrepareCalcData, core_num = 0 is not supported."),
        return false);
    tiling_data_ptr->indice_step = ceil(float(cal_params.max_indice) / cal_params.core_num);
    OP_CHECK_IF(
        tiling_data_ptr->indice_step == 0,
        OP_LOGE(context->GetNodeName(), "PrepareCalcData, indices_step = 0 is not supported."),
        return false);
    tiling_data_ptr->core_num = ceil(float(cal_params.max_indice) / tiling_data_ptr->indice_step);
  }

  return true;
}

static bool CalScatterNdHighPerfBranchParams(const gert::TilingContext* context,
                                             ScatterNdTilingParams* tiling_data_ptr,
                                             const CalParams& cal_params) {
  const int64_t UB_NUM = 2;
  const int64_t UB_NEEDCAST_NUM = 4;

  int64_t alloc_indice_ubsize = cal_params.ub_size / UB_NUM;
  if (cal_params.need_cast == 1) {
    alloc_indice_ubsize = cal_params.ub_size / UB_NEEDCAST_NUM;
  }
  OP_CHECK_IF(alloc_indice_ubsize == 0,
                  OP_LOGE(context->GetNodeName(), "alloc_indice_ubsize = 0 is not supported"),
                  return false);
  OP_CHECK_IF(cal_params.indices_size == 0,
                  OP_LOGE(context->GetNodeName(), "indices_size = 0 is not supported"),
                  return false);
  OP_CHECK_IF(cal_params.core_num == 0,
                  OP_LOGE(context->GetNodeName(), "core_num = 0 is not supported"),
                  return false);
  OP_CHECK_IF(
      cal_params.updates_data_each_block == 0,
      OP_LOGE(context->GetNodeName(), "updates_data_each_block = 0 is not supported"),
      return false);
  OP_CHECK_IF(cal_params.update_size == 0,
                  OP_LOGE(context->GetNodeName(), "update_size = 0 is not supported"),
                  return false);
  int64_t alloc_ub_indicesnum = alloc_indice_ubsize / cal_params.indices_size / tiling_data_ptr->indices_last_dim *
                                tiling_data_ptr->indices_last_dim;
  tiling_data_ptr->tiling_mode = TILING_MODE_16;
  tiling_data_ptr->updates_data_num = cal_params.update_data_num;
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
  tiling_data_ptr->updates_loop_num = cal_params.update_data_num / (alloc_indice_ubsize / cal_params.update_size);
  tiling_data_ptr->updates_last_num = cal_params.update_data_num % (alloc_indice_ubsize / cal_params.update_size);

  return true;
}

static bool CalNotAtomicBranchRunningParams(const gert::TilingContext* context,
                                            ScatterNdTilingParams* tiling_data_ptr,
                                            const CalParams& cal_params) {
  int64_t var_all_size_byte = cal_params.update_size * cal_params.var_num;
  int64_t var_size_byte = cal_params.update_size * tiling_data_ptr->indice_step * cal_params.update_data_num;
  int64_t update_size_byte = cal_params.update_size * cal_params.updates_num;
  int64_t var_ub_size = cal_params.ub_size / 8 * 3;
  int64_t indices_ub_size = cal_params.ub_size / 8 * 2;
  OP_CHECK_IF(var_ub_size == 0,
                  OP_LOGE(context->GetNodeName(), "var_ub_size = 0 is not supported"),
                  return false);
  OP_CHECK_IF(cal_params.indices_size == 0,
                  OP_LOGE(context->GetNodeName(), "indices_size = 0 is not supported"),
                  return false);
  OP_CHECK_IF(cal_params.update_size == 0,
                  OP_LOGE("scatter_nd", "update_size = 0 is not supported"), return false);
  OP_CHECK_IF(
      tiling_data_ptr->indices_last_dim == 0,
      OP_LOGE(context->GetNodeName(), "tiling_data_ptr->indices_last_dim = 0 is not supported"),
      return false);
  OP_CHECK_IF(
      tiling_data_ptr->indices_last_dim == 0,
      OP_LOGE(context->GetNodeName(), "tiling_data_ptr->indices_last_dim = 0 is not supported"),
      return false);
  OP_CHECK_IF(
      cal_params.updates_data_each_block == 0,
      OP_LOGE(context->GetNodeName(), "updates_data_each_block = 0 is not supported"),
      return false);
  tiling_data_ptr->var_loop_num = cal_params.var_num / (var_ub_size / cal_params.update_size);
  tiling_data_ptr->var_last_num = cal_params.var_num % (var_ub_size / cal_params.update_size);
  tiling_data_ptr->updates_loop_num = cal_params.update_data_num / (var_ub_size / cal_params.update_size);
  tiling_data_ptr->updates_last_num = cal_params.update_data_num % (var_ub_size / cal_params.update_size);
  tiling_data_ptr->indices_loop_num =
      cal_params.indices_num / (indices_ub_size / cal_params.indices_size / tiling_data_ptr->indices_last_dim *
                                tiling_data_ptr->indices_last_dim);
  tiling_data_ptr->indices_last_num =
      cal_params.indices_num % (indices_ub_size / cal_params.indices_size / tiling_data_ptr->indices_last_dim *
                                tiling_data_ptr->indices_last_dim);
  tiling_data_ptr->updates_data_num = cal_params.update_data_num;
  tiling_data_ptr->updates_num = cal_params.updates_num;
  tiling_data_ptr->var_num = cal_params.var_num;
  if (cal_params.update_data_num % cal_params.updates_data_each_block == 0) {
    if (update_size_byte <= var_ub_size && var_size_byte <= var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_6;
    } else if (update_size_byte > var_ub_size && var_size_byte <= var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_7;
    } else if (update_size_byte <= var_ub_size && var_size_byte > var_ub_size) {
      tiling_data_ptr->tiling_mode = TILING_MODE_8;
    } else {
      tiling_data_ptr->tiling_mode = TILING_MODE_9;
    }
  } else if (cal_params.update_data_num < cal_params.updates_data_each_block) {
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
    if (cal_params.update_data_num / (var_ub_size / cal_params.update_size) == 0) {
      tiling_data_ptr->tiling_mode = TILING_MODE_14;
    } else {
      tiling_data_ptr->tiling_mode = TILING_MODE_15;
    }
  }

  tiling_data_ptr->var_each_core_data = tiling_data_ptr->indice_step * tiling_data_ptr->updates_data_num;
  int64_t varLastCoreData = cal_params.var_num - tiling_data_ptr->var_each_core_data * (cal_params.core_num - 1);
  tiling_data_ptr->var_each_core_burst_len = tiling_data_ptr->var_each_core_data / cal_params.updates_data_each_block;
  tiling_data_ptr->var_last_core_burst_len = varLastCoreData / cal_params.updates_data_each_block;

  tiling_data_ptr->var_each_core_set_zero_loop_num =
      tiling_data_ptr->var_each_core_data / (var_ub_size / cal_params.update_size);
  tiling_data_ptr->var_each_core_set_zero_last_num =
      tiling_data_ptr->var_each_core_data % (var_ub_size / cal_params.update_size);
  tiling_data_ptr->var_last_core_set_zero_loop_num = varLastCoreData / (var_ub_size / cal_params.update_size);
  tiling_data_ptr->var_last_core_set_zero_last_num = varLastCoreData % (var_ub_size / cal_params.update_size);

  if (tiling_data_ptr->tiling_mode == TILING_MODE_9 || tiling_data_ptr->tiling_mode == TILING_MODE_14 ||
      tiling_data_ptr->tiling_mode == TILING_MODE_15) {
    tiling_data_ptr->var_loop_num = cal_params.update_data_num / (var_ub_size / cal_params.update_size);
    tiling_data_ptr->var_last_num = cal_params.update_data_num % (var_ub_size / cal_params.update_size);
  }

  return true;
}

static ge::graphStatus TilingPrepare4ScatterNd(gert::TilingParseContext* context) {
  auto compile_info = context->GetCompiledInfo<ScatterNdCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4ScatterNd GRAPH_SUCESS.");
  return ge::GRAPH_SUCCESS;
  
}

static bool TIKTiling4ScatterNd(gert::TilingContext* context, const ScatterNdCompileInfo* compile_info) {
  OP_LOGD(context->GetNodeName(), "TIKTiling4ScatterNd running begin");
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  ScatterNdTilingParams* tiling_data_ptr = context->GetTilingData<ScatterNdTilingParams>();
  OP_CHECK_NULL_WITH_CONTEXT(context, tiling_data_ptr);
  tiling_data_ptr->Init();
  tiling_data_ptr->tiling_core_num = compile_info->core_num;
  OP_LOGD(context->GetNodeName(), "set tiling core num: %ld", tiling_data_ptr->tiling_core_num);
  CalcShapeInfo calc_shape_info;

  const gert::StorageShape* indices_storage_shape = context->GetInputShape(SC_IN_INDICES_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, indices_storage_shape);
  calc_shape_info.indices_shape = Ops::Base::EnsureNotScalar(indices_storage_shape->GetStorageShape());

  const gert::StorageShape* x_storage_shape = context->GetInputShape(SC_IN_X_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_storage_shape);
  calc_shape_info.var_shape = Ops::Base::EnsureNotScalar(x_storage_shape->GetStorageShape());

  const gert::StorageShape* y_storage_shape = context->GetOutputShape(SC_OUT_Y_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_storage_shape);
  calc_shape_info.out_shape = Ops::Base::EnsureNotScalar(y_storage_shape->GetStorageShape());

  CalParams cal_params;

  cal_params.indices_back = calc_shape_info.indices_shape.GetDim(calc_shape_info.indices_shape.GetDimNum() - 1);
  OP_CHECK_IF(!CheckScatterNdTensorShape(context, calc_shape_info, cal_params.indices_back),
                  OP_LOGE(context->GetNodeName(), "CheckScatterNdTensorShape is failed!"),
                  return false);

  OP_CHECK_IF(!PrepareCalcData(context, tiling_data_ptr, cal_params, compile_info, calc_shape_info),
                  OP_LOGE(context->GetNodeName(), "PrepareCalcData is failed!"),
                  return false);
  if (cal_params.support_atomic == 1) {
    OP_CHECK_IF(
        !CalScatterNdHighPerfBranchParams(context, tiling_data_ptr, cal_params),
        OP_LOGE(context->GetNodeName(), "CalScatterNdHighPerfBranchParams is failed!"),
        return false);
  } else {
    OP_CHECK_IF(
        !CalNotAtomicBranchRunningParams(context, tiling_data_ptr, cal_params),
        OP_LOGE(context->GetNodeName(), "CalNotAtomicBranchRunningParams is failed!"),
        return false);
  }

  context->SetBlockDim(tiling_data_ptr->core_num);
  
  OP_LOGD(context->GetNodeName(), "set block dim: %u", context->GetBlockDim());
  OP_LOGD(context->GetNodeName(), "TIKTiling4ScatterNd run success");

  return true;
}


static ge::graphStatus Tiling4ScatterNd(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "ScatterNdTiling running begin");
  auto compile_info = reinterpret_cast<const ScatterNdCompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  OP_LOGD(context->GetNodeName(), "ScatterNdTiling is ascendc. runing Smit tiling.");
  return TilingScatterNd(context);
}

// register tiling interface of the Tiling4ScatterNd op.
IMPL_OP_OPTILING(ScatterNd).Tiling(Tiling4ScatterNd).TilingParse<ScatterNdCompileInfo>(TilingPrepare4ScatterNd);
}  // namespace optiling
