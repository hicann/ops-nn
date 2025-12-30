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
 * \file scatter_elements_tiling.cpp
 * \brief scatter elements tiling
 */
#include <map>
#include "scatter_elements_tiling.h"
#include "scatter_elements_tiling_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace {
constexpr size_t INDEX_DATA = 0;
constexpr size_t INDEX_INDICES = 1;
constexpr size_t INDEX_UPDATES = 2;
constexpr size_t UB_DATA_RATIO = 4;
constexpr size_t UB_INDICES_AND_UPDATES_RATIO = 2;
constexpr size_t UB_DATA_RATIO_LAST_AXIS = 2;
constexpr size_t INDEX_AXIS = 0;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr size_t DIM_FOUR = 4;
constexpr size_t DIM_FIVE = 5;
constexpr size_t DIM_SIX = 6;
constexpr size_t DIM_SEVEN = 7;
constexpr size_t BYTES_PER_BLOCK = 32;
const int32_t BLOCK_8 = 32;
const int32_t BLOCK_16 = 16;
const int32_t BLOCK_32 = 8;
const int32_t BLOCK_64 = 4;
const int32_t RESERVED_BYTES = 1024;
// Totally 32 core, 256kb per core. According to the current scheme, data uses 1/4 storage of UB, so each core can hold
// 256 / 4 * 1024 * 32 / 4 = 2097152 elements in data
const int32_t DATA_ONR_ROUND = 2097152;
const int32_t MAX_INDICES_NUM_MERGE_AXIS = 30000;
const int32_t MAX_INDICES_NUM_NON_MERGE_AXIS = 5500;
constexpr size_t TILING_MODE_SAMESHAPE = 0;
constexpr size_t TILING_MODE_DIFFSHAPE = 1;
constexpr size_t TILING_MODE_LASTAXIS_SAME_SHAPE = 2;
constexpr size_t TILING_MODE_LASTAXIS_DIFF_SHAPE = 3;
const int32_t MIN_DIM_FOR_SAMESHAPE_LASTAXIS = 1;
const int32_t MAX_DATA_WIDTH_FOR_OPTI = 64;
const int32_t REPEAT_FOR_OPTI = 8;
}  // namespace

namespace optiling {
static std::map<const ge::DataType, const int32_t> DATA_BLOCK_LEN = {
    {ge::DT_INT8, BLOCK_8}, {ge::DT_UINT8, BLOCK_8}, {ge::DT_FLOAT16, BLOCK_16},
    {ge::DT_FLOAT, BLOCK_32}, {ge::DT_INT32, BLOCK_32}, {ge::DT_BF16, BLOCK_16}};

static std::map<const ge::DataType, const int32_t> INDICES_BLOCK_LEN = {
    {ge::DT_INT32, BLOCK_32}, {ge::DT_INT64, BLOCK_64}};

static void PrintTilingParams(const gert::TilingContext* context, const ScatterElementsTilingParams* param) {
  OP_LOGD(context->GetNodeName(), "dims_indices=%d.", param->dims_indices);
  OP_LOGD(context->GetNodeName(), "dims_data=%d.", param->dims_data);
  OP_LOGD(context->GetNodeName(), "shape_acc_data=%d.", param->shape_acc_data);
  OP_LOGD(context->GetNodeName(), "shape_acc_indices=%d.", param->shape_acc_indices);
  OP_LOGD(context->GetNodeName(), "shape_indices_0=%d.", param->shape_indices_0);
  OP_LOGD(context->GetNodeName(), "shape_indices_1=%d.", param->shape_indices_1);
  OP_LOGD(context->GetNodeName(), "shape_indices_2=%d.", param->shape_indices_2);
  OP_LOGD(context->GetNodeName(), "shape_indices_3=%d.", param->shape_indices_3);
  OP_LOGD(context->GetNodeName(), "shape_indices_4=%d.", param->shape_indices_4);
  OP_LOGD(context->GetNodeName(), "shape_indices_5=%d.", param->shape_indices_5);
  OP_LOGD(context->GetNodeName(), "shape_indices_6=%d.", param->shape_indices_6);
  OP_LOGD(context->GetNodeName(), "shape_indices_7=%d.", param->shape_indices_7);
  OP_LOGD(context->GetNodeName(), "shape_data_0 = %d.", param->shape_data_0);
  OP_LOGD(context->GetNodeName(), "shape_data_1 = %d.", param->shape_data_1);
  OP_LOGD(context->GetNodeName(), "shape_data_2 = %d.", param->shape_data_2);
  OP_LOGD(context->GetNodeName(), "shape_data_3 = %d.", param->shape_data_3);
  OP_LOGD(context->GetNodeName(), "shape_data_4 = %d.", param->shape_data_4);
  OP_LOGD(context->GetNodeName(), "shape_data_5 = %d.", param->shape_data_5);
  OP_LOGD(context->GetNodeName(), "shape_data_6 = %d.", param->shape_data_6);
  OP_LOGD(context->GetNodeName(), "shape_data_7 = %d.", param->shape_data_7);
  OP_LOGD(context->GetNodeName(), "data_block = %d.", param->data_block);
  OP_LOGD(context->GetNodeName(), "repeat = %d.", param->repeat);
  OP_LOGD(context->GetNodeName(), "used_aicore_num = %d.", param->used_aicore_num);
  OP_LOGD(context->GetNodeName(), "batch_num_per_aicore = %d.", param->batch_num_per_aicore);
  OP_LOGD(context->GetNodeName(), "batch_tail = %d.", param->batch_tail);
  OP_LOGD(context->GetNodeName(), "indices_block = %d.", param->indices_block);
  OP_LOGD(context->GetNodeName(), "indices_repeat = %d.", param->indices_repeat);
  OP_LOGD(context->GetNodeName(), "rounds_indices = %d.", param->rounds_indices);
  OP_LOGD(context->GetNodeName(), "tail_indices = %d.", param->tail_indices);
  OP_LOGD(context->GetNodeName(), "tail_indices_block = %d.", param->tail_indices_block);
  OP_LOGD(context->GetNodeName(), "tail_indices_repeat = %d.", param->tail_indices_repeat);
  OP_LOGD(context->GetNodeName(), "updates_block = %d.", param->updates_block);
  OP_LOGD(context->GetNodeName(), "updates_repeat = %d.", param->updates_repeat);
  OP_LOGD(context->GetNodeName(), "tail_updates_block = %d.", param->tail_updates_repeat);
  OP_LOGD(context->GetNodeName(), "tail_updates_repeat = %d.", param->tail_updates_repeat);
  OP_LOGD(context->GetNodeName(), "tiling_mode = %d.", param->tiling_mode);
  OP_LOGD(context->GetNodeName(), "params_row_data = %d.", param->params_row_data);
  OP_LOGD(context->GetNodeName(), "params_row_indices = %d.", param->params_row_indices);
  OP_LOGD(context->GetNodeName(), "params_axis_data = %d.", param->params_axis_data);
  OP_LOGD(context->GetNodeName(), "params_axis_indices = %d.", param->params_axis_indices);
  OP_LOGD(context->GetNodeName(), "axis = %d.", param->axis);
  OP_LOGD(context->GetNodeName(), "rounds = %d.", param->rounds);
  OP_LOGD(context->GetNodeName(), "data_tail_repeat = %d.", param->data_tail_repeat);
  OP_LOGD(context->GetNodeName(), "repeat_per_core = %d.", param->repeat_per_core);
  OP_LOGD(context->GetNodeName(), "rounds_tail = %d.", param->rounds_tail);
}

ge::graphStatus CheckInputList(gert::TilingContext* context, ScatterElementsTilingParams* param,
                               const gert::Shape& data_origin_shape, const gert::Shape& indices_origin_shape) {
  // dims_data = dims_indices
  if (param->dims_data != param->dims_indices) {
    OP_LOGE(context->GetNodeName(), "dims_data(%d) != dims_indices(%d)", param->dims_data,
                                    param->dims_indices);
    return ge::GRAPH_FAILED;
  }

  // dims_data = dims_updates
  auto updates_shape = context->GetInputShape(INDEX_UPDATES);
  OP_CHECK_NULL_WITH_CONTEXT(context, updates_shape);
  const auto& updates_origin_shape = Ops::Base::EnsureNotScalar(updates_shape->GetOriginShape());
  int32_t dims_updates = static_cast<int32_t>(updates_origin_shape.GetDimNum());
  if (param->dims_data != dims_updates) {
    OP_LOGE(context->GetNodeName(), "dims_data(%d) != dims_updates(%d)", param->dims_data,
                                    dims_updates);
    return ge::GRAPH_FAILED;
  }

  // data_dtype = updates_dtype
  auto data_desc = context->GetInputDesc(INDEX_DATA);
  ge::DataType data_dtype = data_desc->GetDataType();
  auto updates_desc = context->GetInputDesc(INDEX_UPDATES);
  ge::DataType updates_dtype = updates_desc->GetDataType();
  if (data_dtype != updates_dtype) {
    OP_LOGE(context->GetNodeName(), "data_dtype(%s) != updates_dtype(%s)",
                                    ge::TypeUtils::DataTypeToAscendString(data_dtype).GetString(), ge::TypeUtils::DataTypeToAscendString(updates_dtype).GetString());
    return ge::GRAPH_FAILED;
  }

  // data and indices are not empty tensor
  // indices_shape = updates_shape in each dim
  for (int32_t i = 0; i < param->dims_indices; i++) {
    if (data_origin_shape.GetDim(i) == 0) {
      OP_LOGE(context->GetNodeName(), "data_shape[%d](%ld) == 0", i,
                                      data_origin_shape.GetDim(i));
      return ge::GRAPH_FAILED;
    }
    if (indices_origin_shape.GetDim(i) == 0) {
      OP_LOGE(context->GetNodeName(), "indices_shape[%d](%ld) == 0", i,
                                      indices_origin_shape.GetDim(i));
      return ge::GRAPH_FAILED;
    }
    if (indices_origin_shape.GetDim(i) != updates_origin_shape.GetDim(i)) {
      OP_LOGE(context->GetNodeName(), "indices_shape[%d](%ld) != updates_shape[%d](%ld)", i,
                                      indices_origin_shape.GetDim(i), i, updates_origin_shape.GetDim(i));
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckList(gert::TilingContext* context, ScatterElementsTilingParams* param,
                          const gert::Shape& data_origin_shape, const gert::Shape& indices_origin_shape) {
  OP_CHECK_IF(CheckInputList(context, param, data_origin_shape, indices_origin_shape) != ge::GRAPH_SUCCESS,
                  OP_LOGE("scatter_elements tiling", "check input list fail"),
                  return ge::GRAPH_FAILED);

  // axis in [-dims_data, dims_data - 1]
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const auto axis = attrs->GetAttrPointer<int32_t>(INDEX_AXIS);
  if (*axis >= param->dims_data) {
    OP_LOGE(context->GetNodeName(), "axis(%d) >= dims_data(%d)", *axis, param->dims_data);
    return ge::GRAPH_FAILED;
  }
  if (*axis < -1 * param->dims_data) {
    OP_LOGE(context->GetNodeName(), "axis(%d) < -dims_data(%d)", *axis, param->dims_data);
    return ge::GRAPH_FAILED;
  }
  if (*axis < 0) {
    param->axis =  static_cast<int32_t>(*axis + param->dims_data);
  } else {
    param->axis = static_cast<int32_t>(*axis);
  }

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalParam(gert::TilingContext* context, const ScatterElementsCompileInfo* compile_info,
                               ScatterElementsTilingParams* param) {
  auto data_desc = context->GetInputDesc(INDEX_DATA);
  OP_CHECK_NULL_WITH_CONTEXT(context, data_desc);
  const ge::DataType data_dtype = data_desc->GetDataType();
  // data, updates, output have same 'block' (because they have same dtype)
  auto iter = DATA_BLOCK_LEN.find(data_dtype);
  OP_CHECK_IF(
    iter == DATA_BLOCK_LEN.end(),
    OP_LOGE(context->GetNodeName(), "do not support data_dtype(%s)",
                                    ge::TypeUtils::DataTypeToAscendString(data_dtype).GetString()),
    return ge::GRAPH_FAILED);
  const int32_t block = DATA_BLOCK_LEN[data_dtype];

  // numbers of elements in data per core
  int32_t data_block = compile_info->ub_size_bytes / UB_DATA_RATIO / BYTES_PER_BLOCK * block;
  if (data_block > param->shape_acc_data) {
    data_block = param->shape_acc_data;
  }
  data_block = (data_block + block - 1) / block * block;
  OP_CHECK_IF(data_block == 0,
                  OP_LOGE(context->GetNodeName(), "data_block = 0"), return ge::GRAPH_FAILED);
  int32_t repeat = data_block / block;

  int32_t rounds = (param->shape_acc_data + data_block - 1) / data_block;
  int32_t used_aicore_num = rounds;
  if (rounds > compile_info->available_aicore_num) {
    used_aicore_num = compile_info->available_aicore_num;
  }
  int32_t batch_num_per_aicore = rounds / used_aicore_num;
  int32_t batch_tail = rounds % used_aicore_num;

  int32_t data_tail = param->shape_acc_data - data_block * (rounds - 1);
  int32_t data_tail_repeat = (data_tail + block - 1) / block;

  auto indices_desc = context->GetInputDesc(INDEX_INDICES);
  OP_CHECK_NULL_WITH_CONTEXT(context, indices_desc);
  const ge::DataType indices_dtype = indices_desc->GetDataType();
  iter = INDICES_BLOCK_LEN.find(indices_dtype);
  OP_CHECK_IF(iter == INDICES_BLOCK_LEN.end(),
    OP_LOGE(context->GetNodeName(), "do not support indices_dtype(%s)",
                                    ge::TypeUtils::DataTypeToAscendString(indices_dtype).GetString()),
    return ge::GRAPH_FAILED);
  const int32_t block_indices = INDICES_BLOCK_LEN[indices_dtype];

  // numbers of elements in indices per core
  int32_t ub_size_bytes_indices = compile_info->ub_size_bytes / UB_INDICES_AND_UPDATES_RATIO *
                                  block / (block + block_indices);
  int32_t indices_block = ub_size_bytes_indices / BYTES_PER_BLOCK * block_indices;
  OP_CHECK_IF(indices_block == 0, OP_LOGE(context->GetNodeName(), "indices_block = 0"),
                  return ge::GRAPH_FAILED);
  int32_t indices_repeat = indices_block / block_indices;
  int32_t rounds_indices = param->shape_acc_indices / indices_block;
  int32_t tail_indices = param->shape_acc_indices % indices_block;
  int32_t tail_indices_block = (tail_indices + block_indices - 1) / block_indices * block_indices;
  int32_t tail_indices_repeat = tail_indices_block / block_indices;

  // numbers of elements in updates per core ( = numbers of indices before);
  // updates_block might be larger than indices_block, but only indices_block is used in the algorithm to ensure that
  // same number of indices and updates are used in each process.
  int32_t updates_block = (indices_block + block - 1) / block * block;
  int32_t updates_repeat = updates_block / block;
  int32_t tail_updates_block = (tail_indices + block - 1) / block * block;
  int32_t tail_updates_repeat = tail_updates_block / block;

  param->data_block = data_block;
  param->repeat = repeat;
  param->used_aicore_num = used_aicore_num;
  param->batch_num_per_aicore = batch_num_per_aicore;
  param->batch_tail = batch_tail;
  param->indices_block = indices_block;
  param->indices_repeat = indices_repeat;
  param->rounds_indices = rounds_indices;
  param->tail_indices = tail_indices;
  param->tail_indices_block = tail_indices_block;
  param->tail_indices_repeat = tail_indices_repeat;
  param->updates_block = updates_block;
  param->updates_repeat = updates_repeat;
  param->tail_updates_block = tail_updates_block;
  param->tail_updates_repeat = tail_updates_repeat;
  param->rounds = rounds;
  param->data_tail_repeat = data_tail_repeat;

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalParamLastAxis(gert::TilingContext* context, const ScatterElementsCompileInfo* compile_info,
                               ScatterElementsTilingParams* param, const gert::Shape& data_origin_shape,
                               const gert::Shape& indices_origin_shape) {
  int32_t rounds = 1;
  for (int32_t i = 0; i < param->axis; i++) {
    rounds *= data_origin_shape.GetDim(i);
  }

  auto data_desc = context->GetInputDesc(INDEX_DATA);
  OP_CHECK_NULL_WITH_CONTEXT(context, data_desc);
  const ge::DataType data_dtype = data_desc->GetDataType();
  // data, updates, output have same 'block' (because they have same dtype)
  const int32_t block = DATA_BLOCK_LEN[data_dtype]; // block is already checked in func CheckModeLastAxis

  int32_t data_width = data_origin_shape.GetDim(param->axis);
  int32_t data_block = (data_width + block - 1) / block * block;
  int32_t repeat = data_block / block;
  int32_t indices_width = indices_origin_shape.GetDim(param->axis);
  auto indices_desc = context->GetInputDesc(INDEX_INDICES);
  OP_CHECK_NULL_WITH_CONTEXT(context, indices_desc);
  const ge::DataType indices_dtype = indices_desc->GetDataType();
  auto iter = INDICES_BLOCK_LEN.find(indices_dtype);
  OP_CHECK_IF(iter == INDICES_BLOCK_LEN.end(),
    OP_LOGE(context->GetNodeName(), "do not support indices_dtype(%s)",
                                    ge::TypeUtils::DataTypeToAscendString(indices_dtype).GetString()),
    return ge::GRAPH_FAILED);
  const int32_t block_indices = INDICES_BLOCK_LEN[indices_dtype];

  int32_t repeat_per_core = 1;
  int32_t rounds_tail = 0;
  // unaligned and same_shape cases
  if ((data_width != data_block) && (param->tiling_mode == TILING_MODE_LASTAXIS_SAME_SHAPE)) {
    while ((repeat_per_core * data_width) % block != 0) {
      repeat_per_core++;
    }
    if ((data_width <= MAX_DATA_WIDTH_FOR_OPTI) && (indices_width <= block_indices)) {
      repeat_per_core = repeat_per_core * REPEAT_FOR_OPTI;
    }
    rounds_tail = rounds % repeat_per_core;
    rounds = (rounds + repeat_per_core - 1) / repeat_per_core;

    int32_t ub_for_data = compile_info->ub_size_bytes / UB_DATA_RATIO_LAST_AXIS; // ub space for data
    int32_t actual_data_in_ub = repeat_per_core * repeat * BYTES_PER_BLOCK; // actual ub space that data uses
    OP_CHECK_IF(actual_data_in_ub > ub_for_data,
      OP_LOGW(context->GetNodeName(), "actual_data_in_ub(%d) > ub_for_data(%d)", actual_data_in_ub, ub_for_data),
      return ge::GRAPH_FAILED);
  }

  int32_t used_aicore_num = rounds;
  if (rounds > compile_info->available_aicore_num) {
    used_aicore_num = compile_info->available_aicore_num;
  }
  int32_t batch_num_per_aicore = rounds / used_aicore_num;
  int32_t batch_tail = rounds % used_aicore_num;

  int32_t ub_left = compile_info->ub_size_bytes - repeat_per_core * repeat * BYTES_PER_BLOCK - RESERVED_BYTES;
  int32_t ub_for_indices = ub_left / (block + block_indices) * block;
  int32_t elem_for_indices = ub_for_indices / BYTES_PER_BLOCK * block_indices;
  int32_t indices_block = (elem_for_indices + block - 1) / block * block; // use block but not block_indices for updates
  int32_t indices_repeat = indices_block / block_indices;
  int32_t updates_repeat = indices_block / block;
  int32_t rounds_indices = indices_width / indices_block;
  int32_t tail_indices = indices_width % indices_block;
  int32_t tail_indices_repeat = (tail_indices + block_indices - 1) / block_indices;
  int32_t tail_updates_repeat = (tail_indices + block - 1) / block;

  param->rounds = rounds;
  param->used_aicore_num = used_aicore_num;
  param->batch_num_per_aicore = batch_num_per_aicore;
  param->batch_tail = batch_tail;
  param->data_block = data_block;
  param->repeat = repeat;
  param->indices_block = indices_block;
  param->updates_block = indices_block;
  param->indices_repeat = indices_repeat;
  param->updates_repeat = updates_repeat;
  param->rounds_indices = rounds_indices;
  param->tail_indices = tail_indices;
  param->tail_indices_repeat = tail_indices_repeat;
  param->tail_updates_repeat = tail_updates_repeat;
  param->repeat_per_core = repeat_per_core;
  param->rounds_tail = rounds_tail;

  return ge::GRAPH_SUCCESS;
}

static int32_t CheckModeLastAxis(gert::TilingContext* context, const ScatterElementsCompileInfo* compile_info,
                                  ScatterElementsTilingParams* param, const gert::Shape& data_origin_shape,
                               const gert::Shape& indices_origin_shape) {
  auto data_desc = context->GetInputDesc(INDEX_DATA);
  OP_CHECK_IF(data_desc == nullptr, OP_LOGE(context->GetNodeName(), "data_desc is null"), -1);
  const ge::DataType data_dtype = data_desc->GetDataType();
  // data, updates, output have same 'block' (because they have same dtype)
  auto iter = DATA_BLOCK_LEN.find(data_dtype);
  OP_CHECK_IF(iter == DATA_BLOCK_LEN.end(),
    OP_LOGE(context->GetNodeName(), "do not support data_dtype(%s)",
                                    ge::TypeUtils::DataTypeToAscendString(data_dtype).GetString()),
    return -1);
  const int32_t block = DATA_BLOCK_LEN[data_dtype];

  int32_t ub_for_data = compile_info->ub_size_bytes / UB_DATA_RATIO_LAST_AXIS; // ub space for data
  int32_t max_data_shape_in_dim = ub_for_data / BYTES_PER_BLOCK * block; // max data shape that ub can hold
  if ((param->axis + 1) != param->dims_data) {
    return -1;
  }
  if (data_origin_shape.GetDim(param->axis) <= max_data_shape_in_dim) {
    int32_t tiling_mode = TILING_MODE_LASTAXIS_SAME_SHAPE;
    // dims_data = 1 must be same shape cases
    if ((param->dims_data > MIN_DIM_FOR_SAMESHAPE_LASTAXIS)) {
      for (int32_t i = 0; i < param->axis; i++) {
        if (data_origin_shape.GetDim(i) != indices_origin_shape.GetDim(i)) {
            tiling_mode = TILING_MODE_LASTAXIS_DIFF_SHAPE;
            break;
        }
      }
    }
    if (tiling_mode == TILING_MODE_LASTAXIS_SAME_SHAPE) {
      return TILING_MODE_LASTAXIS_SAME_SHAPE;
    }
    if ((tiling_mode == TILING_MODE_LASTAXIS_DIFF_SHAPE) && (data_origin_shape.GetDim(param->axis) >= block)) {
      return TILING_MODE_LASTAXIS_DIFF_SHAPE;
    }
    OP_LOGD(context->GetNodeName(), "last axis diff shape cases, but data_shape_in_dim(%ld) < block(%d)",
            data_origin_shape.GetDim(param->axis), block);
  } else {
    OP_LOGD(context->GetNodeName(), "last axis cases, but data_shape_in_dim(%ld) > max_data_shape_in_dim(%d)",
            data_origin_shape.GetDim(param->axis), max_data_shape_in_dim);
  }
  return -1;
}

static ge::graphStatus CalMode(gert::TilingContext* context, const ScatterElementsCompileInfo* compile_info,
                        ScatterElementsTilingParams* param, const gert::Shape& data_origin_shape,
                        const gert::Shape& indices_origin_shape) {
  int32_t tiling_mode = CheckModeLastAxis(context, compile_info, param, data_origin_shape, indices_origin_shape);
  if (tiling_mode != -1) {
      param->tiling_mode = tiling_mode;
    return ge::GRAPH_SUCCESS;
  }

  int32_t params_row_data = 1;
  int32_t params_row_indices = 1;
  int32_t params_axis_data = 0;
  int32_t params_axis_indices = 0;
  tiling_mode = TILING_MODE_DIFFSHAPE;

  for (int32_t i = param->axis + 1; i < param->dims_data; i++) {
    params_row_data *= data_origin_shape.GetDim(i);
    params_row_indices *= indices_origin_shape.GetDim(i);
  }
  if (params_row_data == params_row_indices) {
    params_axis_data = params_row_data * data_origin_shape.GetDim(param->axis);
    params_axis_indices = params_row_indices * indices_origin_shape.GetDim(param->axis);
    if ((param->shape_acc_data / params_axis_data) == (param->shape_acc_indices / params_axis_indices)) {
      tiling_mode = TILING_MODE_SAMESHAPE;
    }
  }
  param->params_row_data = params_row_data;
  param->params_row_indices = params_row_indices;
  param->params_axis_data = params_axis_data;
  param->params_axis_indices = params_axis_indices;
  param->tiling_mode = tiling_mode;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ScatterElements(gert::TilingContext* context) {
  auto compile_info = reinterpret_cast<const ScatterElementsCompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  return ScatterElementsTilingForAscendC(context);
}

ge::graphStatus TilingPrepareScatterElementsForAscendC(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareScatterElementsForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<ScatterElementsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ScatterElements(gert::TilingParseContext* context) {
  auto compile_info = context->GetCompiledInfo<ScatterElementsCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  return TilingPrepareScatterElementsForAscendC(context);
}

// register tiling interface of the ScatterElements op.
IMPL_OP_OPTILING(ScatterElements)
  .Tiling(Tiling4ScatterElements)
  .TilingParse<ScatterElementsCompileInfo>(TilingPrepare4ScatterElements);
}  // namespace optiling
