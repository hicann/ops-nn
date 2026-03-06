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
 * \file unsorted_segment_sum_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_UNSORTED_SEGMENT_SUM_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_UNSORTED_SEGMENT_SUM_H


#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "util/const_util.h"
#include <nlohmann/json.hpp>


namespace optiling {

template <typename T>
static inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}

  /*
 * @brief: get the json class of compile info from context
 * @param [in] context: gert::TilingContext
 * @return bool: std::unique_ptr<nlohmann::json>;
 */
inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context) {
  auto json_str = context->GetCompiledJson();
  OP_CHECK_IF(json_str == nullptr, OP_LOGE(context->GetNodeName(), "json_str is nullptr!"), return nullptr);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = std::make_unique<nlohmann::json>(nlohmann::json::parse(json_str));
  return parsed_object_cinfo;
}

struct TilingPrepareForUnsortedSegmentSumCompileInfo {
  int32_t core_num;
  int32_t ub_size;
  int32_t ub_tensor_num;
  int32_t impl_mode;
  uint32_t max_thread{512};
};

struct UnsortedSegmentSumTilingParamsFp32 {
  // common params
  int32_t select_key_input_scalar;
  int32_t need_core_num_input_scalar;

  // input data params
  // front core
  int32_t input_ele_num_front_core_input_scalar;
  // front part front core
  int32_t input_mov_times_gm2ub_front_part_front_core_input_scalar;
  int32_t input_front_burst_len_front_part_front_core_input_scalar;
  int32_t input_last_burst_len_front_part_front_core_input_scalar;
  int32_t input_front_ele_num_ub_front_part_front_core_input_scalar;
  int32_t input_last_ele_num_ub_front_part_front_core_input_scalar;
  int32_t input_front_rows_front_part_front_core_input_scalar;
  int32_t input_last_rows_front_part_front_core_input_scalar;
  // last part front core
  int32_t input_mov_times_gm2ub_last_part_front_core_input_scalar;
  int32_t input_front_burst_len_last_part_front_core_input_scalar;
  int32_t input_last_burst_len_last_part_front_core_input_scalar;
  int32_t input_front_ele_num_ub_last_part_front_core_input_scalar;
  int32_t input_last_ele_num_ub_last_part_front_core_input_scalar;
  int32_t input_front_rows_last_part_front_core_input_scalar;
  int32_t input_last_rows_last_part_front_core_input_scalar;
  // last core
  int32_t input_ele_num_last_core_input_scalar;
  // front part last core
  int32_t input_mov_times_gm2ub_front_part_last_core_input_scalar;
  int32_t input_front_burst_len_front_part_last_core_input_scalar;
  int32_t input_last_burst_len_front_part_last_core_input_scalar;
  int32_t input_front_ele_num_ub_front_part_last_core_input_scalar;
  int32_t input_last_ele_num_ub_front_part_last_core_input_scalar;
  int32_t input_front_rows_front_part_last_core_input_scalar;
  int32_t input_last_rows_front_part_last_core_input_scalar;
  // last part last core
  int32_t input_mov_times_gm2ub_last_part_last_core_input_scalar;
  int32_t input_front_burst_len_last_part_last_core_input_scalar;
  int32_t input_last_burst_len_last_part_last_core_input_scalar;
  int32_t input_front_ele_num_ub_last_part_last_core_input_scalar;
  int32_t input_last_ele_num_ub_last_part_last_core_input_scalar;
  int32_t input_front_rows_last_part_last_core_input_scalar;
  int32_t input_last_rows_last_part_last_core_input_scalar;

  // e num params
  int32_t e_num_input_scalar;
  int32_t e_mov_times_gm2ub_input_scalar;
  int32_t e_ub2gm_front_burst_len_input_scalar;
  int32_t e_num_front_part_input_scalar;
  int32_t e_ub2gm_last_burst_len_input_scalar;
  int32_t e_num_last_part_input_scalar;

  // ids params
  int32_t ids_size_input_scalar;
  int32_t ids_ele_num_front_core_input_scalar;
  int32_t ids_mov_times_gm2ub_front_core_input_scalar;
  int32_t ids_front_burst_len_front_core_input_scalar;
  int32_t ids_last_burst_len_front_core_input_scalar;
  int32_t ids_ele_num_ub_front_part_front_core_input_scalar;
  int32_t ids_ele_num_ub_last_part_front_core_input_scalar;
  int32_t ids_ele_num_last_core_input_scalar;
  int32_t ids_mov_times_gm2ub_last_core_input_scalar;
  int32_t ids_front_burst_len_last_core_input_scalar;
  int32_t ids_last_burst_len_last_core_input_scalar;
  int32_t ids_ele_num_ub_front_part_last_core_input_scalar;
  int32_t ids_ele_num_ub_last_part_last_core_input_scalar;

  // output init params
  int32_t output_ub_init_last_repeat_time_front_part_front_core_input_scalar;
  int32_t output_ub_init_times_front_part_front_core_input_scalar;
  int32_t output_ub_init_last_repeat_time_last_part_front_core_input_scalar;
  int32_t output_ub_init_times_last_part_front_core_input_scalar;
  int32_t output_ub_init_last_repeat_time_front_part_last_core_input_scalar;
  int32_t output_ub_init_times_front_part_last_core_input_scalar;
  int32_t output_ub_init_last_repeat_time_last_part_last_core_input_scalar;
  int32_t output_ub_init_times_last_part_last_core_input_scalar;
  int32_t input_last_axis_align_front_part_ele_num_input_scalar;
  int32_t input_last_axis_align_floor_ele_num_input_scalar;
  int32_t last_part_vadd_mask_input_scalar;
  int32_t e_gm2ub_last_burst_len_input_scalar;
  int32_t output_ub_init_last_row_last_repeat_time_front_part_front_core_input_scalar;
  int32_t output_ub_init_last_row_times_front_part_front_core_input_scalar;
  int32_t output_ub_init_last_row_last_repeat_time_last_part_front_core_input_scalar;
  int32_t output_ub_init_last_row_times_last_part_front_core_input_scalar;
  int32_t output_ub_init_last_row_last_repeat_time_front_part_last_core_input_scalar;
  int32_t output_ub_init_last_row_times_front_part_last_core_input_scalar;
  int32_t output_ub_init_last_row_last_repeat_time_last_part_last_core_input_scalar;
  int32_t output_ub_init_last_row_times_last_part_last_core_input_scalar;

  int32_t num_segments;

  int32_t tiling_core_num;
  int32_t repeat_front_front_part_front_core;
  int32_t col_sub_block_front_front_part_front_core;
  int32_t repeat_last_front_part_front_core;
  int32_t col_sub_block_last_front_part_front_core;
  int32_t repeat_front_last_part_front_core;
  int32_t col_sub_block_front_last_part_front_core;
  int32_t repeat_last_last_part_front_core;
  int32_t col_sub_block_last_last_part_front_core;
  int32_t repeat_front_front_part_last_core;
  int32_t col_sub_block_front_front_part_last_core;
  int32_t repeat_last_front_part_last_core;
  int32_t col_sub_block_last_front_part_last_core;
  int32_t repeat_front_last_part_last_core;
  int32_t col_sub_block_front_last_part_last_core;
  int32_t repeat_last_last_part_last_core;
  int32_t col_sub_block_last_last_part_last_core;
  int32_t e_num_sub;
  int32_t vadd_repeat_255;
  int32_t vadd_repeat_64;
  int32_t vadd_repeat_last;
  int32_t move_pad;
  int32_t max_cache_n_num;
  int32_t repeat_remove_pad;
  int32_t col_block_remove_pad;
  int32_t cache_num_block;
};

struct UnsortedSegmentSumTilingParamsInt32 {
  // common params
  int32_t select_key_input_scalar;
  int32_t need_core_num_input_scalar;
  int32_t num_segments_front_core_input_scalar;
  int32_t num_segments_last_core_input_scalar;

  // ids params
  int32_t ids_size_input_scalar;
  int32_t ids_mov_times_gm2ub_input_scalar;
  int32_t ids_ele_num_ub_front_part_input_scalar;
  int32_t ids_front_burst_len_input_scalar;
  int32_t ids_ele_num_ub_last_part_input_scalar;
  int32_t ids_last_burst_len_input_scalar;

  // e num params
  int32_t e_num_input_scalar;
  int32_t e_mov_times_gm2ub_input_scalar;
  int32_t e_ub2gm_front_burst_len_input_scalar;
  int32_t e_num_front_part_input_scalar;
  int32_t repeat_time_front_part_input_scalar;
  int32_t e_ub2gm_last_burst_len_input_scalar;
  int32_t e_num_last_part_input_scalar;
  int32_t repeat_time_last_part_input_scalar;
  int32_t align_scalar;
  int32_t align_scalar_lastcore;

  int32_t e_gm2ub_front_burst_len_input_scalar;
  int32_t e_gm2ub_last_burst_len_input_scalar;
  int32_t num_segment_max;
  int32_t num_segment_max_time;
  int32_t num_segment_max_time_lastcore;
  int32_t front_num_segment;
  int32_t front_num_segment_last;
  int32_t front_num_segment_lastcore;
  int32_t front_num_segment_last_lastcore;
  int32_t e_ub2gm_front_burst_len_input_scalar_lastcore;
  int32_t e_ub2gm_last_burst_len_input_scalar_lastcore;
  int32_t repeat_times;
  int32_t repeat_times_last_part;
  int32_t repeat_times_last_part_lastcore;
  int32_t e_mov_times_gm2ub_input_scalar_lastcore;
  int32_t repeat_time_front_part_input_scalar_lastcore;
  int32_t num_segments;

  int32_t tiling_core_num;
};
}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_UNSORTED_SEGMENT_SUM_H
