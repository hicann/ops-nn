/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file gather_v2_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_V2_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_V2_H_
#include <cstdint>
#include <nlohmann/json.hpp>
#include <sstream>

namespace optiling {
struct GatherV2CompileInfo {
  int64_t ub_size{1};
  int64_t block_size{1};
  int64_t l1_size{0};
  int64_t core_num{1};
  int64_t params_dsize{1};
  int64_t indices_dsize{1};
  bool is_tik{false};
  bool is_gather_v2{true};
  int64_t impl_mode{0};
  bool is_preprocessed;
  int64_t socVersion{0};
};

struct GatherV2TilingParams {
  int64_t tiling_mode = 0;
  int64_t params_pre = 1;
  int64_t params_axis = 1;
  int64_t params_row = 1;
  int64_t indices_num = 1;
  int64_t cache_params = 0;
  int64_t need_core_num = 0;
  int64_t tail_process_core = 0;
  int64_t indices_num_each_core = 0;
  int64_t indices_num_remaining = 0;
  int64_t indices_loop_num = 0;
  int64_t indices_row_num_once = 0;
  int64_t indices_row_num_last = 0;
  int64_t row_num_once_ub = 0;
  int64_t row_num_once_tail_ub = 0;
  int64_t inner_loop_num = 0;
  int64_t row_num_last_ub = 0;
  int64_t row_num_last_tail_ub = 0;
  int64_t inner_loop_num_last = 0;
  int64_t params_total = 0;
  int64_t one_row_loop = 0;
  int64_t one_row_tail = 0;
  int64_t params_pre_each_core = 0;
  int64_t params_pre_remaining = 0;
  int64_t indices_row = 1;
  int64_t params_batch_each_core = 1;
  int64_t params_batch_remaining = 0;
  int64_t params_batch = 1;
  int64_t socVersion = 0;
  int64_t x_split_num = 1;
  int64_t x_buffer_size = 0;
  int64_t indices_buffer_size = 0;
  int64_t indices_split_num = 0;
  int64_t mask_buffer_size = 0;
  int64_t total_count_num = 0;
  int64_t count_time = 0;
  int64_t median_core_id = 0;
  int64_t pre_core_compute_size = 0;
  int64_t median_core_compute_size = 0;
  int64_t post_core_compute_size = 0;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GATHER_V2_RUNTIME2_H_

