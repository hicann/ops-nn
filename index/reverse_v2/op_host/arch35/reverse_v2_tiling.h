/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reverse.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_H_

#include <nlohmann/json.hpp>

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
if ((ptr) == nullptr) {                                                                        \
  const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
  OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                        \
  REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                           \
  return ret;                                                                                  \
}

namespace optiling {
const size_t MAX_REVERSE_LEN = 8;
// elements num in one block for int16
const int64_t F16_NUM_ONE_BLOCK = 16;
// vnhwc process the min numbers
const int64_t F16_NUM_FOR_VNHWC = 256;
const int64_t MIN_MERGED_SIZE = 7;
const int64_t SIZE_SPLIT_THRESH = 512;

// define tiling data
// tiling_0: do not reverse with last dim, and last dim < 16
constexpr int64_t KEY_0 = 0;
// tiling_1: do not reverse with last dim, and last dim is 16 align and < 512
constexpr int64_t KEY_1 = 1;
// tiling_2: do not reverse with last dim, and last dim is not 16 align and < 512
constexpr int64_t KEY_2 = 2;
// tiling_3: do not reverse with last dim, and last dim > 512
constexpr int64_t KEY_3 = 3;
// tiling_4: do reverse with last dim, and last dim < 128
constexpr int64_t KEY_4 = 4;
// tiling_5: do reverse with last dim, and last dim > 128 and < 512
constexpr int64_t KEY_5 = 5;
// tiling_6: do reverse with last dim, and last dim > 512
constexpr int64_t KEY_6 = 6;
// tiling_11: do reverse with topk, when last dim is less
constexpr int64_t KEY_11 = 11;

using StatusArray = std::array<bool, MAX_REVERSE_LEN + 1>;
using ShapeArray = std::array<int64_t, MAX_REVERSE_LEN + 1>;

template <typename T>
bool ReadCompileItem(const nlohmann::json& all_vars, const std::string& name, T& value) {
  if (all_vars.empty()) {
    return false;
  }

  if (all_vars.count(name) == 0) {
    return false;
  }

  value = all_vars[name].get<T>();
  return true;
}

struct ResizeV2TilingData {
  int64_t tiling_key;
  std::array<int64_t, MAX_REVERSE_LEN - 1> inner_shape_array{};
  std::array<int64_t, MAX_REVERSE_LEN - 1> inner_axis_array{};
  std::array<int64_t, MAX_REVERSE_LEN - 1> outer_shape_array{};
  std::array<int64_t, MAX_REVERSE_LEN - 1> outer_axis_array{};
  int64_t is_split_axis_reverse;
  int64_t split_part_num;
  int64_t split_dim;
  // add for performance
  int64_t inner_real_dims;
  int64_t outer_real_dims;
  int64_t core_num;

  void Init() {
    for (size_t i = 0; i < inner_shape_array.size(); ++i) {
      inner_shape_array[i] = 1;
      inner_axis_array[i] = 0;
    }
    for (size_t i = 0; i < outer_shape_array.size(); ++i) {
      outer_shape_array[i] = 1;
      outer_axis_array[i] = 0;
    }
    is_split_axis_reverse = 0;
    split_dim = 0;
    inner_real_dims = 0;
    outer_real_dims = 0;
    core_num = 0;
  }

  std::string ToString() const {
    std::string string_out = "tiling key = " + std::to_string(tiling_key) + "; ";
    string_out = string_out + "tiling data = [" + std::to_string(tiling_key) + " ";
    for (size_t i = 0; i < inner_shape_array.size(); ++i) {
      string_out = string_out + std::to_string(inner_shape_array[i]) + " ";
    }
    for (size_t i = 0; i < inner_axis_array.size(); ++i) {
      string_out = string_out + std::to_string(inner_axis_array[i]) + " ";
    }
    for (size_t i = 0; i < outer_shape_array.size(); ++i) {
      string_out = string_out + std::to_string(outer_shape_array[i]) + " ";
    }
    for (size_t i = 0; i < outer_axis_array.size(); ++i) {
      string_out = string_out + std::to_string(outer_axis_array[i]) + " ";
    }
    string_out = string_out + std::to_string(is_split_axis_reverse) + " ";
    string_out = string_out + std::to_string(split_part_num) + " ";
    string_out = string_out + std::to_string(split_dim) + " ";
    string_out = string_out + std::to_string(inner_real_dims) + " ";
    string_out = string_out + std::to_string(outer_real_dims) + " ";
    string_out = string_out + std::to_string(core_num) + " ";
    string_out = string_out + "]";

    return string_out;
  }
};

struct ReverseRunInfo {
  ShapeArray reverse_shape;
  StatusArray reverse_status;
  size_t real_shape_size;
  int64_t shape_size;
  // used for merging the reverse_shape
  int64_t merged_wr_idx;
  // save the shape[0] value before do merge shape
  int64_t first_dim_num;

  void Init() {
    for (size_t i = 0; i < reverse_status.size(); ++i) {
      reverse_status[i] = false;
      reverse_shape[i] = 1;
    }
    real_shape_size = static_cast<size_t>(0);
    shape_size = 0;
    // when merge the axes, will write data from back to front
    // wr_idx will point the last data at first
    merged_wr_idx = reverse_status.size() - 1;
    first_dim_num = 1;
  }

  std::string OriginInfoToString() const {
    std::string shape_str = "reverse_shape: [";
    std::string status_str = "reverse_status: [";
    for (size_t i = 0; i < real_shape_size; ++i) {
      shape_str = shape_str + std::to_string(reverse_shape[i]) + ", ";
      status_str = status_str + std::to_string(reverse_status[i]) + ", ";
    }
    shape_str = shape_str + "]";
    status_str = status_str + "]";

    std::string string_out = shape_str + "; " + status_str;
    string_out = string_out + "; real_shape_size = " + std::to_string(real_shape_size) + ", ";
    string_out = string_out + "; shape_size = " + std::to_string(shape_size) + ", ";
    string_out = string_out + "; first_dim_num = " + std::to_string(first_dim_num) + ", ";

    return string_out;
  }

  std::string MergeInfoToString() const {
    std::string shape_str = "reverse_shape: [";
    std::string status_str = "reverse_status: [";
    for (size_t i = merged_wr_idx; i < reverse_status.size(); ++i) {
      shape_str = shape_str + std::to_string(reverse_shape[i]) + ", ";
      status_str = status_str + std::to_string(reverse_status[i]) + ", ";
    }
    shape_str = shape_str + "]";
    status_str = status_str + "]";

    std::string string_out = shape_str + "; " + status_str;
    string_out = string_out + "; shape_size = " + std::to_string(shape_size) + ", ";
    string_out = string_out + "; merged_wr_idx = " + std::to_string(merged_wr_idx) + ", ";
    string_out = string_out + "; first_dim_num = " + std::to_string(first_dim_num) + ", ";

    return string_out;
  }
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_REVERSE_H_
