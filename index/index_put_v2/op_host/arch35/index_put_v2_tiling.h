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
 * \file index_put_v2.h
 * \brief dsl index put tiling file
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_BOUNDING_BOX_DECODE_RUNTIME2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_BOUNDING_BOX_DECODE_RUNTIME2_H

#include <cstdint>
namespace optiling {
struct IndexPutV2CompileInfo {
  int64_t ub_max_size;
  int64_t x_data_each_block;
  int64_t core_num;
  int64_t each_repeat_block_number;
  int64_t indices_count;
  int64_t soc_version;
};

struct IndexPutV2Params {
  // calcu param
  int64_t box_number;
  int64_t core_data;
  int64_t core_used;
  int64_t copy_loop;
  int64_t copy_tail;
  int64_t last_copy_loop;
  int64_t last_copy_tail;
  int64_t reserver_dim_number;
  int64_t value_number;
  int64_t indices_number;
  int64_t indices_list_number;
  int64_t dim_0;
  int64_t dim_1;
  int64_t dim_2;
  int64_t dim_3;
  int64_t dim_4;
  int64_t dim_5;
  int64_t dim_6;
  int64_t dim_7;
  int64_t reserver_dim_0;
  int64_t reserver_dim_1;
  int64_t reserver_dim_2;
  int64_t reserver_dim_3;
  int64_t reserver_dim_4;
  int64_t reserver_dim_5;
  int64_t reserver_dim_6;
  int64_t reserver_dim_7;
  int64_t tiling_mode;
  int64_t available_ub_size;
  int64_t tiling_core_num;
  int64_t mask_dim;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_BOUNDING_BOX_DECODE_RUNTIME2_H