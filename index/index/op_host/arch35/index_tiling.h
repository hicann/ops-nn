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
 * \file index.h
 * \brief dsl index tiling h
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_V2_INDEX_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_V2_INDEX_H

#include <cstdint>

namespace optiling {
struct IndexTilingData {
    int64_t batch_num_list[10];
    int64_t core_num_var;
    int64_t output_size;
    int64_t output_num_per_batch;
    int64_t output_batch_num;
    int64_t input_size;
    int64_t input_shape_dim_num;
    int64_t index_size;
    int64_t masks_num;
    int64_t tiling_mode;
};

struct IndexCompileInfo {
    int32_t task_align;
    int32_t core_num;
    int32_t available_size;
    int32_t indices_num;
    int32_t align_num;
    int32_t after_v200;
    uint64_t ubSize;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_V2_INDEX_H