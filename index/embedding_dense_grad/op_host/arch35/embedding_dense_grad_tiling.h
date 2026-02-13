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
 * \file embedding_dense_grad_tiling.h
 * \brief embedding_dense_grad_tiling
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_EMBEDDING_DENSE_GRAD_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_EMBEDDING_DENSE_GRAD_TILING_H

#include <cstdint>

namespace optiling {
struct EmbeddingDenseGradCompileInfo {
  int32_t core_num;
  int32_t scale_grad_by_freq;
  uint32_t maxCoreNum{0};
  uint32_t ubSizePlatform{0};
  uint32_t maxThreadNum{0};
};

struct EmbeddingDenseGradTilingParams {
  int32_t numel_indices;
  int32_t embedding_dim;
  int32_t mode_of_cal;
  int32_t core_num;
  int32_t num_weights;
  int32_t padding_idx;
  int32_t tiling_core_num;
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_EMBEDDING_DENSE_GRAD_TILING_H

