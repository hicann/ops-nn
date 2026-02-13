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
 * \file gather_nd_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_ND_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_ND_TILING_H_
#include <cstdint>

namespace optiling {
const int32_t LAST_DIM_MAX = 8;

struct GatherNdCompileInfo {
  int32_t core_num;
  int32_t ub_size;
  int32_t l1_size;
  int32_t params_dsize;
  int32_t indices_dsize;
  bool is_tik{false};
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GATHER_ND_TILING_RUNTIME2_H_ 