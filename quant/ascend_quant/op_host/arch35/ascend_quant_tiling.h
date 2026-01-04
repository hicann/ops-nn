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
 * \file ascend_quant_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ASCEND_QUANT_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ASCEND_QUANT_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
using namespace ge;
using namespace Ops::NN::Optiling;
constexpr int32_t BASE_DIM_NUM = 4;
constexpr int32_t SUBTRACTOR_TWO = 2;
constexpr int32_t SUBTRACTOR_THREE = 3;
constexpr int32_t SUBTRACTOR_FOUR = 4;
constexpr int64_t NC1HWC0_INDEX_N = 0;
constexpr int64_t NC1HWC0_INDEX_C1 = 1;
constexpr int64_t NC1HWC0_INDEX_H = 2;
constexpr int64_t NC1HWC0_INDEX_W = 3;
constexpr int64_t NC1HWC0_INDEX_C0 = 4;
constexpr int64_t NDC1HWC0_INDEX_N = 0;
constexpr int64_t NDC1HWC0_INDEX_D = 1;
constexpr int64_t NDC1HWC0_INDEX_C1 = 2;
constexpr int64_t NDC1HWC0_INDEX_H = 3;
constexpr int64_t NDC1HWC0_INDEX_W = 4;
constexpr int64_t NDC1HWC0_INDEX_C0 = 5;
constexpr int64_t CONST_INDEX_N = 0;
constexpr int64_t CONST_INDEX_C1 = 1;
constexpr int64_t CONST_INDEX_HW = 2;
constexpr int64_t CONST_INDEX_C0 = 3;
constexpr int64_t COM_INFO_UB_INDEX = 0;
constexpr int64_t COM_INFO_CORE_INDEX = 1;
constexpr int64_t COM_INFO_FUSE_FLAG_INDEX = 2;
const std::string CONST = "const";
const std::map<int, int> c0_map = {{ge::DT_INT8, 32}, {ge::DT_INT4, 64}};
const std::map<int, int> c1_trans_map = {{ge::DT_INT8, 2}, {ge::DT_INT4, 4}};

struct AscendQuantCompileInfo {
    std::vector<int64_t> var_index_list;
    std::vector<int64_t> info;
    std::string mode;
};

struct TilingParams {
    int32_t block_dim{0};
    int32_t block_tiling_axis{-1};
    int64_t block_factor{1};
    int32_t ub_tiling_axis{-1};
    int64_t ub_factor{1};
    int32_t is_fuse_block{1};
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_ASCEND_QUANT_H_
