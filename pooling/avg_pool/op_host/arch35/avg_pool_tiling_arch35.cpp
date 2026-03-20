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
 * \file avg_pool_tiling_arch35.cpp
 * \brief tiling function of avg_pool
 */
#include <vector>
#include "register/op_impl_registry.h"
#include "pooling/avg_pool/op_host/avg_pool_tiling_common.h"

struct CubeTilingCommonParseInfo {
    int32_t fmapC1 = 0;
    bool correctRangeFlag = false;
    std::string tilingType = "";
    std::vector<std::string> varMap;
    std::vector<std::string> tilingKeyList;
    std::vector<std::vector<std::string>> customVarsList;
    std::vector<std::vector<int64_t>> defaultRangeList;
    std::vector<std::vector<int64_t>> tilingRangeList;
    std::vector<int32_t> numBlocksList;
    std::vector<std::vector<int32_t>> repoSeedsList;
    std::vector<std::vector<int64_t>> repoRangeList;
    std::vector<std::vector<int64_t>> costRangeList;
};

struct AvgPoolTilingParseInfo : CubeTilingCommonParseInfo {
  int64_t stridesH = 0;
  int64_t stridesW = 0;
  int64_t kSizeH = 0;
  int64_t kSizeW = 0;
  int64_t filter = 0;
  int64_t ub_ele = 0;
  int64_t core_num = 0;
  int64_t ksize_h = 0;
  int64_t ksize_w = 0;
  int64_t strides_h = 0;
  int64_t strides_w = 0;
  int64_t padding = 0;
};

namespace gert {
/**
 * @brief Op tiling compile info parse function of AvgPool (runtime2.0).
 * @param [inout] context
 * @return succeeded or not
 */
ge::graphStatus TilingPrepareForAvgPool(TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief Op tiling function of AvgPool (runtime2.0).
 * @param [inout] context
 * @return ucceeded or not
 */
ge::graphStatus TilingForAvgPool(TilingContext* context)
{
    return optiling::Tiling4AvgPoolRegBase(context);
}

// register op tiling interface of AvgPool (runtime2.0)
IMPL_OP_OPTILING(AvgPool).Tiling(TilingForAvgPool).TilingParse<AvgPoolTilingParseInfo>(TilingPrepareForAvgPool);
} // namespace gert
