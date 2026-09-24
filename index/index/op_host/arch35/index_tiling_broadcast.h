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
 * \file index_tiling_broadcast.h
 * \brief Broadcast tiling class for Index operator (indices broadcast scenario)
 */

#ifndef INDEX_TILING_BROADCAST_H
#define INDEX_TILING_BROADCAST_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/index_tiling_data.h"
#include "index_tiling.h"

namespace optiling {
using namespace Index;

class IndexBroadcastTiling : public IndexTilingCommon {
public:
    explicit IndexBroadcastTiling(gert::TilingContext* context) : IndexTilingCommon(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    // customized functions
    ge::graphStatus ComputeBroadcastInfo();
    uint32_t ParamsDtypeImprove(uint32_t lastDimSize, uint32_t dataTypeBytes);
    uint64_t UpdateTilingData();

private:
    uint64_t inputShapes_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t inputLength_ = 0;
    uint32_t inputDimNum_ = 0;
    uint64_t outputLength_ = 0;
    uint32_t indicesNum_ = 0;
    uint32_t indexedSizesNum_ = 0;
    uint64_t indexSize_ = 0;
    uint32_t broadcastDimNum_ = 0;
    uint64_t broadcastShape_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t indexBcStride_[8][8] = {{0}};
    uint64_t factor4Index_ = 1;
    uint64_t xDtype_ = 0;
    IndexBroadcastTilingData* tilingData_{nullptr};
};

} // namespace optiling
#endif // INDEX_TILING_BROADCAST_H
