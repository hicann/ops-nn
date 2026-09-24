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
 * \file index_tiling_nocon_broadcast.h
 * \brief Non-continuous view x indices broadcast tiling class for Index operator (priority 5)
 */
#ifndef INDEX_TILING_NOCON_BROADCAST_H
#define INDEX_TILING_NOCON_BROADCAST_H

#include <string>
#include <vector>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/index_tiling_data.h"
#include "index_tiling.h"
#include "index_tiling_no_continuous.h"

namespace optiling {
using namespace Index;
class IndexNoConBroadcastTiling : public IndexNonContinuousTiling {
public:
    explicit IndexNoConBroadcastTiling(gert::TilingContext* context) : IndexNonContinuousTiling(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus ComputeBroadcastInfo();
    void SetTilingData();

private:
    static constexpr uint32_t NOCON_BC_MAX_DIM = 4;
    uint32_t broadcastDimNum_ = 0;
    int64_t broadcastShape_[NOCON_BC_MAX_DIM] = {0, 0, 0, 0};
    int64_t indexBcStride_[NOCON_BC_MAX_DIM][NOCON_BC_MAX_DIM] = {{0}};
    int64_t indexInputShape_[NOCON_BC_MAX_DIM] = {0, 0, 0, 0};
    uint64_t indexSize_ = 0; // numel(B)
    gert::Shape indexShapesVec_[NOCON_BC_MAX_DIM];
    gert::Stride indexStridesVec_[NOCON_BC_MAX_DIM];
    IndexNoConBroadcastTilingData* tilingData_{nullptr};
};
} // namespace optiling
#endif // INDEX_TILING_NOCON_BROADCAST_H
