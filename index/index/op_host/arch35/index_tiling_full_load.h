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
 * \file index_tiling_full_load.h
 * \brief Full load tiling class for Index operator
 */
#ifndef INDEX_TILING_FULL_LOAD_H
#define INDEX_TILING_FULL_LOAD_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/index_tiling_data.h"
#include "index_tiling.h"

namespace optiling {
using namespace Index;

class IndexFullLoadTiling : public IndexTilingCommon {
public:
	explicit IndexFullLoadTiling(gert::TilingContext* context) : IndexTilingCommon(context) {}

private:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus SetMaskMode();
    ge::graphStatus CalcUBBuffer();

private:
    uint64_t usedCoreNum_{0};
    gert::Shape inputShape_;
    gert::Shape maskShape_;
    gert::Shape outputShape_;
    int64_t maskIndices_{0};
    ge::DataType inputDtype_{ge::DT_UNDEFINED};
    ge::DataType indicesDtype_{ge::DT_UNDEFINED};
    int64_t ubIndices_{0};
    int64_t indicesTensorNum_{0};
    int64_t blockSize_{0};
    uint8_t maskMode_{0};
    int64_t lastAxisSize_{1};
    IndexFullLoadTilingData* tilingData_;
};

} // namespace optiling
#endif // INDEX_TILING_FULL_LOAD_H
