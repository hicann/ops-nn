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
 * \brief index tiling full load
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_FULL_LOAD_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_FULL_LOAD_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
static constexpr int64_t MAX_DIM_NUM = 8;

BEGIN_TILING_DATA_DEF(IndexFullLoadTilingData)
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_DIM_NUM, inputStride);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_DIM_NUM, inputShape);
TILING_DATA_FIELD_DEF(uint32_t, indicesNum);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, eachCoreSize);
TILING_DATA_FIELD_DEF(int64_t, normalCoreLoop);
TILING_DATA_FIELD_DEF(int64_t, normalCoreTail);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreTail);
TILING_DATA_FIELD_DEF(int64_t, lastDimSize);
TILING_DATA_FIELD_DEF(int64_t, inputQueSize);
TILING_DATA_FIELD_DEF(int64_t, indicesQueSize);
TILING_DATA_FIELD_DEF(int64_t, outputQueSize);
TILING_DATA_FIELD_DEF(int64_t, ubIndices);
TILING_DATA_FIELD_DEF(int64_t, inputShapeSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Index_211, IndexFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(Index_220, IndexFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(Index_221, IndexFullLoadTilingData)

class IndexFullLoadTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit IndexFullLoadTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~IndexFullLoadTiling() override
    {}

private:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus SetMaskMode();
    ge::graphStatus CalcUBBuffer();

private:
    uint64_t coreNum_{0};
    uint64_t usedCoreNum_{0};
    uint64_t ubSize_{0};
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
    IndexFullLoadTilingData tilingData_;
};
} // namespace optiling
#endif