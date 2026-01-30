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
 * \file index_tiling.h
 * \brief ac index tiling h
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_H_
#pragma once
#include "index_tiling_common.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

/////////////////////////////////////
// tilingdata define
/////////////////////////////////////
BEGIN_TILING_DATA_DEF(IndexDefaultTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputLength);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(IndexSimtTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputLength);
TILING_DATA_FIELD_DEF(uint64_t, outputLength);
TILING_DATA_FIELD_DEF(uint64_t, indexSize);
TILING_DATA_FIELD_DEF(uint32_t, inputDimNum);
TILING_DATA_FIELD_DEF(uint32_t, indexedDimNum);
TILING_DATA_FIELD_DEF(uint32_t, indexedSizesNum);
TILING_DATA_FIELD_DEF(uint32_t, accumulateMode);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, inputShape);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(IndexPerfSimtTilingData)
TILING_DATA_FIELD_DEF(uint64_t, outputLength);
TILING_DATA_FIELD_DEF(uint64_t, inputShape0);
TILING_DATA_FIELD_DEF(uint64_t, inputShape1);
END_TILING_DATA_DEF;

// simt template ascendc tools
REGISTER_TILING_DATA_CLASS(Index, IndexDefaultTilingData);
REGISTER_TILING_DATA_CLASS(Index_1001, IndexPerfSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_1002, IndexPerfSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_1004, IndexPerfSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_1008, IndexPerfSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_11001, IndexPerfSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_11002, IndexPerfSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_11004, IndexPerfSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_11008, IndexPerfSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_1, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_2, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_4, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_8, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_16, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_10001, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_10002, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_10004, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_10008, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(Index_10016, IndexSimtTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2, IndexSimtTilingData);

class IndexSimtTiling : public IndexTilingCommon {
public:
    explicit IndexSimtTiling(gert::TilingContext* context) : IndexTilingCommon(context)
    {}
    bool isIndexPut_{false};

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    // customized functions
    ge::graphStatus GetParamsShapeInfo();
    uint32_t ParamsDtypeImprove(uint32_t lastDimSize, uint32_t dataTypeBytes);
    uint64_t UpdateTilingData();
    ge::graphStatus GenerateTilingKey();
    ge::graphStatus GenIndexTilingKey();

private:
    bool accumulateMode_;
    uint64_t inputShapes_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t inputLength_;
    uint32_t inputDimNum_;
    uint64_t outputLength_;
    uint32_t indexedDimNum_;
    uint32_t tilingKey_;
    bool isPerfTemplate_{false};
    IndexSimtTilingData tilingData_;
    IndexPerfSimtTilingData PerfTilingData_;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_H_
