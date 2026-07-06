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
 * \file dynamic_dual_level_mx_quant_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_DUAL_LEVEL_MX_QUANT_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_DUAL_LEVEL_MX_QUANT_TILING_H
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/arch35/dynamic_dual_level_mx_quant_tiling_data.h"

namespace optiling {

struct DynamicDualLevelMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicDualLevelMxQuantTilingParam {
    int64_t tilingKey = 0;
    int64_t totalCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t ubSize = 0;
    int64_t roundMode = 0;
    int64_t level0BlockSize = 512;
    int64_t level1BlockSize = 32;
    int64_t rowSize = 0;
    int64_t colSize = 1;
    int64_t blockSizeRow = 0;
    int64_t blockSizeCol = 0;
    int64_t rowBlockNum = 0;
    int64_t colBlockNum = 0;
    int64_t rowTileNum = 0;
    int64_t colTileNum = 0;
    int64_t normalTileRowBlockNum = 0;
    int64_t normalTileColBlockNum = 0;
    int64_t tailTileRowBlockNum = 0;
    int64_t tailTileColBlockNum = 0;
    int64_t normalTileRowSize = 0;
    int64_t tailTileRowSize = 0;
    int64_t normalTileRowLoopNum = 0;
    int64_t normalTileColLoopNum = 0;
    int64_t tailTileRowLoopNum = 0;
    int64_t tailTileColLoopNum = 0;
    int64_t ubFactor = 0;
    int64_t tailAlignNum = 0;
    int64_t copyMethod = 0;
    int64_t needSmoothScale = 0;
    int64_t workspaceSize = 0;
};

enum class RoundModeList {
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

class DynamicDualLevelMxQuantTiling {
public:
    explicit DynamicDualLevelMxQuantTiling(gert::TilingContext* context) : context_(context){};
    ~DynamicDualLevelMxQuantTiling(){};
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetAttr();
    ge::graphStatus CheckRequiredDtype() const;
    ge::graphStatus CheckRequiredShape() const;
    ge::graphStatus CheckSmoothScaleDtypeShape();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus MergeAxis();
    ge::graphStatus SetTilingParams();
    ge::graphStatus AutoTiling();

    void SetTilingKey();
    void SetTilingData();
    void PrintTilingData();
    ge::graphStatus SplitCore();

    RoundModeList GetRoundMode(const std::string& roundMode);

    template <typename T>
    ge::graphStatus GetAndValidateAttr(const gert::RuntimeAttrs* attrs, int64_t index, T& target, T expectedValue,
                                       const std::string& attrName, const std::string& expectedMsg) const;

private:
    uint64_t roundMode_ = 0;
    gert::TilingContext* context_ = nullptr;
    DynamicDualLevelMxQuantTilingData* tilingData = nullptr;
    DynamicDualLevelMxQuantTilingParam tilingParams{};
    bool needSmoothScale = false;
    mutable size_t dimNum_ = 0;
    mutable int64_t lastDim_ = 0;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_DUAL_LEVEL_MX_QUANT_TILING_H
