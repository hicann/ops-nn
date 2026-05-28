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
 * \file scatter_nd_common_base_tiling.h
 * \brief scatter_nd_common_base_tiling
 */

#ifndef SCATTER_ND_COMMON_BASE_TILING_H
#define SCATTER_ND_COMMON_BASE_TILING_H


#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/arch35/scatter_nd_common_struct.h"
#include "../op_kernel/arch35/scatter_nd_max_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

constexpr uint16_t OPTILING_MAX_RANK_COUNT = 7;
constexpr uint16_t OPTILING_MAX_SHAPE_RANK = 8;
static constexpr uint16_t INPUT_IDX_INDICES = 1;
static constexpr uint16_t INPUT_IDX_UPDATES = 2;
static constexpr uint16_t OUTPUT_IDX_SHAPE = 0;
static constexpr uint32_t ONE = 1;
static constexpr int64_t TWO = 2;

struct ScatterNdCommonCompileInfo {
  int64_t core_num;
  int64_t ub_size;
};

class ScatterNdCommonBaseTiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit ScatterNdCommonBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }
    ~ScatterNdCommonBaseTiling() override
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override
    {}

    uint32_t GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend);
    ge::graphStatus GetCastType();
    void SetStride();

public:
    uint64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t varTypeSize_ = 0;
    int64_t indicesTypeSize_ = 0;
    int64_t outOfSetTypeSize_ = 0;
    uint32_t rankSize_ = 0;
    int64_t updateShapeSize_ = 0;
    uint64_t outputShapeSize_ = 1;
    int64_t indiceShapeSize_ = 0;
    int64_t indicesAxis_ = 0;
    int64_t varInAxis_ = 1;
    int64_t afterAxis_ = 1;
    uint64_t indiceCastMode_ = 0;  // 0: 不Cast; 1：int32 Cast int16; 2：int64 Cast int32; 3：int64 Cast int16; 4:int32 Cast uint8; 5:int64 Cast uint8.
    int64_t indiceCastDtypeSize_ = 0;
    uint64_t strideList_[OPTILING_MAX_RANK_COUNT] = {0};
    uint64_t outPutShape_[OPTILING_MAX_SHAPE_RANK] = {0};

    ge::DataType updateDtype_ = ge::DT_UNDEFINED;
    ge::DataType indiceDtype_ = ge::DT_UNDEFINED;
    ge::DataType indiceCastDtype_ = ge::DT_UNDEFINED;
    ge::DataType outOfSetDtype_ = ge::DT_UNDEFINED;
};
} // namespace optiling
#endif // SCATTER_ND_COMMON_BASE_TILING_H
