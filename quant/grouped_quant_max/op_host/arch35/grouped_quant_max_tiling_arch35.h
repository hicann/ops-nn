/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GROUPED_QUANT_MAX_REGBASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GROUPED_QUANT_MAX_REGBASE_TILING_H

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"

namespace optiling {

using namespace Ops::NN::Optiling;

struct GroupedQuantMaxCompileInfo {
    uint64_t ubSize = 0;
    int32_t vectorCoreNum = 0;
};

enum class RoundMode {
    MODE_RINT = 0,
    MODE_ROUND = 1,
    MODE_HYBRID = 2,
    MODE_UNDEFINED = -1,
};

class GroupedQuantMaxRegbase {
public:
    explicit GroupedQuantMaxRegbase(gert::TilingContext* context) : context_(context){};
    ge::graphStatus DoTiling();

protected:
    ge::graphStatus GetPlatform();
    ge::graphStatus GetOpParam();
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckAttrs();
    ge::graphStatus CheckShape(const gert::Shape& xShape, const gert::Shape& scaleShape,
                               const gert::Shape& groupListShape, const gert::Shape& yShape,
                               const gert::Shape& amaxShape) const;
    RoundMode GetRoundMode(const std::string& roundMode);
    void MergeInputShape(const gert::Shape& input);
    ge::graphStatus CalcTiling();
    void CalcTilingKey();
    ge::graphStatus CalcUbFactor();
    ge::graphStatus WriteTilingData();

private:
    gert::TilingContext* context_ = nullptr;

    int64_t coreNum_{0};
    uint64_t ubSize_{0};
    int64_t reserveUb_{2048};

    gert::Shape xInputShape_;
    ge::DataType xDtype_{ge::DT_UNDEFINED};
    ge::DataType scaleDtype_{ge::DT_UNDEFINED};
    ge::DataType groupListDtype_{ge::DT_UNDEFINED};
    ge::DataType yDtype_{ge::DT_UNDEFINED};
    ge::DataType amaxDtype_{ge::DT_UNDEFINED};
    RoundMode roundMode_ = RoundMode::MODE_UNDEFINED;
    int32_t dstType_ = 0;
    int64_t actualCoreNum_{0};
    int64_t blockTailCoreNum_{0};
    int64_t blockFactor_{-1};
    int64_t blockTailFactor_{-1};
    int64_t ubFactor_{-1};
    int64_t dim1_{0};
    int64_t numGroups_{0};
    uint64_t tilingKey_{0};
    std::string errorMsg_ = "";
    int64_t dtypeSizeX_{0};
    int64_t dtypeSizeY_{0};
};

} // namespace optiling
#endif