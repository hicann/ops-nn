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
 * \file grouped_dynamic_hi_f4_quant_tiling.h
 * \brief
 */

#ifndef OPS_NN_DYNAMIC_HIF4_QUANT_TILING_H
#define OPS_NN_DYNAMIC_HIF4_QUANT_TILING_H
#include <cstdint>
#include <vector>
#include <string>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/grouped_dynamic_hi_f4_quant_tilingdata.h"

namespace optiling {

struct GroupedDynamicHiF4QuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct GroupedDynamicHiF4QuantTilingParam {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t ubSize{0};
    uint32_t vfLen{0};
    uint32_t workspaceSize{0};
    int64_t tilingKey{0};
    int64_t groupNum{0};
    int64_t rowSize{0};
    int64_t colSize{0};
    int64_t blockRowSize{0};
    int64_t tailBlockRowSize{0};
    int64_t blockColSize{0};
    int64_t blockRowNum{0};
    int64_t ubMaxblockNum{0};
    int64_t scaleAlg{0}; // 缩放算法选择：0-默认(阈值为4/2)，1-阈值3.75/1.875
};

class GroupedDynamicHiF4QuantTiling {
public:
    explicit GroupedDynamicHiF4QuantTiling(gert::TilingContext* context) : context_(context) {}
    ~GroupedDynamicHiF4QuantTiling() {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckDtype() const;
    ge::graphStatus CheckShape();
    ge::graphStatus GetAttr();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus SetTilingData();

    void SetTilingKey();
    void PrintTilingData();
    void SplitCore();
    void CalcMaxUbAvailable(const int64_t blockH, const int64_t blockW);

private:
    gert::TilingContext* context_ = nullptr;
    GroupedDynamicHiF4QuantTilingData tilingData_;
    GroupedDynamicHiF4QuantTilingParam tilingParams_;
};
} // namespace optiling
#endif // OPS_NN_DYNAMIC_HIF4_QUANT_TILING_H
