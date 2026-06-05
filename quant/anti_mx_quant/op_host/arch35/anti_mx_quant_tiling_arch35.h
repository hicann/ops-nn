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
 * \file anti_mx_quant_tiling_arch35.h
 * \brief Tiling parameter and class declarations for AntiMxQuant (arch35).
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ANTI_MX_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ANTI_MX_QUANT_H

#include <cstdint>
#include <string>
#include <set>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tilingdata.h"

namespace optiling {
using namespace Ops::NN::Optiling;

struct AntiMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct AntiMxQuantTilingParam {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
    uint32_t vfLen{0};
    int64_t axis{0};
    int64_t dstType{0};
    int64_t blockSize{0};
    bool isTailAxis{false};
    int64_t preAxisSize{1};
    int64_t axisSize{1};
    int64_t postAxisSize{1};
    int64_t usedCoreNum{0};
    int64_t tilingKey{0};
    // Tail axis tiling data
    int64_t rowBlockLoopNum{0};
    int64_t colBlockLoopNum{0};
    int64_t colTileNum{0};
    int64_t rowTileNum{0};
    int64_t rowNum{1};
    int64_t colNum{1};
    int64_t colNormalBlockNum{0};
    int64_t colTailLen{0};
    int64_t rowNormalBlockNum{0};
    int64_t rowTailLen{0};
    int64_t maxUbBlockNum{0};
};

class AntiMxQuantTailAxisTiling {
public:
    explicit AntiMxQuantTailAxisTiling(gert::TilingContext* context, const AntiMxQuantTilingParam& tilingParam)
        : context_(context), tilingParam_(tilingParam), tilingData_{}
    {}
    ~AntiMxQuantTailAxisTiling()
    {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus SetTilingParams();
    void CalcTilingKey() const;
    void CalcAxisSize(const gert::Shape& xShape);
    ge::graphStatus AutoTiling();
    std::set<int64_t> FindSplitCombo(int64_t usedCoreNum) const;
    ge::graphStatus SetTilingDataForTailAxis();
    void PrintTilingDataForTailAxis();
    int64_t CalcBytesPerBlock() const;

private:
    gert::TilingContext* context_ = nullptr;
    AntiMxQuantTilingParam tilingParam_;
    AntiMxQuantTilingData tilingData_{};
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_ANTI_MX_QUANT_H
