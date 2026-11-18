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
 * \file dynamic_hi_f4_quant_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_HIF4_QUANT_REGBASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_HIF4_QUANT_REGBASE_TILING_H
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/dynamic_hi_f4_quant_tilingdata.h"

namespace optiling {
using namespace Ops::NN::Optiling;

struct DynamicHiF4QuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicHiF4QuantTilingParam {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t axis{0};
    int64_t scaleAlg{0}; // 缩放算法选择：0-默认(阈值为4/2)，1-阈值3.75/1.875
    int64_t ubSize{0};
    int64_t blockSize{0};
    int64_t blockW{0};
    int64_t splitBlockH{0};
    int64_t dim0{1};
    int64_t dimNeg2{1};
    int64_t dimNeg1{1};
    int64_t dimNeg2SplitBlockNum{0};
    int64_t dimNeg1BlockNum{0};
    int64_t dimNeg2Tail{0};
    int64_t dimNeg1Tail{0};
    int64_t groupPerUb{0};
    int64_t totalTaskNum{0};
    int64_t blockPerHeadCore{0};
    int64_t blockPerTailCore{0};
    int64_t headCoreNum{0};
    int64_t dimNeg2IsOdd{0};
    int64_t dimNeg1IsOdd{0};
    int64_t dimNeg1IsPad{0};
    int64_t tilingKey{0};
    int64_t blockCountPerBatch{0};
    int64_t scale1ColCountPerBatch{0};
    int64_t scale2RowCountPerBatch{0};
    int64_t workspaceSize{0};
    int64_t ubFormer{0};   // -1 轴：单 tile 的 group 数
    int64_t groupNum{0};   // -1 轴：全局总 group 数
    int64_t totalTiles{0}; // -1 轴：全局总 tile 数
};

class DynamicHiF4QuantTiling {
public:
    explicit DynamicHiF4QuantTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetAttr();
    ge::graphStatus CheckDtype() const;
    ge::graphStatus CheckShape() const;
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus MergeAxis();
    ge::graphStatus SetTilingParamsNeg1();
    ge::graphStatus SetTilingParamsNeg2();

    void SetTilingKey();
    int64_t HiF4GetUbFormerNeg1();
    void SetTilingDataNeg1();
    void SetTilingDataNeg2();
    void PrintTilingData();
    void SplitCore(int64_t blockSize);

private:
    uint64_t quantAxis_ = 0;
    gert::TilingContext* context_ = nullptr;
    DynamicHiF4QuantTilingData tilingData{};
    DynamicHiF4QuantTilingParam tilingParams{};
};

} // namespace optiling
#endif
