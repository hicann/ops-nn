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
 * \file dynamic_mx_quant_with_dual_axis_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/arch35/dynamic_mx_quant_with_dual_axis_tilingdata.h"

namespace optiling {
struct DynamicMxQuantWithDualAxisCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicMxQuantWithDualAxisTilingParam {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t ubSize{0};
    int64_t roundMode;
    int64_t dstType{0};
    int64_t scaleAlg{0};
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
    float dstTypeMax{0.0};
    float invDstTypeMax{0.0};
};

enum class RoundModeList
{
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

class DynamicMxQuantWithDualAxisTiling {
public:
    explicit DynamicMxQuantWithDualAxisTiling(gert::TilingContext* context) : context_(context)
    {}
    ~DynamicMxQuantWithDualAxisTiling()
    {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetAttr();
    ge::graphStatus CheckDtype() const;
    ge::graphStatus CheckShape() const;
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus MergeAxis();
    ge::graphStatus SetTilingParams();

    void SetTilingKey();
    void SetTilingData();
    void PrintTilingData();
    void SplitCore(int64_t blockW, int64_t blockSize);

    RoundModeList GetRoundMode(const std::string& roundMode);

private:
    uint64_t roundMode_ = 0;
    uint64_t scaleAlg_ = 0;
    uint64_t mode_ = 0;
    gert::TilingContext* context_ = nullptr;
    DynamicMxQuantWithDualAxisTilingData tilingData;
    DynamicMxQuantWithDualAxisTilingParam tilingParams;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H