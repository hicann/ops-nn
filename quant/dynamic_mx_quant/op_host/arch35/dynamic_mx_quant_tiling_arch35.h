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
 * \file dynamic_mx_quant_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_H
#include <cstdint>
#include <vector>
#include <string>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/dynamic_mx_quant_tilingdata.h"


namespace optiling {
using namespace Ops::NN::Optiling;

struct DynamicMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicMxQuantTilingParam {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
    uint32_t vfLen{0};
    uint32_t workspaceSize{0};
    int64_t axis{0};
    int64_t roundMode;
    int64_t dstType{0};
    int64_t blockSize{0};
    int64_t scaleAlg{0};
    bool isTailAxis{false};
    bool isPad{false};
    int64_t tailBlockSize{0};
    int64_t blockSizeNumInAxis{0};
    int64_t preAxisSize{1};
    int64_t axisSize{1};
    int64_t quantAxisSize{1};
    int64_t postAxisSize{1};
    int64_t mxScaleSize{0};
    int64_t ubDim{0};
    int64_t ubFactor{0};
    int64_t uo{1};
    int64_t usedCoreNum{0};
    int64_t tailUbFactor{0};
    int64_t blockFactor{0};
    int64_t tailBlockFactor{0};
    int64_t tilingKey{0};
    // not tail axis tiling data
    int64_t mAlignSize{1};
    int64_t nAlignSize{1};
    int64_t nAlignNum{1};
    int64_t mAlignBlockCount{1};
    int64_t nAlignBlockCount{1};
    int64_t mAlignGroupCount{1};
    int64_t quantAxisIsOdd{0};
    int64_t totalGroupNum{0};
    int64_t groupPerCore{0};
    int64_t groupPerTail{0};
    int64_t groupPerUb{0};
    int64_t totalBlockNum{0};
    int64_t blockNumPerTask{0};
    int64_t totalTaskNum{0};
    int64_t rowPerHeadCore{0};
    int64_t rowPerTailCore{0};
    bool needPadPostAxis{false};
    bool isOptimize{false};
    // tail axis tiling data
    bool isBlockSize32{false};
    int64_t colTileNum{0};
    int64_t rowTileNum{0};
    int64_t rowNum{1};
    int64_t colNum{1};
    int64_t colNormalBlockNum{0};
    int64_t colTailLen{0};
    int64_t rowNormalBlockNum{0};
    int64_t rowTailLen{0};
    int64_t maxUbBlockNum{0};
    int64_t rowBlockLoopNum{0};
    int64_t colBlockLoopNum{0};

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

class DynamicMxQuantOptimzieTiling {
public:
    explicit DynamicMxQuantOptimzieTiling(gert::TilingContext* context, const DynamicMxQuantTilingParam& tilingParam)
        : context_(context), tilingParam_(tilingParam)
    {}
    ~DynamicMxQuantOptimzieTiling()
    {}
    ge::graphStatus DoTiling();

private:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->PostTiling->GetTilingKey
    ge::graphStatus SetTilingParam();
    void SetTilingData();
    void SetTilingKey();
    void PrintTilingData();
    void SplitCore();
private:
    gert::TilingContext* context_ = nullptr;
    DynamicMxQuant4OptimizeTilingData tilingData;
    DynamicMxQuantTilingParam tilingParam_;
};

class DynamicMxQuantTailAxisTiling {
public:
    explicit DynamicMxQuantTailAxisTiling(gert::TilingContext* context_, const DynamicMxQuantTilingParam& tilingParam_)
        : context(context_), tilingParam(tilingParam_)
    {}
    ~DynamicMxQuantTailAxisTiling()
    {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus SetTilingParams();
    void CalcTilingKeyForTail(DynamicMxQuantTilingParam& tilingParam, const ge::DataType& outputType);
    void CalcAxisSize(DynamicMxQuantTilingParam& tilingParam, const gert::Shape& xShape);
    void AutoTiling(DynamicMxQuantTilingParam& tilingParam);
    std::set<int64_t> FindSplitCombo(int64_t usedCoreNum);
    void SetTilingDataForTailAxis(DynamicMxQuantTailAxisTilingData& tilingData, const DynamicMxQuantTilingParam& tilingParam);
    ge::graphStatus TailAxisSetTilingData(gert::TilingContext* context, DynamicMxQuantTailAxisTilingData& tilingData);
    void PrintTilingDataForTailAxis(gert::TilingContext* context, const DynamicMxQuantTailAxisTilingData& tilingData);

private:
    gert::TilingContext* context = nullptr;
    DynamicMxQuantTailAxisTilingData tilingData;
    DynamicMxQuantTilingParam tilingParam;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_H