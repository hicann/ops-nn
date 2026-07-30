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
 * \file quant_batch_matmul_v4_weight_quant_mx_swat_tiling.cpp
 * \brief
 */

#include "quant_batch_matmul_v4_weight_quant_mx_swat_tiling.h"

#include <sstream>
#include <string>

#include "error_util.h"
#include "log/log.h"
#include "quant_batch_matmul_v4_weight_quant_mx_swat_tiling_solver.h"

namespace optiling {
using namespace matmul_v4;

namespace {
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t L1_FOUR_BUFFER = 4UL;
constexpr const char* OP_TYPE = "QuantBatchMatmulV4";

struct WeightQuantMxSwatScenario {
    ge::DataType aDtype;
    ge::DataType bDtype;
    ge::DataType x1ScaleDtype;
    ge::DataType x2ScaleDtype;
};

bool MatchWeightQuantMxSwatScenario(const WeightQuantMxSwatScenario& scenario)
{
    if (scenario.aDtype == ge::DT_FLOAT8_E4M3FN &&
        (scenario.bDtype == ge::DT_FLOAT4_E2M1 || scenario.bDtype == ge::DT_FLOAT) &&
        scenario.x1ScaleDtype == ge::DT_FLOAT8_E8M0 && scenario.x2ScaleDtype == ge::DT_FLOAT8_E8M0) {
        return true;
    }
    return false;
}

ge::graphStatus GetWeightQuantMxSwatScenario(const gert::TilingContext* context, WeightQuantMxSwatScenario& scenario)
{
    if (context == nullptr) {
        OP_LOGE(OP_TYPE, "Tiling context is null.");
        return ge::GRAPH_FAILED;
    }

    const auto* x1Desc = context->GetInputDesc(X1_INDEX);
    const auto* x2Desc = context->GetInputDesc(X2_INDEX);
    const auto* x1ScaleDesc = context->GetOptionalInputDesc(X1_SCALE_INDEX);
    const auto* x2ScaleDesc = context->GetOptionalInputDesc(X2_SCALE_INDEX);
    if (unlikely(x1Desc == nullptr || x2Desc == nullptr)) {
        OP_LOGE(context, "Get required x1/x2 desc failed");
        return ge::GRAPH_FAILED;
    }
    if (unlikely(x1ScaleDesc == nullptr || x2ScaleDesc == nullptr)) {
        const bool hasMxScale = (x1ScaleDesc != nullptr && x1ScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0) ||
                                (x2ScaleDesc != nullptr && x2ScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0);
        if (hasMxScale) {
            OP_LOGE(context, "MX x1Scale/x2Scale desc is incomplete");
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(context, "x1Scale/x2Scale desc is missing, skip MX SWAT template");
        return ge::GRAPH_PARAM_INVALID;
    }
    scenario = {x1Desc->GetDataType(), x2Desc->GetDataType(), x1ScaleDesc->GetDataType(), x2ScaleDesc->GetDataType()};
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus QuantBatchMatmulV4WeightQuantMxSwatTiling::GetShapeAttrsInfo()
{
    WeightQuantMxSwatScenario scenario{};
    auto status = GetWeightQuantMxSwatScenario(context_, scenario);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    if (!MatchWeightQuantMxSwatScenario(scenario)) {
        return ge::GRAPH_PARAM_INVALID;
    }
    return QuantBatchMatmulV4TilingBase::GetShapeAttrsInfo();
}

bool QuantBatchMatmulV4WeightQuantMxSwatTiling::IsCapable() { return IsWeightQuantMxSwatScenario(); }

bool QuantBatchMatmulV4WeightQuantMxSwatTiling::IsWeightQuantMxSwatScenario() const
{
    WeightQuantMxSwatScenario scenario{inputParams_.aDtype, inputParams_.bDtype, inputParams_.x1ScaleDtype,
                                       inputParams_.x2ScaleDtype};
    return MatchWeightQuantMxSwatScenario(scenario);
}

ge::graphStatus QuantBatchMatmulV4WeightQuantMxSwatTiling::DoOpTiling()
{
    OP_TILING_CHECK(CheckTilingDataCapacity(&tilingData_, sizeof(tilingData_)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "unable to get pointer of SWAT tiling data"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CustomCheck(), OP_LOGE(inputParams_.opName, "Custom check failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "Check CoreNum failed."), return ge::GRAPH_FAILED);

    WeightQuantMxSwatPlatformParam platform = {aicNum_,
                                               aicoreParams_.ubSize,
                                               aicoreParams_.l1Size,
                                               aicoreParams_.l0aSize,
                                               aicoreParams_.l0bSize,
                                               aicoreParams_.l0cSize};
    WeightQuantMxSwatShapeParam shape = {inputParams_.mSize, inputParams_.nSize, inputParams_.kSize};
    std::string reason;
    WeightQuantMxSwatTilingSolver doubleBufferSolver(DB_SIZE);
    if (!doubleBufferSolver.Solve(platform, shape, inputParams_.groupSize, inputParams_.hasBias,
                                  inputParams_.hasX1Scale, inputParams_.hasX2Scale, inputParams_.weightNz,
                                  inputParams_.cDtype, tilingData_, reason)) {
        OP_LOGD(inputParams_.opName, "2-buffer SWAT tiling skipped: %s", reason.c_str());
        reason.clear();
        WeightQuantMxSwatTilingSolver fourBufferSolver(L1_FOUR_BUFFER, true);
        OP_CHECK_IF(!fourBufferSolver.Solve(platform, shape, inputParams_.groupSize, inputParams_.hasBias,
                                            inputParams_.hasX1Scale, inputParams_.hasX2Scale, inputParams_.weightNz,
                                            inputParams_.cDtype, tilingData_, reason),
                    OP_LOGE(inputParams_.opName, "Unable to get SWAT tiling for mnk[%lu, %lu, %lu]: %s",
                            inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, reason.c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4WeightQuantMxSwatTiling::GetWorkspaceSize()
{
    workspaceSize_ = 0UL;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4WeightQuantMxSwatTiling::PostTiling()
{
    auto status = SerializeTilingData(&tilingData_, sizeof(tilingData_), tilingData_.usedCoreNum);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    PrintSwatTilingData(true);
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4WeightQuantMxSwatTiling::PrintSwatTilingData(bool debugLevel) const
{
    DumpSwatTilingDataToLog(debugLevel);
}

void QuantBatchMatmulV4WeightQuantMxSwatTiling::DumpSwatTilingDataToLog(bool debugLevel) const
{
    std::stringstream ss;
    ss << "m/n/k: " << tilingData_.m << "/" << tilingData_.n << "/" << tilingData_.k
       << " baseM/baseN/baseK: " << tilingData_.baseM << "/" << tilingData_.baseN << "/" << tilingData_.baseK
       << " tileShapeKL1: " << tilingData_.tileShapeKL1 << " tileShapeScaleKL1: " << tilingData_.tileShapeScaleKL1
       << " usedCoreNum: " << tilingData_.usedCoreNum << " cubeNumBlocksM/N: " << tilingData_.cubeNumBlocksM << "/"
       << tilingData_.cubeNumBlocksN << " mTailTile/nTailTile: " << tilingData_.mTailTile << "/"
       << tilingData_.nTailTile << " mBaseTailSplitCnt/nBaseTailSplitCnt: " << tilingData_.mBaseTailSplitCnt << "/"
       << tilingData_.nBaseTailSplitCnt << " mTailMain/nTailMain: " << tilingData_.mTailMain << "/"
       << tilingData_.nTailMain << " nBubSize/kBubSize: " << tilingData_.nBubSize << "/" << tilingData_.kBubSize
       << " groupSize: " << tilingData_.groupSize << " weightNz: " << tilingData_.weightNz
       << " hasBias: " << tilingData_.hasBias << " l1BufferNum: " << tilingData_.l1BufferNum;
    if (debugLevel) {
        OPS_LOG_D(inputParams_.opName, "SWAT tiling data: %s", ss.str().c_str());
    } else {
        OPS_LOG_E(inputParams_.opName, "SWAT tiling data: %s", ss.str().c_str());
    }
}
} // namespace optiling
