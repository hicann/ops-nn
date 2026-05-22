/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_sliding_window_perblock_basic_api_tiling.cpp
 * \brief
 */

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_perblock_basic_api_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

namespace {
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint32_t CORE_RATIO = 2U;

constexpr uint8_t L1_TWO_BUFFER = 2;
constexpr uint8_t L1_FOUR_BUFFER = 4;

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::PERBLOCK_BASIC_API_ASW;
} // namespace

namespace optiling {

AdaptiveSlidingWindowPerblockBasicAPITiling::AdaptiveSlidingWindowPerblockBasicAPITiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context), tilingData_(tilingDataSelf_)
{
    Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

AdaptiveSlidingWindowPerblockBasicAPITiling::AdaptiveSlidingWindowPerblockBasicAPITiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out)
    : AdaptiveSlidingWindowTiling(context, nullptr), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

void AdaptiveSlidingWindowPerblockBasicAPITiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
    }
}

bool AdaptiveSlidingWindowPerblockBasicAPITiling::IsCapable()
{
    return inputParams_.isPerBlock && inputParams_.bFormat == ge::FORMAT_ND;
}

bool AdaptiveSlidingWindowPerblockBasicAPITiling::CheckCoreNum() const
{
    if (compileInfo_.aivNum != CORE_RATIO * compileInfo_.aicNum) {
        OP_LOGE(
            inputParams_.opName, "For perblock template, aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.",
            compileInfo_.aicNum, compileInfo_.aivNum);
        return false;
    }
    return true;
}

uint64_t AdaptiveSlidingWindowPerblockBasicAPITiling::GetBatchCoreCnt() const
{
    return inputParams_.batchC;
}

const void* AdaptiveSlidingWindowPerblockBasicAPITiling::GetTilingData() const
{
    return &tilingData_;
}

bool AdaptiveSlidingWindowPerblockBasicAPITiling::CalcBasicBlock()
{
    BaseBlockCalculator calculator(inputParams_, compileInfo_, GetBatchCoreCnt());
    if (!calculator.Compute(BaseBlockMode::PERBLOCK)) {
        return false;
    }
    const BaseBlockRes& baseBlockRes = calculator.GetOutput();
    adaptiveWin_.baseM = baseBlockRes.baseM;
    adaptiveWin_.baseN = baseBlockRes.baseN;
    adaptiveWin_.baseK = baseBlockRes.baseK;
    adaptiveWin_.useTailWinLogic = baseBlockRes.useTailWinLogic;
    return true;
}

bool AdaptiveSlidingWindowPerblockBasicAPITiling::CalL1Tiling()
{
    basicTiling_.usedCoreNum = CalUsedCoreNum();
    OP_LOGD(inputParams_.opName, "CoreNum: %u", basicTiling_.usedCoreNum);
    basicTiling_.baseM = adaptiveWin_.baseM;
    basicTiling_.baseN = adaptiveWin_.baseN;
    basicTiling_.baseK = adaptiveWin_.baseK;
    basicTiling_.stepM = 1U;
    basicTiling_.stepN = 1U;
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.singleCoreK = inputParams_.kSize;

    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c =
        ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize) &&
         CheckBiasAndScale(basicTiling_.baseN, DB_SIZE)) ?
            DB_SIZE :
            1U;

    L1TilingDataCalculator l1Calculator(
        inputParams_, compileInfo_, basicTiling_.baseM, basicTiling_.baseN, basicTiling_.baseK);
    if (!l1Calculator.Compute(L1TilingMode::DEFAULT)) {
        return false;
    }
    const L1TilingData& l1TilingData = l1Calculator.GetOutput();
    basicTiling_.depthA1 = static_cast<uint32_t>(l1TilingData.depthKa_);
    basicTiling_.depthB1 = static_cast<uint32_t>(l1TilingData.depthKb_);
    basicTiling_.stepKa = static_cast<uint32_t>(l1TilingData.stepKa_);
    basicTiling_.stepKb = static_cast<uint32_t>(l1TilingData.stepKb_);
    return true;
}

void AdaptiveSlidingWindowPerblockBasicAPITiling::AnalyseFullLoadInfo()
{
    isAFullLoad_ = false;
    isBFullLoad_ = false;
    isABFullLoad_ = false;
}

void AdaptiveSlidingWindowPerblockBasicAPITiling::CalcTailRoundBasicBlockSplit()
{
    CalcTailBasicBlock();
}

ge::graphStatus AdaptiveSlidingWindowPerblockBasicAPITiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = inputParams_.mSize;
    tilingData_.matmulTiling.n = inputParams_.nSize;
    tilingData_.matmulTiling.k = inputParams_.kSize;

    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1UL : 0UL;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);
    CalculateNBufferNum4Perblock();
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowPerblockBasicAPITiling::CalculateNBufferNum4Perblock()
{
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    uint64_t kAL1 = tilingData_.matmulTiling.stepKa * tilingData_.matmulTiling.baseK;
    uint64_t fourBufUsedL1Size =
        GetSizeWithDataType((basicTiling_.baseM + basicTiling_.baseN) * kAL1, inputParams_.aDtype) * L1_FOUR_BUFFER;
    if (tilingData_.matmulTiling.stepKa == tilingData_.matmulTiling.stepKb &&
        fourBufUsedL1Size <= aicoreParams_.l1Size && kAL1 * L1_TWO_BUFFER < inputParams_.kSize) {
        tilingData_.matmulTiling.nBufferNum = L1_FOUR_BUFFER;
    } else {
        tilingData_.matmulTiling.nBufferNum = L1_TWO_BUFFER;
    }
}

void AdaptiveSlidingWindowPerblockBasicAPITiling::SetTilingData()
{
    QuantBatchMatMulV3TilingUtil::SetCommonTilingData(inputParams_, tilingData_);
    tilingData_.params.x1QuantMode = inputParams_.groupSizeM == 1UL ?
                                         static_cast<uint32_t>(optiling::BasicQuantMode::PERGROUP_MODE) :
                                         static_cast<uint32_t>(optiling::BasicQuantMode::PERBLOCK_MODE);
    tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERBLOCK_MODE);
    tilingData_.adaptiveSlidingWin.mTailTile = adaptiveWin_.mTailTile;
    tilingData_.adaptiveSlidingWin.nTailTile = adaptiveWin_.nTailTile;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.mTailMain = static_cast<uint32_t>(adaptiveWin_.mTailMain);
    tilingData_.adaptiveSlidingWin.nTailMain = static_cast<uint32_t>(adaptiveWin_.nTailMain);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulV3, AdaptiveSlidingWindowPerblockBasicAPITiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
