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
 * \file adaptive_sliding_window_mx_basic_api_tiling.cpp
 * \brief
 */

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_mx_basic_api_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

namespace {
constexpr uint64_t AFULLLOAD_SINGLE_CORE_A_SCALER = 2UL;
constexpr uint64_t WINDOW_LEN = 4UL;

constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;

constexpr uint32_t SCALER_FACTOR_MAX = 127;
constexpr uint32_t SCALER_FACTOR_MIN = 1;
constexpr uint32_t SCALER_FACTOR_DEFAULT = 1;

constexpr uint8_t L1_TWO_BUFFER = 2;
constexpr uint8_t L1_FOUR_BUFFER = 4;

constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::MX_BASIC_API_ASW;
} // namespace

namespace optiling {

AdaptiveSlidingWindowMXBasicAPITiling::AdaptiveSlidingWindowMXBasicAPITiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context), tilingData_(tilingDataSelf_)
{
    Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

AdaptiveSlidingWindowMXBasicAPITiling::AdaptiveSlidingWindowMXBasicAPITiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out)
    : AdaptiveSlidingWindowTiling(context, nullptr), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

void AdaptiveSlidingWindowMXBasicAPITiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
    }
    withoutBatchTilingData_ = DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData();
    useWithoutBatchTilingData_ = false;
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

bool AdaptiveSlidingWindowMXBasicAPITiling::IsWithoutBatchTilingData() const
{
    return IsTensorapiCapable() && inputParams_.batchC == 1UL;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::IsCapable()
{
    bool isMxfp8 = (inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2) &&
                   (inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype == ge::DT_FLOAT8_E5M2) &&
                   inputParams_.isMxPerGroup;
    bool isMxfp4 = inputParams_.isMxPerGroup && inputParams_.aDtype == ge::DT_FLOAT4_E2M1 &&
                   inputParams_.bDtype == ge::DT_FLOAT4_E2M1;
    return isMxfp8 || isMxfp4;
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetBatchCoreCnt() const
{
    return inputParams_.batchC;
}

const void* AdaptiveSlidingWindowMXBasicAPITiling::GetTilingData() const
{
    return useWithoutBatchTilingData_ ? static_cast<const void*>(&withoutBatchTilingData_) :
                                        static_cast<const void*>(&tilingData_);
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CalcBasicBlock()
{
    BaseBlockCalculator calculator(inputParams_, compileInfo_, GetBatchCoreCnt());
    if (!calculator.Compute(BaseBlockMode::DEFAULT)) {
        return false;
    }
    const BaseBlockRes& baseBlockRes = calculator.GetOutput();
    adaptiveWin_.baseM = baseBlockRes.baseM;
    adaptiveWin_.baseN = baseBlockRes.baseN;
    adaptiveWin_.baseK = baseBlockRes.baseK;
    adaptiveWin_.useTailWinLogic = baseBlockRes.useTailWinLogic;
    return true;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CalL1Tiling()
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

    L1TilingMode mode = isAFullLoad_ ? L1TilingMode::A_L1_FULL_LOAD : L1TilingMode::DEFAULT;
    L1TilingDataCalculator l1Calculator(
        inputParams_, compileInfo_, basicTiling_.baseM, basicTiling_.baseN, basicTiling_.baseK);
    if (!l1Calculator.Compute(mode)) {
        return false;
    }
    const L1TilingData& l1TilingData = l1Calculator.GetOutput();
    basicTiling_.depthA1 = static_cast<uint32_t>(l1TilingData.depthKa_);
    basicTiling_.depthB1 = static_cast<uint32_t>(l1TilingData.depthKb_);
    basicTiling_.stepKa = static_cast<uint32_t>(l1TilingData.stepKa_);
    basicTiling_.stepKb = static_cast<uint32_t>(l1TilingData.stepKb_);
    basicTiling_.scaleFactorA = static_cast<uint32_t>(l1TilingData.scaleFactorA_);
    basicTiling_.scaleFactorB = static_cast<uint32_t>(l1TilingData.scaleFactorB_);
    return true;
}

ge::graphStatus AdaptiveSlidingWindowMXBasicAPITiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = inputParams_.mSize;
    tilingData_.matmulTiling.n = inputParams_.nSize;
    tilingData_.matmulTiling.k = inputParams_.kSize;

    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1UL : 0UL;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);

    tilingData_.matmulTiling.scaleKL1 = std::min(
        basicTiling_.scaleFactorA * basicTiling_.stepKa * basicTiling_.baseK,
        basicTiling_.scaleFactorB * basicTiling_.stepKb * basicTiling_.baseK);
    CalculateNBufferNum4MX();
    if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
        basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
        tilingData_.matmulTiling.scaleFactorA = basicTiling_.scaleFactorA;
        tilingData_.matmulTiling.scaleFactorB = basicTiling_.scaleFactorB;
    } else {
        tilingData_.matmulTiling.scaleFactorA = SCALER_FACTOR_DEFAULT;
        tilingData_.matmulTiling.scaleFactorB = SCALER_FACTOR_DEFAULT;
    }
    if (useWithoutBatchTilingData_) {
        SetWithoutBatchTilingData();
    }
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowMXBasicAPITiling::CalculateNBufferNum4MX()
{
    tilingData_.matmulTiling.stepKa = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
    tilingData_.matmulTiling.stepKb = tilingData_.matmulTiling.stepKa;
    uint64_t kL1 = tilingData_.matmulTiling.stepKa * tilingData_.matmulTiling.baseK;
    uint64_t usedL1Size = GetSizeWithDataType(basicTiling_.baseN * kL1, inputParams_.bDtype) * L1_FOUR_BUFFER;
    usedL1Size +=
        GetSizeWithDataType(
            basicTiling_.baseN * ops::CeilDiv(static_cast<uint64_t>(tilingData_.matmulTiling.scaleKL1), MX_GROUP_SIZE),
            inputParams_.scaleDtype) *
        L1_TWO_BUFFER;
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.biasDtype) * L1_TWO_BUFFER;
    }
    if (isAFullLoad_) {
        uint64_t scaleK = ops::CeilDiv(inputParams_.kSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, MXFP_DIVISOR_SIZE);
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kAligned, inputParams_.aDtype) +
                      GetSizeWithDataType(basicTiling_.baseM * scaleK, inputParams_.perTokenScaleDtype);
    } else {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kL1, inputParams_.aDtype) * L1_FOUR_BUFFER;
        usedL1Size += GetSizeWithDataType(
                          basicTiling_.baseM *
                              ops::CeilDiv(static_cast<uint64_t>(tilingData_.matmulTiling.scaleKL1), MX_GROUP_SIZE),
                          inputParams_.perTokenScaleDtype) *
                      L1_TWO_BUFFER;
    }
    tilingData_.matmulTiling.nBufferNum = usedL1Size < aicoreParams_.l1Size ? L1_FOUR_BUFFER : L1_TWO_BUFFER;
}

void AdaptiveSlidingWindowMXBasicAPITiling::UpdateAFullLoadStatus()
{
    uint64_t realBaseMSize = adaptiveWin_.mBaseTailSplitCnt == 1UL ? adaptiveWin_.baseM : adaptiveWin_.mTailMain;
    uint64_t singleCoreASize = GetSizeWithDataType(realBaseMSize * inputParams_.kSize, inputParams_.aDtype);
    isAFullLoad_ = singleCoreASize <= aicoreParams_.l1Size / AFULLLOAD_SINGLE_CORE_A_SCALER &&
                   adaptiveWin_.mBlockCnt < WINDOW_LEN && aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0 &&
                   adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
    if (isAFullLoad_ && adaptiveWin_.baseM != realBaseMSize) {
        adaptiveWin_.baseM = realBaseMSize;
        adaptiveWin_.mBaseTailSplitCnt = 1UL;
        adaptiveWin_.mTailMain = 0UL;
    }
}

void AdaptiveSlidingWindowMXBasicAPITiling::AnalyseFullLoadInfo()
{
    isABFullLoad_ = false;
    isBFullLoad_ = false;
    UpdateAFullLoadStatus();
}

void AdaptiveSlidingWindowMXBasicAPITiling::CalcTailRoundBasicBlockSplit()
{
    if (!adaptiveWin_.useTailWinLogic) {
        return;
    }
    if (isAFullLoad_) {
        CalcTailBasicBlockAfullLoad();
    } else {
        CalcTailBasicBlock();
    }
}

void AdaptiveSlidingWindowMXBasicAPITiling::SetTilingData()
{
    useWithoutBatchTilingData_ = IsWithoutBatchTilingData();
    tilingDataSize_ = useWithoutBatchTilingData_ ?
        sizeof(DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData) :
        sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);

    QuantBatchMatMulV3TilingUtil::SetCommonTilingData(inputParams_, tilingData_);
    tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
    tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
    tilingData_.adaptiveSlidingWin.mTailTile = adaptiveWin_.mTailTile;
    tilingData_.adaptiveSlidingWin.nTailTile = adaptiveWin_.nTailTile;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.mTailMain = static_cast<uint32_t>(adaptiveWin_.mTailMain);
    tilingData_.adaptiveSlidingWin.nTailMain = static_cast<uint32_t>(adaptiveWin_.nTailMain);

    if (useWithoutBatchTilingData_) {
        SetWithoutBatchTilingData();
    }
}

void AdaptiveSlidingWindowMXBasicAPITiling::SetWithoutBatchTilingData()
{
    withoutBatchTilingData_.m = static_cast<uint32_t>(inputParams_.mSize);
    withoutBatchTilingData_.n = static_cast<uint32_t>(inputParams_.nSize);
    withoutBatchTilingData_.k = static_cast<uint32_t>(inputParams_.kSize);
    withoutBatchTilingData_.scaleKL1 = tilingData_.matmulTiling.scaleKL1;
    withoutBatchTilingData_.baseM = static_cast<uint16_t>(basicTiling_.baseM);
    withoutBatchTilingData_.baseN = static_cast<uint16_t>(basicTiling_.baseN);
    withoutBatchTilingData_.baseK = static_cast<uint16_t>(basicTiling_.baseK);
    withoutBatchTilingData_.stepKa = tilingData_.matmulTiling.stepKa;
    withoutBatchTilingData_.stepKb = tilingData_.matmulTiling.stepKb;
    withoutBatchTilingData_.groupSizeM = static_cast<uint16_t>(inputParams_.groupSizeM);
    withoutBatchTilingData_.groupSizeN = static_cast<uint16_t>(inputParams_.groupSizeN);
    withoutBatchTilingData_.groupSizeK = static_cast<uint16_t>(inputParams_.groupSizeK);
    withoutBatchTilingData_.mTailTile = static_cast<uint16_t>(adaptiveWin_.mTailTile);
    withoutBatchTilingData_.nTailTile = static_cast<uint16_t>(adaptiveWin_.nTailTile);
    withoutBatchTilingData_.mBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.mBaseTailSplitCnt);
    withoutBatchTilingData_.nBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.nBaseTailSplitCnt);
    withoutBatchTilingData_.mTailMain = static_cast<uint16_t>(adaptiveWin_.mTailMain);
    withoutBatchTilingData_.nTailMain = static_cast<uint16_t>(adaptiveWin_.nTailMain);
    withoutBatchTilingData_.x1QuantMode = static_cast<uint8_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
    withoutBatchTilingData_.x2QuantMode = static_cast<uint8_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
    withoutBatchTilingData_.isBias = tilingData_.matmulTiling.isBias;
    withoutBatchTilingData_.biasDtype = static_cast<uint8_t>(inputParams_.biasDtype);
    withoutBatchTilingData_.nBufferNum = tilingData_.matmulTiling.nBufferNum;
    withoutBatchTilingData_.dbL0C = tilingData_.matmulTiling.dbL0C;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulV3, AdaptiveSlidingWindowMXBasicAPITiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
