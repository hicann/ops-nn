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
 * \file adaptive_sliding_window_basic_api_tiling.cc
 * \brief
 */

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "common/inc/error_util.h"
#include "adaptive_sliding_window_basic_api_tiling.h"

using Ops::NN::MathUtil;
using namespace QuantBatchMatmulV3Arch35TilingKey;

namespace {
constexpr uint64_t AFULLLOAD_SINGLE_CORE_A_SCALER = 2UL;
constexpr uint64_t WINDOW_LEN = 4UL;

constexpr uint32_t SCALER_FACTOR_MAX = 127;
constexpr uint32_t SCALER_FACTOR_MIN = 1;
constexpr uint32_t SCALER_FACTOR_DEFAULT = 1;

constexpr uint8_t L1_TWO_BUFFER = 2;
constexpr uint8_t L1_FOUR_BUFFER = 4;

constexpr uint32_t MX_GROUP_SIZE = 32;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;

constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32UL;
}

namespace optiling {

AdaptiveSlidingWindowBasicAPITiling::AdaptiveSlidingWindowBasicAPITiling(gert::TilingContext *context)
    : AdaptiveSlidingWindowTiling(context), tilingData_(tilingDataSelf_)
{
    Reset();
}

AdaptiveSlidingWindowBasicAPITiling::AdaptiveSlidingWindowBasicAPITiling(
    gert::TilingContext *context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData *out)
    : AdaptiveSlidingWindowTiling(context, nullptr), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
}

void AdaptiveSlidingWindowBasicAPITiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

ge::graphStatus AdaptiveSlidingWindowBasicAPITiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

bool AdaptiveSlidingWindowBasicAPITiling::IsCapable()
{
    bool isMxfp8 = (inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2) &&
                   (inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype == ge::DT_FLOAT8_E5M2) &&
                   inputParams_.isMxPerGroup;
    bool isCubePerTensor =
        inputParams_.isPerTensor &&
        ((inputParams_.aDtype == ge::DT_INT8 && inputParams_.biasDtype == ge::DT_INT32 && !inputParams_.isPertoken) ||
         ((inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2 ||
           inputParams_.aDtype == ge::DT_HIFLOAT8) &&
          inputParams_.scaleDtype == ge::DT_UINT64));
    bool isCubePerChannel =
        inputParams_.isPerChannel && (inputParams_.scaleDtype == ge::DT_UINT64 ||
                                      inputParams_.scaleDtype == ge::DT_INT64 || inputParams_.cDtype == ge::DT_INT32);
    return (((inputParams_.isDoubleScale && !inputParams_.isPerChannel) || isCubePerTensor || isCubePerChannel ||
             inputParams_.isPerBlock) &&
            inputParams_.bFormat == ge::FORMAT_ND) ||
           isMxfp8;
}

uint64_t AdaptiveSlidingWindowBasicAPITiling::GetBatchCoreCnt() const
{
    return inputParams_.batchC;
}

ge::graphStatus AdaptiveSlidingWindowBasicAPITiling::DoOpTiling()
{
    OP_LOGD(inputParams_.opName, "DoOpTiling of adaptive sliding window basic api tiling strategy.");
    if (!AnalyseSlidingWinInfo()) {
        OP_LOGE(inputParams_.opName, "DoOpTiling fail");
        return ge::GRAPH_FAILED;
    }
    AdaptiveSlidingWindowTiling::LoadBalanceDataReset();
    if (!AdaptiveSlidingWindowTiling::OptimizeEdgeBasicBlock()) {
        OP_LOGE(inputParams_.opName, "OptimizeEdgeBasicBlock fail");
        return ge::GRAPH_FAILED;
    }
    AdaptiveSlidingWindowTiling::SetBf16Compat();
    AdaptiveSlidingWindowTiling::CalL1Tiling();
    if (inputParams_.isPertoken) {
        AdaptiveSlidingWindowTiling::CalcUbTiling();
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveSlidingWindowBasicAPITiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = inputParams_.mSize;
    tilingData_.matmulTiling.n = inputParams_.nSize;
    tilingData_.matmulTiling.k = inputParams_.kSize;

    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1UL : 0UL;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);

    if (inputParams_.isMxPerGroup) {
        tilingData_.matmulTiling.scaleKL1 =
            std::min(basicTiling_.scaleFactorA * basicTiling_.stepKa * basicTiling_.baseK,
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
    } else if (inputParams_.isPerBlock) {
        CalculateNBufferNum4Perblock();
    } else if (inputParams_.isDoubleScale || inputParams_.isPerTensor || inputParams_.isPerChannel) {
        CalculateNBufferNum4Cube();
    }
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowBasicAPITiling::CalculateNBufferNum4Perblock()
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

void AdaptiveSlidingWindowBasicAPITiling::CalculateNBufferNum4Cube()
{
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    uint64_t kL1 = 0;
    if (isAFullLoad_) {
        kL1 = basicTiling_.stepKb * tilingData_.matmulTiling.baseK;
    } else {
        uint64_t stepK = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
        kL1 = stepK * tilingData_.matmulTiling.baseK;
    }
    uint64_t usedL1Size =
        GetSizeWithDataType(basicTiling_.baseN * kL1, inputParams_.bDtype) * L1_FOUR_BUFFER;
    if (inputParams_.isPerChannel) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.scaleDtype) * L1_TWO_BUFFER;
    }
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.biasDtype) * L1_TWO_BUFFER;
    }
    if (isAFullLoad_) {
        uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, !inputParams_.transA ? CUBE_REDUCE_BLOCK : CUBE_BLOCK);
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kAligned, inputParams_.aDtype);
    } else {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kL1, inputParams_.aDtype) *
                    L1_FOUR_BUFFER;
    }
    tilingData_.matmulTiling.nBufferNum = usedL1Size < aicoreParams_.l1Size ? L1_FOUR_BUFFER : L1_TWO_BUFFER;
    if (tilingData_.matmulTiling.nBufferNum == L1_FOUR_BUFFER) {
        tilingData_.matmulTiling.stepKa = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
        tilingData_.matmulTiling.stepKb = tilingData_.matmulTiling.stepKa;
    }
}

void AdaptiveSlidingWindowBasicAPITiling::CalculateNBufferNum4MX()
{
    tilingData_.matmulTiling.stepKa = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
    tilingData_.matmulTiling.stepKb = tilingData_.matmulTiling.stepKa;
    uint64_t kL1 = tilingData_.matmulTiling.stepKa * tilingData_.matmulTiling.baseK;
    uint64_t usedL1Size = GetSizeWithDataType(basicTiling_.baseN * kL1, inputParams_.bDtype) * L1_FOUR_BUFFER;
    usedL1Size += GetSizeWithDataType(
                    basicTiling_.baseN * ops::CeilDiv(tilingData_.matmulTiling.scaleKL1, MX_GROUP_SIZE),
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
                        basicTiling_.baseM * ops::CeilDiv(tilingData_.matmulTiling.scaleKL1, MX_GROUP_SIZE),
                        inputParams_.perTokenScaleDtype) *
                    L1_TWO_BUFFER;
    }
    tilingData_.matmulTiling.nBufferNum = usedL1Size < aicoreParams_.l1Size ? L1_FOUR_BUFFER : L1_TWO_BUFFER;
}

uint64_t AdaptiveSlidingWindowBasicAPITiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB), GetBiasMode(),
        GetKernelType());
}

ge::graphStatus AdaptiveSlidingWindowBasicAPITiling::PostTiling()
{
    if (AdaptiveSlidingWindowTiling::PostTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(inputParams_.opName, "Call AdaptiveSlidingWindowTiling PostTiling failed!");
        return ge::GRAPH_FAILED;
    }
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveSlidingWindowBasicAPITiling::AnalyseSlidingWinInfo()
{
    if (!AdaptiveSlidingWindowTiling::CalcBasicBlock()) {
        OP_LOGE(inputParams_.opName, "inappropriate basicBlock");
        return false;
    }
    adaptiveWin_.mBlockCnt = ops::CeilDiv(inputParams_.mSize, adaptiveWin_.baseM);
    adaptiveWin_.nBlockCnt = ops::CeilDiv(inputParams_.nSize, adaptiveWin_.baseN);
    adaptiveWin_.totalBlockCnt = GetBatchCoreCnt() * adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt;
    adaptiveWin_.mTail = inputParams_.mSize - (adaptiveWin_.mBlockCnt - 1UL) * adaptiveWin_.baseM;
    adaptiveWin_.nTail = inputParams_.nSize - (adaptiveWin_.nBlockCnt - 1UL) * adaptiveWin_.baseN;
    adaptiveWin_.totalWinCnt = ops::CeilDiv(adaptiveWin_.totalBlockCnt, aicoreParams_.aicNum);
    adaptiveWin_.tailWinBlockCnt = (adaptiveWin_.totalBlockCnt) % aicoreParams_.aicNum;
    IsAFullLoad();
    if (adaptiveWin_.useTailWinLogic) {
        if (isAFullLoad_) {
            AdaptiveSlidingWindowTiling::CalcTailBasicBlockAfullLoad();
        } else {
            AdaptiveSlidingWindowTiling::CalcTailBasicBlock();
        }
    }
    return true;
}

void AdaptiveSlidingWindowBasicAPITiling::IsAFullLoad()
{
    if (inputParams_.isPerBlock) {
        isAFullLoad_ = false;
        return;
    }
    uint64_t singleCoreASize =
        GetSizeWithDataType(static_cast<uint64_t>(adaptiveWin_.baseM) * inputParams_.kSize, inputParams_.aDtype);
    isAFullLoad_ = singleCoreASize <= aicoreParams_.l1Size / AFULLLOAD_SINGLE_CORE_A_SCALER &&
                   adaptiveWin_.mBlockCnt < WINDOW_LEN && aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0 &&
                   adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
}

void AdaptiveSlidingWindowBasicAPITiling::SetTilingData()
{
    tilingData_.params.batchA = inputParams_.batchA;
    tilingData_.params.batchB = inputParams_.batchB;
    tilingData_.params.batchC = inputParams_.batchC;
    tilingData_.params.batchA1 = inputParams_.batchA1;
    tilingData_.params.batchA2 = inputParams_.batchA2;
    tilingData_.params.batchA3 = inputParams_.batchA3;
    tilingData_.params.batchA4 = inputParams_.batchA4;
    tilingData_.params.batchB1 = inputParams_.batchB1;
    tilingData_.params.batchB2 = inputParams_.batchB2;
    tilingData_.params.batchB3 = inputParams_.batchB3;
    tilingData_.params.batchB4 = inputParams_.batchB4;
    tilingData_.params.batchC1 = inputParams_.batchC1;
    tilingData_.params.batchC2 = inputParams_.batchC2;
    tilingData_.params.batchC3 = inputParams_.batchC3;
    tilingData_.params.batchC4 = inputParams_.batchC4;
    tilingData_.params.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    if (inputParams_.isMxPerGroup) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
    } else if (inputParams_.isPerBlock) {
        tilingData_.params.x1QuantMode = inputParams_.groupSizeM == 1UL ?
                                             static_cast<uint32_t>(optiling::BasicQuantMode::PERGROUP_MODE) :
                                             static_cast<uint32_t>(optiling::BasicQuantMode::PERBLOCK_MODE);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERBLOCK_MODE);
    } else if (inputParams_.isDoubleScale) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerTensor) {
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerChannel) {
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERCHANNEL_MODE);
    }
    tilingData_.params.biasThreeDim = static_cast<uint32_t>(inputParams_.batchBias > 1U);
    tilingData_.adaptiveSlidingWin.mTailTile = adaptiveWin_.mTailTile;
    tilingData_.adaptiveSlidingWin.nTailTile = adaptiveWin_.nTailTile;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.mTailMain = static_cast<uint32_t>(adaptiveWin_.mTailMain);
    tilingData_.adaptiveSlidingWin.nTailMain = static_cast<uint32_t>(adaptiveWin_.nTailMain);
    tilingData_.params.groupSizeM = static_cast<uint32_t>(inputParams_.groupSizeM);
    tilingData_.params.groupSizeN = static_cast<uint32_t>(inputParams_.groupSizeN);
    tilingData_.params.groupSizeK = static_cast<uint32_t>(inputParams_.groupSizeK);
}
}  // namespace optiling