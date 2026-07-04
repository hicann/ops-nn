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
 * \file adaptive_sliding_window_mix_tiling.cpp
 * \brief
 */
#include <vector>

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_mix_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

namespace {

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::MIX_ASW;
} // namespace

namespace optiling {

AdaptiveSlidingWindowMixTiling::AdaptiveSlidingWindowMixTiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context)
{
    Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
}

AdaptiveSlidingWindowMixTiling::AdaptiveSlidingWindowMixTiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3TilingDataParams* out)
    : AdaptiveSlidingWindowTiling(context, out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
}

bool AdaptiveSlidingWindowMixTiling::IsCapable()
{
    // Mandatory input descs have been validated by GetShapeAttrsInfo before IsCapable.
    isSupportS4S4_ =
        inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4 && !compileInfo_.supportMmadS8S4;
    const auto originADtype = inputParams_.aDtype;
    const auto originBDtype = inputParams_.bDtype;
    if (isSupportS4S4_) {
        inputParams_.aDtype = ge::DT_INT8;
        inputParams_.bDtype = ge::DT_INT8;
    }
    SetBf16Compat();
    bool isScaleVecPostProcess = inputParams_.isPerChannel &&
                                 !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64);
    bool isFp8OrHif8TTBiasMix = IsFp8OrHif8TTFloatBiasMix(inputParams_);
    bool capable = (isScaleVecPostProcess || inputParams_.isPertoken || isBf16Mix_ ||
                    isFp8OrHif8TTBiasMix || isSupportS4S4_) &&
                   inputParams_.cDtype != ge::DT_INT32;
    if (!capable && isSupportS4S4_) {
        inputParams_.aDtype = originADtype;
        inputParams_.bDtype = originBDtype;
    }
    return capable;
}

bool AdaptiveSlidingWindowMixTiling::CheckCoreNum() const
{
    if (compileInfo_.aivNum != qmmv3_tiling_const::CORE_RATIO * compileInfo_.aicNum) {
        OP_LOGE(
            inputParams_.opName, "For mix template, aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.",
            compileInfo_.aicNum, compileInfo_.aivNum);
        return false;
    }
    return true;
}

ge::graphStatus AdaptiveSlidingWindowMixTiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    if (isSupportS4S4_) {
        uint64_t aInt8Size = inputParams_.batchC * inputParams_.mSize * inputParams_.kSize;
        uint64_t bInt8Size = inputParams_.kSize * inputParams_.nSize;
        workspaceSize_ += ops::CeilAlign(aInt8Size, qmmv3_tiling_const::BASIC_BLOCK_SIZE_128) +
                          ops::CeilAlign(bInt8Size, qmmv3_tiling_const::BASIC_BLOCK_SIZE_128);
    }
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveSlidingWindowMixTiling::CalcBasicBlock()
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

void AdaptiveSlidingWindowMixTiling::UpdateAFullLoadStatus()
{
    if (inputParams_.batchA != 1UL || IsMxKOdd() || IsMxBackwardTrans() || isTilingOut_ || inputParams_.isPerBlock ||
        inputParams_.cDtype == ge::DT_INT32) {
        isAFullLoad_ = false;
        return;
    }
    uint64_t realBaseMSize = adaptiveWin_.mBaseTailSplitCnt == 1UL ? adaptiveWin_.baseM : adaptiveWin_.mTailMain;
    uint64_t singleCoreASizeWithFullLoad =
        realBaseMSize *
        (inputParams_.transA ?
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::CUBE_BLOCK), inputParams_.aDtype) :
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype), qmmv3_tiling_const::CUBE_REDUCE_BLOCK));
    // Empirical A full-load cap: keep A under 256KB so L1 still has room for B-side data and auxiliary buffers.
    constexpr uint64_t A_L1_LOAD_THRESHOLD = 256 * 1024UL;
    bool blockCntCmp = (adaptiveWin_.mBlockCnt <= adaptiveWin_.nBlockCnt);
    isAFullLoad_ =
        singleCoreASizeWithFullLoad <= A_L1_LOAD_THRESHOLD && blockCntCmp &&
        (((adaptiveWin_.mBlockCnt == 1UL || adaptiveWin_.mBlockCnt == 2UL || adaptiveWin_.mBlockCnt == 4UL) &&
          aicoreParams_.aicNum >= adaptiveWin_.mBlockCnt && aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0) ||
         adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum);
    if (isAFullLoad_ && adaptiveWin_.baseM != realBaseMSize) {
        adaptiveWin_.baseM = realBaseMSize;
        adaptiveWin_.mBaseTailSplitCnt = 1UL;
        adaptiveWin_.mTailMain = 0UL;
    }
}

void AdaptiveSlidingWindowMixTiling::AnalyseFullLoadInfo()
{
    isABFullLoad_ = false;
    isBFullLoad_ = false;
    UpdateAFullLoadStatus();
}

void AdaptiveSlidingWindowMixTiling::CalcTailRoundBasicBlockSplit()
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

bool AdaptiveSlidingWindowMixTiling::CalL1Tiling()
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
        ((basicTiling_.baseM * basicTiling_.baseN * qmmv3_tiling_const::DATA_SIZE_L0C * qmmv3_tiling_const::DOUBLE_BUFFER_NUM <= aicoreParams_.l0cSize) &&
         CheckBiasAndScale(basicTiling_.baseN, qmmv3_tiling_const::DOUBLE_BUFFER_NUM)) ?
            qmmv3_tiling_const::DOUBLE_BUFFER_NUM :
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
    return true;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulV3, AdaptiveSlidingWindowMixTiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
