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
 * \file adaptive_sliding_window_cube_tiling.cpp
 * \brief
 */
#include <vector>

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_cube_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

namespace {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint32_t NUM_HALF = 2;
constexpr uint64_t MIN_CARRY_DATA_SIZE_32K = 32 * 1024UL;
constexpr uint64_t FULL_LOAD_DATA_SIZE_64K = 64 * 1024UL;
constexpr uint64_t CACHE_LINE_512B = 512UL;
constexpr uint32_t MAX_STEPK_With_BL1_FULL = 8U;

const std::vector<int32_t> supportedNpuArch = {
    static_cast<int32_t>(NpuArch::DAV_3510), static_cast<int32_t>(NpuArch::DAV_RESV)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::CUBE_ASW;
} // namespace

namespace optiling {

AdaptiveSlidingWindowCubeTiling::AdaptiveSlidingWindowCubeTiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context)
{
    Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
}

AdaptiveSlidingWindowCubeTiling::AdaptiveSlidingWindowCubeTiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3TilingDataParams* out)
    : AdaptiveSlidingWindowTiling(context, out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
}

bool AdaptiveSlidingWindowCubeTiling::IsCapable()
{
    bool isCubePerTensor =
        inputParams_.isPerTensor &&
        ((inputParams_.aDtype == ge::DT_INT8 && inputParams_.biasDtype == ge::DT_INT32 && !inputParams_.isPertoken) ||
         ((inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2 ||
           inputParams_.aDtype == ge::DT_HIFLOAT8) &&
          inputParams_.scaleDtype == ge::DT_UINT64));
    bool isCubePerChannel =
        inputParams_.isPerChannel && (inputParams_.scaleDtype == ge::DT_UINT64 ||
                                      inputParams_.scaleDtype == ge::DT_INT64 || inputParams_.cDtype == ge::DT_INT32);
    bool isFp8OrHif8TTBiasMix = IsFp8OrHif8TTFloatBiasMix();
    return compileInfo_.supportMmadS8S4 ||
           (!isFp8OrHif8TTBiasMix &&
            ((inputParams_.isDoubleScale && !inputParams_.isPerChannel) || isCubePerTensor || isCubePerChannel) &&
            inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ);
}

bool AdaptiveSlidingWindowCubeTiling::CalL1Tiling()
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
    L1TilingMode mode = L1TilingMode::DEFAULT;
    if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        if (isABFullLoad_) {
            mode = L1TilingMode::PASS_OPTIMIZED_AB_FULL_LOAD;
        } else if (isAFullLoad_) {
            mode = L1TilingMode::PASS_OPTIMIZED_A_FULL_LOAD;
        } else if (isBFullLoad_) {
            mode = L1TilingMode::PASS_OPTIMIZED_B_FULL_LOAD;
        } else {
            mode = L1TilingMode::PASS_OPTIMIZED;
        }
    } else if (isAFullLoad_) {
        mode = L1TilingMode::A_L1_FULL_LOAD;
    }

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

void AdaptiveSlidingWindowCubeTiling::UpdateAFullLoadStatus()
{
    if (inputParams_.batchA != 1UL || IsMxKOdd() || IsMxBackwardTrans() || isTilingOut_ || inputParams_.isPerBlock ||
        inputParams_.cDtype == ge::DT_INT32) {
        isAFullLoad_ = false;
        return;
    }
    uint64_t realBaseMSize = compileInfo_.supportMmadS8S4 || adaptiveWin_.mBaseTailSplitCnt == 1UL ?
                                 adaptiveWin_.baseM :
                                 adaptiveWin_.mTailMain;
    uint64_t singleCoreASizeWithFullLoad =
        realBaseMSize *
        (inputParams_.transA ?
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.aDtype) :
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype), CUBE_REDUCE_BLOCK));
    constexpr uint64_t A_L1_LOAD_THRESHOLD = 256 * 1024UL;
    bool blockCntCmp = (adaptiveWin_.mBlockCnt <= adaptiveWin_.nBlockCnt);
    if (compileInfo_.supportMmadS8S4) {
        blockCntCmp = true;
    }
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

void AdaptiveSlidingWindowCubeTiling::UpdateBFullLoadStatus()
{
    if (!compileInfo_.supportMmadS8S4 || inputParams_.batchA != 1 || inputParams_.batchB != 1 ||
        inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8 ||
        !(inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16) ||
        !(inputParams_.isPerTensor || inputParams_.isPerChannel) || isAFullLoad_ || isTilingOut_ ||
        inputParams_.cDtype == ge::DT_INT32) {
        isBFullLoad_ = false;
        return;
    }
    uint64_t singleCoreBSizeWithFullLoad =
        adaptiveWin_.baseN *
        (inputParams_.transB ?
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype), CUBE_REDUCE_BLOCK) :
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.bDtype));

    isBFullLoad_ =
        singleCoreBSizeWithFullLoad <= aicoreParams_.l1Size / NUM_HALF &&
        adaptiveWin_.mBlockCnt > adaptiveWin_.nBlockCnt &&
        (((adaptiveWin_.nBlockCnt == 1UL || adaptiveWin_.nBlockCnt == 2UL || adaptiveWin_.nBlockCnt == 4UL) &&
          aicoreParams_.aicNum >= adaptiveWin_.nBlockCnt && aicoreParams_.aicNum % adaptiveWin_.nBlockCnt == 0) ||
         adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum);
}

void AdaptiveSlidingWindowCubeTiling::UpdateABFullLoadStatus()
{
    if (inputParams_.batchA != 1 || inputParams_.batchB != 1 || isTilingOut_ || inputParams_.isPerBlock ||
        !compileInfo_.supportMmadS8S4 || inputParams_.isPertoken || inputParams_.aDtype != ge::DT_INT8 ||
        !(inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16)) {
        isABFullLoad_ = false;
        return;
    }
    uint64_t singleCoreASizeWithFullLoad =
        adaptiveWin_.baseM *
        (inputParams_.transA ?
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.aDtype) :
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype), CUBE_REDUCE_BLOCK));
    uint64_t singleCoreBSizeWithFullLoad =
        adaptiveWin_.baseN *
        (inputParams_.transB ?
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype), CUBE_REDUCE_BLOCK) :
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.bDtype));

    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t singleCoreBiasSize = inputParams_.hasBias ? adaptiveWin_.baseN * biasDtypeSize : 0UL;
    uint64_t singleCoreScaleSize = inputParams_.isPerChannel ? adaptiveWin_.baseN * scaleDtypeSize : 0UL;
    constexpr uint64_t AB_L1_SINGLE_LOAD_THRESHOLD = 64 * 1024UL;
    constexpr uint64_t AB_L1_BOTH_LOAD_THRESHOLD = 272 * 1024UL;
    bool hasEnoughL1 =
        aicoreParams_.l1Size >= singleCoreASizeWithFullLoad &&
        aicoreParams_.l1Size - singleCoreASizeWithFullLoad >= singleCoreBSizeWithFullLoad &&
        aicoreParams_.l1Size - singleCoreASizeWithFullLoad - singleCoreBSizeWithFullLoad >= singleCoreBiasSize &&
        aicoreParams_.l1Size - singleCoreASizeWithFullLoad - singleCoreBSizeWithFullLoad - singleCoreBiasSize >=
            singleCoreScaleSize;
    bool isABSizeInRange = AB_L1_BOTH_LOAD_THRESHOLD >= singleCoreASizeWithFullLoad &&
                           AB_L1_BOTH_LOAD_THRESHOLD - singleCoreASizeWithFullLoad >= singleCoreBSizeWithFullLoad;

    isABFullLoad_ = hasEnoughL1 && adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum &&
                    (singleCoreASizeWithFullLoad <= AB_L1_SINGLE_LOAD_THRESHOLD ||
                     singleCoreBSizeWithFullLoad <= AB_L1_SINGLE_LOAD_THRESHOLD) &&
                    isABSizeInRange;
}

void AdaptiveSlidingWindowCubeTiling::AnalyseFullLoadInfo()
{
    UpdateABFullLoadStatus();
    if (isABFullLoad_) {
        adaptiveWin_.useTailWinLogic = false;
        return;
    }
    UpdateAFullLoadStatus();
    UpdateBFullLoadStatus();
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulV3, AdaptiveSlidingWindowCubeTiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
