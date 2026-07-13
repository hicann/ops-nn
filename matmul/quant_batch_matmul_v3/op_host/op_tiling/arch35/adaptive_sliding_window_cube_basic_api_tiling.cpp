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
 * \file adaptive_sliding_window_cube_basic_api_tiling.cpp
 * \brief
 */

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_cube_basic_api_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

namespace {
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510),
                                               static_cast<int32_t>(NpuArch::DAV_RESV)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::CUBE_BASIC_API_ASW;
} // namespace

namespace optiling {

AdaptiveSlidingWindowCubeBasicAPITiling::AdaptiveSlidingWindowCubeBasicAPITiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context), tilingData_(tilingDataSelf_)
{
    Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

AdaptiveSlidingWindowCubeBasicAPITiling::AdaptiveSlidingWindowCubeBasicAPITiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out)
    : AdaptiveSlidingWindowTiling(context, nullptr), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

void AdaptiveSlidingWindowCubeBasicAPITiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
    }
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::IsCapable()
{
    // Mandatory input descs have been validated by GetShapeAttrsInfo before IsCapable.
    if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        return inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
               inputParams_.bFormat == ge::FORMAT_ND;
    }
    isSupportS4S4_ = inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4 &&
                     !compileInfo_.supportMmadS8S4;
    const auto originADtype = inputParams_.aDtype;
    const auto originBDtype = inputParams_.bDtype;
    if (isSupportS4S4_) {
        inputParams_.aDtype = ge::DT_INT8;
        inputParams_.bDtype = ge::DT_INT8;
    }
    bool isCubeBasicApiCapable = IsCubeBasicApiCapable(inputParams_);
    bool isFp8OrHif8TTBiasMix = IsFp8OrHif8TTFloatBiasMix(inputParams_);
    bool capable = !isFp8OrHif8TTBiasMix && ((isCubeBasicApiCapable && inputParams_.bFormat == ge::FORMAT_ND) ||
                                             IsWeightNzNonMxCubeBasicApiCapable(inputParams_));
    if (!capable && isSupportS4S4_) {
        inputParams_.aDtype = originADtype;
        inputParams_.bDtype = originBDtype;
    }
    return capable;
}

ge::graphStatus AdaptiveSlidingWindowCubeBasicAPITiling::GetWorkspaceSize()
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

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::GetBatchCoreCnt() const { return inputParams_.batchC; }

const void* AdaptiveSlidingWindowCubeBasicAPITiling::GetTilingData() const { return &tilingData_; }

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::GetApiLevel(NpuArch npuArch) const
{
    switch (npuArch) {
        case NpuArch::DAV_3510: {
            const bool isTensorapiCapable = IsTensorapiCapable();
            if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && !isTensorapiCapable) {
                return static_cast<uint64_t>(QMMApiLevel::HIGH_LEVEL);
            }
            return isTensorapiCapable ? static_cast<uint64_t>(QMMApiLevel::BLAZE_LEVEL) :
                                        static_cast<uint64_t>(QMMApiLevel::BASIC_LEVEL);
        }
        case NpuArch::DAV_RESV:
            if (inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
                inputParams_.bFormat == ge::FORMAT_ND) {
                return static_cast<uint64_t>(QMMApiLevel::BASIC_LEVEL);
            }
            return static_cast<uint64_t>(QMMApiLevel::HIGH_LEVEL);
        default:
            return static_cast<uint64_t>(QMMApiLevel::HIGH_LEVEL);
    }
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::CalL1Tiling()
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
    basicTiling_.dbL0c = ((basicTiling_.baseM * basicTiling_.baseN * qmmv3_tiling_const::DATA_SIZE_L0C *
                               qmmv3_tiling_const::DOUBLE_BUFFER_NUM <=
                           aicoreParams_.l0cSize) &&
                          CheckBiasAndScale(basicTiling_.baseN, qmmv3_tiling_const::DOUBLE_BUFFER_NUM)) ?
                             qmmv3_tiling_const::DOUBLE_BUFFER_NUM :
                             1U;

    L1TilingMode mode = L1TilingMode::DEFAULT;
    if (isAFullLoad_) {
        mode = L1TilingMode::A_L1_FULL_LOAD;
    } else if (isBFullLoad_) {
        mode = L1TilingMode::B_L1_FULL_LOAD;
    } else if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        mode = L1TilingMode::PASS_OPTIMIZED;
    }
    L1TilingDataCalculator l1Calculator(inputParams_, compileInfo_, basicTiling_.baseM, basicTiling_.baseN,
                                        basicTiling_.baseK);
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

ge::graphStatus AdaptiveSlidingWindowCubeBasicAPITiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = inputParams_.mSize;
    tilingData_.matmulTiling.n = inputParams_.nSize;
    tilingData_.matmulTiling.k = inputParams_.kSize;

    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1UL : 0UL;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);
    CalculateNBufferNum4Cube();
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowCubeBasicAPITiling::CalculateNBufferNum4Cube()
{
    tilingData_.matmulTiling.kAL1 = basicTiling_.stepKa * tilingData_.matmulTiling.baseK;
    tilingData_.matmulTiling.kBL1 = basicTiling_.stepKb * tilingData_.matmulTiling.baseK;
    uint64_t kL1 = 0;
    if (isAFullLoad_) {
        kL1 = basicTiling_.stepKb * tilingData_.matmulTiling.baseK;
    } else {
        uint64_t stepK = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
        kL1 = stepK * tilingData_.matmulTiling.baseK;
    }
    uint64_t usedL1Size = 0UL;
    if (isBFullLoad_) {
        uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, inputParams_.transB ?
                                                                   qmmv3_tiling_const::CUBE_REDUCE_BLOCK :
                                                                   qmmv3_tiling_const::CUBE_BLOCK);
        usedL1Size = GetSizeWithDataType(basicTiling_.baseN * kAligned, inputParams_.bDtype);
    } else {
        usedL1Size = GetSizeWithDataType(basicTiling_.baseN * kL1, inputParams_.bDtype) *
                     qmmv3_tiling_const::L1_FOUR_BUFFER;
    }
    if (inputParams_.isPerChannel) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.scaleDtype) *
                      qmmv3_tiling_const::L1_TWO_BUFFER;
    }
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.biasDtype) *
                      qmmv3_tiling_const::L1_TWO_BUFFER;
    }
    if (isAFullLoad_) {
        uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, !inputParams_.transA ?
                                                                   qmmv3_tiling_const::CUBE_REDUCE_BLOCK :
                                                                   qmmv3_tiling_const::CUBE_BLOCK);
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kAligned, inputParams_.aDtype);
    } else {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kL1, inputParams_.aDtype) *
                      qmmv3_tiling_const::L1_FOUR_BUFFER;
    }
    tilingData_.matmulTiling.nBufferNum = usedL1Size <= aicoreParams_.l1Size ? qmmv3_tiling_const::L1_FOUR_BUFFER :
                                                                               qmmv3_tiling_const::L1_TWO_BUFFER;
    if (tilingData_.matmulTiling.nBufferNum == qmmv3_tiling_const::L1_FOUR_BUFFER) {
        tilingData_.matmulTiling.kAL1 = std::min(basicTiling_.stepKa, basicTiling_.stepKb) *
                                        tilingData_.matmulTiling.baseK;
        tilingData_.matmulTiling.kBL1 = tilingData_.matmulTiling.kAL1;
    }
}

void AdaptiveSlidingWindowCubeBasicAPITiling::UpdateAFullLoadStatus()
{
    uint64_t realBaseMSize = adaptiveWin_.mBaseTailSplitCnt == 1UL ? adaptiveWin_.baseM : adaptiveWin_.mTailMain;
    uint64_t singleCoreASize = realBaseMSize *
                               (inputParams_.transA ?
                                    GetSizeWithDataType(
                                        ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::CUBE_BLOCK),
                                        inputParams_.aDtype) :
                                    ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype),
                                                   qmmv3_tiling_const::CUBE_REDUCE_BLOCK));

    isAFullLoad_ = singleCoreASize <= aicoreParams_.l1Size / qmmv3_tiling_const::AFULLLOAD_SINGLE_CORE_A_SCALER &&
                   adaptiveWin_.mBlockCnt < qmmv3_tiling_const::WINDOW_LEN &&
                   aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0 &&
                   adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
    if (isAFullLoad_ && adaptiveWin_.baseM != realBaseMSize) {
        adaptiveWin_.baseM = realBaseMSize;
        adaptiveWin_.mBaseTailSplitCnt = 1UL;
        adaptiveWin_.mTailMain = 0UL;
    }
}

void AdaptiveSlidingWindowCubeBasicAPITiling::UpdateBFullLoadStatus()
{
    if (isAFullLoad_ || compileInfo_.npuArch != NpuArch::DAV_RESV) {
        isBFullLoad_ = false;
        return;
    }
    uint64_t realBaseNSize = adaptiveWin_.nBaseTailSplitCnt == 1UL ? adaptiveWin_.baseN : adaptiveWin_.nTailMain;
    uint64_t singleCoreBSize = realBaseNSize *
                               (inputParams_.transB ?
                                    ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype),
                                                   qmmv3_tiling_const::CUBE_REDUCE_BLOCK) :
                                    GetSizeWithDataType(
                                        ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::CUBE_BLOCK),
                                        inputParams_.bDtype));

    isBFullLoad_ = singleCoreBSize <= aicoreParams_.l1Size / qmmv3_tiling_const::AFULLLOAD_SINGLE_CORE_B_SCALER &&
                   adaptiveWin_.nBlockCnt < qmmv3_tiling_const::WINDOW_LEN &&
                   aicoreParams_.aicNum % adaptiveWin_.nBlockCnt == 0 &&
                   adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
    if (isBFullLoad_ && adaptiveWin_.baseN != realBaseNSize) {
        adaptiveWin_.baseN = realBaseNSize;
        adaptiveWin_.nBaseTailSplitCnt = 1UL;
        adaptiveWin_.nTailMain = 0UL;
    }
}

void AdaptiveSlidingWindowCubeBasicAPITiling::AnalyseFullLoadInfo()
{
    isABFullLoad_ = false;
    isBFullLoad_ = false;
    UpdateAFullLoadStatus();
    UpdateBFullLoadStatus();
}

void AdaptiveSlidingWindowCubeBasicAPITiling::SetTilingData()
{
    QuantBatchMatMulV3TilingUtil::SetCommonTilingData(inputParams_, tilingData_);
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::DEFAULT);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERCHANNEL_MODE);
        if (inputParams_.isDoubleScale) {
            tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
            tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
        } else if (inputParams_.isPerTensor) {
            tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
        }
    } else if (inputParams_.isDoubleScale) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerTensor) {
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerChannel) {
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERCHANNEL_MODE);
    }
    tilingData_.adaptiveSlidingWin.mTailTile = adaptiveWin_.mTailTile;
    tilingData_.adaptiveSlidingWin.nTailTile = adaptiveWin_.nTailTile;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.mTailMain = static_cast<uint32_t>(adaptiveWin_.mTailMain);
    tilingData_.adaptiveSlidingWin.nTailMain = static_cast<uint32_t>(adaptiveWin_.nTailMain);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantBatchMatmulV3, AdaptiveSlidingWindowCubeBasicAPITiling, supportedNpuArch,
                                   TILING_PRIORITY);
} // namespace optiling
