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
 * \file rms_norm_grad_quant_regbase_big_m_tiling.cpp
 * \brief RmsNormGradQuant regbase tiling file
 */

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "rms_norm_grad_quant_tiling.h"

namespace optiling {

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_SIXTY_THREE = 63;

constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t MAX_CORE_NUM = 64;
constexpr static uint64_t ULONG_BIT_LEN = 64;

constexpr static int64_t MFACTOR_DEFAULT = 64;
constexpr static uint32_t TILING_KEY_BIG_M = 9000;

bool RmsNormGradQuantBigMTiling::IsCapable()
{
    // 合轴后（m, n）按照n轴分核数超过总核数一半时返回
    if (cols_ > static_cast<int64_t>(vlFp32_ * aivCoreNum_ / CONST_TWO)) {
        return false;
    }

    // m<2*n, 或者按m轴分核数小于总核数一半返回
    if (rows_ < cols_ * CONST_TWO || rows_ < static_cast<int64_t>(MFACTOR_DEFAULT * aivCoreNum_ / CONST_TWO)) {
        return false;
    }

    return true;
}

ge::graphStatus RmsNormGradQuantBigMTiling::DoOpTiling()
{
    computeModeDx_ = rms_norm_grad_quant::ComputeModeDx::FULL_LOAD;
    computeModeDgamma_ = rms_norm_grad_quant::ComputeModeDgamma::BIG_M;
    // dgamma 切分
    ge::graphStatus statusGamma = DgammaDoTiling();
    OP_CHECK_IF(statusGamma != ge::GRAPH_SUCCESS, , return statusGamma);

    // dx 切分
    ge::graphStatus statusDx = CalcTilingDataDx();
    OP_CHECK_IF(statusDx != ge::GRAPH_SUCCESS, , return statusDx);

    tilingData_.dxTilingData.usedCoreNumDx = usedCoreNumDx_;
    tilingData_.dxTilingData.cols = cols_;
    tilingData_.dxTilingData.rows = rows_;
    tilingData_.dxTilingData.blockFactorDx = blockFactorDx_;
    tilingData_.dxTilingData.bodyPart = bodyPart_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantBigMTiling::DgammaDoTiling()
{
    OP_TILING_CHECK(DgammaDoTilingStg0() != ge::GRAPH_SUCCESS,
                    OP_LOGI(context_->GetNodeName(), "Big M template dgamma do tiling stage 0 failed."),
                    return ge::GRAPH_PARAM_INVALID);

    OP_TILING_CHECK(DgammaDoTilingStg1() != ge::GRAPH_SUCCESS,
                    OP_LOGI(context_->GetNodeName(), "Big M template dgamma do tiling stage 1 failed."),
                    return ge::GRAPH_PARAM_INVALID);

    OP_LOGD(context_->GetNodeName(), "Big M template dgamma tiling success.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantBigMTiling::DgammaDoTilingStg0()
{
    // m 切分，合轴后的shape为（m, n）沿m轴做reduce
    constexpr static int64_t mFactorBlockAligned = MFACTOR_DEFAULT;

    int64_t quantBlocksNeeded = Ops::Base::CeilDiv(rows_, mFactorBlockAligned);
    usedCoreNumDgamma_ = quantBlocksNeeded < static_cast<int64_t>(aivCoreNum_) ? quantBlocksNeeded : aivCoreNum_;

    int64_t quantMPerBlock = Ops::Base::FloorDiv(rows_, usedCoreNumDgamma_);
    int64_t quantRemainder = rows_ - usedCoreNumDgamma_ * quantMPerBlock;

    int64_t quantMainMToProcess = quantMPerBlock + 1;
    int64_t quantTailMToProcess = quantMPerBlock;

    int64_t quantMainMLoop = Ops::Base::FloorDiv(quantMainMToProcess, mFactorBlockAligned);
    int64_t quantMainMTotalLoop = Ops::Base::CeilDiv(quantMainMToProcess, mFactorBlockAligned);
    int64_t quantMainMTail = quantMainMToProcess - quantMainMLoop * mFactorBlockAligned;
    int64_t quantMainBasicBlockLoop = FindNearestPower2(quantMainMTotalLoop);
    int64_t quantMainFoldCount = quantMainMLoop - quantMainBasicBlockLoop;

    int64_t quantMainCacheBufferCount = 1;
    int64_t quantMainResultCacheId = 0;
    if (quantMainBasicBlockLoop != 0) {
        quantMainCacheBufferCount = ULONG_BIT_LEN - static_cast<int64_t>(
                                                        __builtin_clzl(static_cast<uint64_t>(quantMainBasicBlockLoop)));
        quantMainResultCacheId = GetCacheID(quantMainBasicBlockLoop - 1);
    }

    int64_t quantTailMLoop = Ops::Base::FloorDiv(quantTailMToProcess, mFactorBlockAligned);
    int64_t quantTailMTotalLoop = Ops::Base::CeilDiv(quantTailMToProcess, mFactorBlockAligned);
    int64_t quantTailMTail = quantTailMToProcess - quantTailMLoop * mFactorBlockAligned;
    int64_t quantTailBasicBlockLoop = FindNearestPower2(quantTailMTotalLoop);
    int64_t quantTailFoldCount = quantTailMLoop - quantTailBasicBlockLoop;

    int64_t quantTailCacheBufferCount = 1;
    int64_t quantTailResultCacheId = 0;
    if (quantTailBasicBlockLoop != 0) {
        quantTailCacheBufferCount = ULONG_BIT_LEN - static_cast<int64_t>(
                                                        __builtin_clzl(static_cast<uint64_t>(quantTailBasicBlockLoop)));
        quantTailResultCacheId = GetCacheID(quantTailBasicBlockLoop - 1);
    }

    // n切分
    constexpr static int64_t gammaDefaultNfactor = 64;

    int64_t quantDyDtypeSize = dyDtype_ == ge::DataType::DT_FLOAT ? CONST_FOUR : CONST_TWO;

    int64_t quantNFactorMax = (ubSize_ - mFactorBlockAligned * sizeof(float) * CONST_THREE) /
                              (mFactorBlockAligned * (CONST_SIX * quantDyDtypeSize + sizeof(float)) +
                               sizeof(float) * (CONST_THREE + quantMainCacheBufferCount));
    OP_TILING_CHECK(quantNFactorMax < blockSize_ / gammaDefaultNfactor,
                    OP_LOGI(context_->GetNodeName(),
                            "Big M template is not capable. merged shape is (%lu, %lu), ub size: %luB, "
                            "quantNFactorMax: %ld.",
                            rows_, cols_, ubSize_, quantNFactorMax),
                    return ge::GRAPH_PARAM_INVALID);

    int64_t quantDyBlockLen = blockSize_ / quantDyDtypeSize;
    int64_t quantNFactorBlockAligned = Ops::Base::FloorAlign(quantNFactorMax, quantDyBlockLen);
    quantNFactorBlockAligned = quantNFactorBlockAligned > cols_ ? cols_ : quantNFactorBlockAligned;
    quantNFactorBlockAligned = Ops::Base::CeilAlign(quantNFactorBlockAligned, quantDyBlockLen);
    int64_t quantNLoop = Ops::Base::FloorDiv(cols_, quantNFactorBlockAligned);
    int64_t quantNTail = cols_ - quantNLoop * quantNFactorBlockAligned;

    // 参数设置
    tilingData_.dgammaUsedCoreNum = usedCoreNumDgamma_;
    tilingData_.dgammaMPerBlock = quantMPerBlock;
    tilingData_.dgammaMReminder = quantRemainder;
    tilingData_.dgammaNloop = quantNLoop;
    tilingData_.dgammaNtail = quantNTail;
    tilingData_.dgammaMfactorBlockAligned = mFactorBlockAligned;
    tilingData_.dgammaNfactorBlockAligned = quantNFactorBlockAligned;

    tilingData_.dgammaMToProcessMainBlock = quantMainMToProcess;
    tilingData_.dgammaMLoopMainBlock = quantMainMLoop;
    tilingData_.dgammaMTotalLoopMainBlock = quantMainMTotalLoop;
    tilingData_.dgammaMTailMainBlock = quantMainMTail;
    tilingData_.dgammaBasicBlockLoopMainBlock = quantMainBasicBlockLoop;
    tilingData_.dgammaMainFoldCountMainBlock = quantMainFoldCount;
    tilingData_.dgammaCacheBufferCountMainBlock = quantMainCacheBufferCount;
    tilingData_.dgammaResultCacheIDMainBlock = quantMainResultCacheId;

    tilingData_.dgammaMToProcessTailBlock = quantTailMToProcess;
    tilingData_.dgammaMLoopTailBlock = quantTailMLoop;
    tilingData_.dgammaMTotalLoopTailBlock = quantTailMTotalLoop;
    tilingData_.dgammaMTailTailBlock = quantTailMTail;
    tilingData_.dgammaBasicBlockLoopTailBlock = quantTailBasicBlockLoop;
    tilingData_.dgammaMainFoldCountTailBlock = quantTailFoldCount;
    tilingData_.dgammaCacheBufferCountTailBlock = quantTailCacheBufferCount;
    tilingData_.dgammaResultCacheIDTailBlock = quantTailResultCacheId;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantBigMTiling::DgammaDoTilingStg1()
{
    int64_t quantAInnerMax = ubSize_ / CONST_TWO / sizeof(float) / (usedCoreNumDgamma_ + 1);

    int64_t quantBlockLen = blockSize_ / sizeof(float);

    OP_TILING_CHECK(
        quantAInnerMax < quantBlockLen,
        OP_LOGI(context_->GetNodeName(),
                "Big M template is not capable for dgamma compute,  aInnerMax in stage1 is %ld .", quantAInnerMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t quantAInnerMaxAligned = Ops::Base::FloorAlign(quantAInnerMax, quantBlockLen);

    int64_t quantAInner = cols_ < quantAInnerMaxAligned ? cols_ : quantAInnerMaxAligned;
    int64_t quantAInnerAligned = Ops::Base::CeilAlign(quantAInner, quantBlockLen);

    int64_t quantAOuter = Ops::Base::CeilDiv(cols_, quantAInnerAligned);
    int64_t quantATail = cols_ - (quantAOuter - 1) * quantAInnerAligned;

    tilingData_.dgammaAInnerAlignedStg1 = quantAInnerAligned;
    tilingData_.dgammaAOuterStg1 = quantAOuter;
    tilingData_.dgammaATailStg1 = quantATail;

    return ge::GRAPH_SUCCESS;
}

uint64_t RmsNormGradQuantBigMTiling::GetTilingKey() const
{
    rms_norm_grad_quant::RmsNormGradQuantTilingKey tilingKey;

    tilingKey.SetComputeModeDx(computeModeDx_);
    tilingKey.SetComputeModeDgamma(computeModeDgamma_);
    tilingKey.SetComputeModeOffsetX(hasOffsetX_);
    tilingKey.SetComputeModeDivMode(divMode_);

    return tilingKey.GetTilingKey();
}

ge::graphStatus RmsNormGradQuantBigMTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    int64_t wsSize = usedCoreNumDgamma_ * cols_ * sizeof(float) + workspaceSize_;
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = static_cast<size_t>(wsSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantBigMTiling::PostTiling()
{
    int64_t usedCoreNums = usedCoreNumDx_ > usedCoreNumDgamma_ ? usedCoreNumDx_ :
                                                                 usedCoreNumDgamma_; // usedCoreNumDx_ 为0？
    context_->SetBlockDim(usedCoreNums);
    context_->SetScheduleMode(1); // Set to batch mode, all cores start simultaneously
    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", aivCoreNum_);
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(tilingData_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(tilingData_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

int64_t RmsNormGradQuantBigMTiling::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

int64_t RmsNormGradQuantBigMTiling::FindNearestPower2(const int64_t quantValue)
{
    if (quantValue <= CONST_ONE) {
        return CONST_ZERO;
    } else if (quantValue <= CONST_TWO) {
        return CONST_ONE;
    } else if (quantValue <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t quantNum = quantValue - CONST_ONE;
        const int64_t quantPower = CONST_SIXTY_THREE - __builtin_clzl(quantNum);
        return (CONST_ONE << quantPower);
    }
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormGradQuant, RmsNormGradQuantBigMTiling, 500);

} // namespace optiling
