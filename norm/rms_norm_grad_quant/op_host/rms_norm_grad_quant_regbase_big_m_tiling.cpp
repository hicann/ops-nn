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
    tilingData_.dxTilingData.ubFactor = ubFactor_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantBigMTiling::DgammaDoTiling()
{
    OP_TILING_CHECK(
        DgammaDoTilingStg0() != ge::GRAPH_SUCCESS,
        OP_LOGI(context_->GetNodeName(), "Big M template dgamma do tiling stage 0 failed."),
        return ge::GRAPH_PARAM_INVALID);

    OP_TILING_CHECK(
        DgammaDoTilingStg1() != ge::GRAPH_SUCCESS,
        OP_LOGI(context_->GetNodeName(), "Big M template dgamma do tiling stage 1 failed."),
        return ge::GRAPH_PARAM_INVALID);

    OP_LOGD(context_->GetNodeName(), "Big M template dgamma tiling success.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantBigMTiling::DgammaDoTilingStg0()
{
    // m 切分，合轴后的shape为（m, n）沿m轴做reduce
    constexpr static int64_t mFactorBlockAligned = MFACTOR_DEFAULT;

    int64_t blocksNeeded = Ops::Base::CeilDiv(rows_, mFactorBlockAligned);
    usedCoreNumDgamma_ = blocksNeeded < static_cast<int64_t>(aivCoreNum_) ? blocksNeeded : aivCoreNum_;

    int64_t mPerBlock = Ops::Base::FloorDiv(rows_, usedCoreNumDgamma_);
    int64_t remainder = rows_ - usedCoreNumDgamma_ * mPerBlock;

    int64_t mToProcessMainBlock = mPerBlock + 1;
    int64_t mToProcessTailBlock = mPerBlock;

    int64_t mLoopMainBlock = Ops::Base::FloorDiv(mToProcessMainBlock, mFactorBlockAligned);
    int64_t mTotalLoopMainBlock = Ops::Base::CeilDiv(mToProcessMainBlock, mFactorBlockAligned);
    int64_t mTailMainBlock = mToProcessMainBlock - mLoopMainBlock * mFactorBlockAligned;
    int64_t basicBlockLoopMainBlock = FindNearestPower2(mTotalLoopMainBlock);
    int64_t mainFoldCountMainBlock = mLoopMainBlock - basicBlockLoopMainBlock;

    int64_t cacheBufferCountMainBlock = 1;
    int64_t resultCacheIDMainBlock = 0;
    if (basicBlockLoopMainBlock != 0) {
        cacheBufferCountMainBlock =
            ULONG_BIT_LEN - static_cast<int64_t>(__builtin_clzl(static_cast<uint64_t>(basicBlockLoopMainBlock)));
        resultCacheIDMainBlock = GetCacheID(basicBlockLoopMainBlock - 1);
    }

    int64_t mLoopTailBlock = Ops::Base::FloorDiv(mToProcessTailBlock, mFactorBlockAligned);
    int64_t mTotalLoopTailBlock = Ops::Base::CeilDiv(mToProcessTailBlock, mFactorBlockAligned);
    int64_t mTailTailBlock = mToProcessTailBlock - mLoopTailBlock * mFactorBlockAligned;
    int64_t basicBlockLoopTailBlock = FindNearestPower2(mTotalLoopTailBlock);
    int64_t mainFoldCountTailBlock = mLoopTailBlock - basicBlockLoopTailBlock;

    int64_t cacheBufferCountTailBlock = 1;
    int64_t resultCacheIDTailBlock = 0;
    if (basicBlockLoopTailBlock != 0) {
        cacheBufferCountTailBlock =
            ULONG_BIT_LEN - static_cast<int64_t>(__builtin_clzl(static_cast<uint64_t>(basicBlockLoopTailBlock)));
        resultCacheIDTailBlock = GetCacheID(basicBlockLoopTailBlock - 1);
    }

    // n切分
    constexpr static int64_t gammaDefaultNfactor = 64;

    int64_t dyDtypeSize = dyDtype_ == ge::DataType::DT_FLOAT ? CONST_FOUR : CONST_TWO;

    int64_t nFactorMax = (ubSize_ - mFactorBlockAligned * sizeof(float) * CONST_THREE) /
                         (mFactorBlockAligned * (CONST_SIX * dyDtypeSize + sizeof(float)) +
                          sizeof(float) * (CONST_THREE + cacheBufferCountMainBlock));
    OP_TILING_CHECK(
        nFactorMax < blockSize_ / gammaDefaultNfactor,
        OP_LOGI(
            context_->GetNodeName(),
            "Big M template is not capable. merged shape is (%lu, %lu), ub size: %luB, "
            "nFactorMax: %ld.",
            rows_, cols_, ubSize_, nFactorMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t dyBlockLen = blockSize_ / dyDtypeSize;
    int64_t nFactorBlockAligned = Ops::Base::FloorAlign(nFactorMax, dyBlockLen);
    nFactorBlockAligned = nFactorBlockAligned > cols_ ? cols_ : nFactorBlockAligned;
    nFactorBlockAligned = Ops::Base::CeilAlign(nFactorBlockAligned, dyBlockLen);
    int64_t nLoop = Ops::Base::FloorDiv(cols_, nFactorBlockAligned);
    int64_t nTail = cols_ - nLoop * nFactorBlockAligned;

    // 参数设置
    tilingData_.dgammaUsedCoreNum = usedCoreNumDgamma_;
    tilingData_.dgammaMPerBlock = mPerBlock;
    tilingData_.dgammaMReminder = remainder;
    tilingData_.dgammaNloop = nLoop;
    tilingData_.dgammaNtail = nTail;
    tilingData_.dgammaMfactorBlockAligned = mFactorBlockAligned;
    tilingData_.dgammaNfactorBlockAligned = nFactorBlockAligned;

    tilingData_.dgammaMToProcessMainBlock = mToProcessMainBlock;
    tilingData_.dgammaMLoopMainBlock = mLoopMainBlock;
    tilingData_.dgammaMTotalLoopMainBlock = mTotalLoopMainBlock;
    tilingData_.dgammaMTailMainBlock = mTailMainBlock;
    tilingData_.dgammaBasicBlockLoopMainBlock = basicBlockLoopMainBlock;
    tilingData_.dgammaMainFoldCountMainBlock = mainFoldCountMainBlock;
    tilingData_.dgammaCacheBufferCountMainBlock = cacheBufferCountMainBlock;
    tilingData_.dgammaResultCacheIDMainBlock = resultCacheIDMainBlock;

    tilingData_.dgammaMToProcessTailBlock = mToProcessTailBlock;
    tilingData_.dgammaMLoopTailBlock = mLoopTailBlock;
    tilingData_.dgammaMTotalLoopTailBlock = mTotalLoopTailBlock;
    tilingData_.dgammaMTailTailBlock = mTailTailBlock;
    tilingData_.dgammaBasicBlockLoopTailBlock = basicBlockLoopTailBlock;
    tilingData_.dgammaMainFoldCountTailBlock = mainFoldCountTailBlock;
    tilingData_.dgammaCacheBufferCountTailBlock = cacheBufferCountTailBlock;
    tilingData_.dgammaResultCacheIDTailBlock = resultCacheIDTailBlock;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradQuantBigMTiling::DgammaDoTilingStg1()
{
    int64_t aInnerMax = ubSize_ / CONST_TWO / sizeof(float) / (usedCoreNumDgamma_ + 1);

    int64_t blockLen = blockSize_ / sizeof(float);

    OP_TILING_CHECK(
        aInnerMax < blockLen,
        OP_LOGI(
            context_->GetNodeName(), "Big M template is not capable for dgamma compute,  aInnerMax in stage1 is %ld .",
            aInnerMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t aInnerMaxAligned = Ops::Base::FloorAlign(aInnerMax, blockLen);

    int64_t aInner = cols_ < aInnerMaxAligned ? cols_ : aInnerMaxAligned;
    int64_t aInnerAligned = Ops::Base::CeilAlign(aInner, blockLen);

    int64_t aOuter = Ops::Base::CeilDiv(cols_, aInnerAligned);
    int64_t aTail = cols_ - (aOuter - 1) * aInnerAligned;

    tilingData_.dgammaAInnerAlignedStg1 = aInnerAligned;
    tilingData_.dgammaAOuterStg1 = aOuter;
    tilingData_.dgammaATailStg1 = aTail;

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
    int64_t usedCoreNums = usedCoreNumDx_ > usedCoreNumDgamma_ ? usedCoreNumDx_ : usedCoreNumDgamma_; // usedCoreNumDx_ 为0？
    context_->SetBlockDim(usedCoreNums);
    context_->SetScheduleMode(1); // Set to batch mode, all cores start simultaneously
    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", aivCoreNum_);
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        sizeof(tilingData_) > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(tilingData_),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData_)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

int64_t RmsNormGradQuantBigMTiling::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

int64_t RmsNormGradQuantBigMTiling::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - __builtin_clzl(num);
        return (CONST_ONE << pow);
    }
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormGradQuant, RmsNormGradQuantBigMTiling, 500);

} // namespace optiling