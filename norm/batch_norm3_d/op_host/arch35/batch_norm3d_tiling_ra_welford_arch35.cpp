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
 * \file batch_norm3d_tiling_ra_welford_arch35.cpp
 * \brief
 */
#include "batch_norm3d_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_RA_WELFORD = 500000;

constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t A_UB_FACTOR_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t RA_BINARY_ADD_THRESHOLD = 4;
constexpr int64_t MEAN_VAR_BUFFER_NUM = 2;
constexpr int64_t RUNNING_MEAN_VAR_BUFFER_NUM = 4;
constexpr int64_t BETA_GAMMA_BUFFER_NUM = 2;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
} // namespace

namespace optiling {
class BatchNorm3DRAWelfordTilingBase : public BatchNorm3DRegbaseTilingBase {
public:
    explicit BatchNorm3DRAWelfordTilingBase(gert::TilingContext* context) : BatchNorm3DRegbaseTilingBase(context) {}
    ~BatchNorm3DRAWelfordTilingBase() override = default;

    void Reset(gert::TilingContext* context) override { BatchNorm3DRegbaseTilingBase::Reset(context); }

protected:
    bool IsCapable() override
    {
        // 只支持ra 场景
        if (r0_ != CONST_ONE) {
            return false;
        }
        return true;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void SetInputInfo();
    ge::graphStatus BinaryAddTiling(int64_t elemSize, int64_t theLeastAPerCore);

private:
    BatchNorm3DRAWelfordTilingData batchNormTilingData;
};

void BatchNorm3DRAWelfordTilingBase::SetInputInfo()
{
    // dim
    batchNormTilingData.r = r1_;
    batchNormTilingData.a = a_;

    // attr
    batchNormTilingData.epsilon = epsilon_;
    batchNormTilingData.momentum = exponentialAvgFactor_;
}

// aFactor must be aligned
ge::graphStatus BatchNorm3DRAWelfordTilingBase::BinaryAddTiling(int64_t elemSize, int64_t aFactor)
{
    // rFactor
    int64_t runningMeanVarSize = aFactor * static_cast<int64_t>(sizeof(float)) * RUNNING_MEAN_VAR_BUFFER_NUM;
    int64_t saveMeanRstdSize = aFactor * static_cast<int64_t>(sizeof(float)) * MEAN_VAR_BUFFER_NUM;
    int64_t betaGammaSize = aFactor * static_cast<int64_t>(sizeof(float)) * BETA_GAMMA_BUFFER_NUM;
    int64_t xSizePerR = aFactor * elemSize * DOUBLE_BUFFER_NUM;
    int64_t ySizePerR = aFactor * elemSize * DOUBLE_BUFFER_NUM;
    int64_t tmpMeanM2PerR = aFactor * static_cast<int64_t>(sizeof(float)) * MEAN_VAR_BUFFER_NUM;
    int64_t tmpCountPerR = sizeof(float);

    int64_t ubSizeCanUse = aicoreParams_.ubSize - runningMeanVarSize - saveMeanRstdSize - betaGammaSize;
    int64_t rFactor = ubSizeCanUse / (xSizePerR + ySizePerR + tmpMeanM2PerR + tmpCountPerR);
    rFactor = Ops::Base::FloorAlign(rFactor, RA_BINARY_ADD_THRESHOLD);
    OP_CHECK_IF(rFactor == 0, OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "rfactor is 0."),
                return ge::GRAPH_FAILED);
    int64_t rFactorAlign = Ops::Base::CeilAlign(static_cast<int64_t>(rFactor * sizeof(float)),
                                                static_cast<int64_t>(blockSize_)) /
                           sizeof(float);
    if ((rFactor != rFactorAlign) &&
        ((rFactor * (xSizePerR + ySizePerR + tmpMeanM2PerR) + rFactorAlign * tmpCountPerR) > ubSizeCanUse)) {
        rFactor -= RA_BINARY_ADD_THRESHOLD;
    }
    if (rFactor > r1_) {
        rFactor = Ops::Base::FloorAlign(r1_, RA_BINARY_ADD_THRESHOLD);
    }
    batchNormTilingData.rFactor = rFactor;

    int64_t binaryQuotient = RA_BINARY_ADD_THRESHOLD;
    while (binaryQuotient < rFactor) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;
    batchNormTilingData.binaryAddQuotient = binaryQuotient;
    int64_t binaryAddNum = binaryQuotient / RA_BINARY_ADD_THRESHOLD;
    int64_t binaryAddK = 0;
    int64_t curBinaryAddNum = 1;
    while (curBinaryAddNum < binaryAddNum) {
        binaryAddK++;
        curBinaryAddNum *= BINARY_ADD_COEF_FOUR;
    }
    if (curBinaryAddNum == binaryAddNum) {
        batchNormTilingData.binaryAddK = binaryAddK;
        batchNormTilingData.binaryAddLast = 0;
    } else if (curBinaryAddNum == binaryAddNum * BINARY_ADD_COEF) {
        batchNormTilingData.binaryAddK = binaryAddK - 1;
        batchNormTilingData.binaryAddLast = 1;
    } else {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Binary add calculate error.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNorm3DRAWelfordTilingBase::DoOpTiling()
{
    SetInputInfo();
    // core num
    int64_t elemSize = FLOAT32_BYTES;
    if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        elemSize = FLOAT16_BYTES;
    }
    int64_t theLeastAPerCore = blockSize_ / elemSize;
    int64_t blockFactor = Ops::Base::CeilDiv(a_, static_cast<int64_t>(aicoreParams_.blockDim));
    if (blockFactor < theLeastAPerCore) {
        blockFactor = theLeastAPerCore;
    }
    blockFactor = Ops::Base::CeilAlign(blockFactor * elemSize, static_cast<int64_t>(blockSize_)) / elemSize;
    usedCoreNums_ = Ops::Base::CeilDiv(a_, blockFactor);
    batchNormTilingData.aBlockFactor = blockFactor;
    batchNormTilingData.blockNum = usedCoreNums_;
    int64_t aFactor = vlFp32_ * A_UB_FACTOR_COEF;
    if (blockFactor <= static_cast<int64_t>(vlFp32_ * A_UB_FACTOR_COEF)) {
        aFactor = Ops::Base::CeilAlign(blockFactor * elemSize, static_cast<int64_t>(blockSize_)) / elemSize;
    }
    batchNormTilingData.aFactor = aFactor;

    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < r1_) {
        powerOfTwoForR *= BINARY_ADD_COEF;
    }
    batchNormTilingData.powerOfTwoForR = powerOfTwoForR;

    batchNormTilingData.useRunningMeanVar = useRunningMeanVar_;

    return BinaryAddTiling(elemSize, aFactor);
}

uint64_t BatchNorm3DRAWelfordTilingBase::GetTilingKey() const { return TILINGKEY_RA_WELFORD; }

ge::graphStatus BatchNorm3DRAWelfordTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto* tilingDataOut = context_->GetTilingData<BatchNorm3DRAWelfordTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataOut);
    *tilingDataOut = batchNormTilingData;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm3D, BatchNorm3DRAWelfordTilingBase, 15000);
} // namespace optiling
