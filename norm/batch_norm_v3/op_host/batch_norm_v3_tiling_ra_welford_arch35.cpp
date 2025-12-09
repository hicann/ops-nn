/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file batch_norm_v3_tiling_ra_welford_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "batch_norm_v3_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_RA_WELFORD = 500000;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
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
class BatchNormV3RAWelfordTilingBase : public BatchNormV3RegbaseTilingBase {
public:
    explicit BatchNormV3RAWelfordTilingBase(gert::TilingContext* context) : BatchNormV3RegbaseTilingBase(context)
    {}
    ~BatchNormV3RAWelfordTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormV3RegbaseTilingBase::Reset(context);
        blockNum = 0;
    }

protected:
    bool IsCapable() override
    {
        if (format != FORMAT_NHWC && format != FORMAT_NDHWC) {
            // NCHW和NCDHW场景中，如果r0是1，也放在RA模版处理
            if (r0 != 1) {
                return false;
            }
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
    int64_t blockNum;
    BatchNormV3RAWelfordTilingData batchNormV3TilingData;
};

void BatchNormV3RAWelfordTilingBase::SetInputInfo()
{
    // dim
    batchNormV3TilingData.set_r(r1);
    batchNormV3TilingData.set_a(a);

    // attr
    batchNormV3TilingData.set_epsilon(epsilon);
    batchNormV3TilingData.set_momentum(momentum);
}

// aFactor must be aligned
ge::graphStatus BatchNormV3RAWelfordTilingBase::BinaryAddTiling(int64_t elemSize, int64_t aFactor)
{
    // rFactor
    int64_t runningMeanVarSize = aFactor * static_cast<int64_t>(sizeof(float)) * RUNNING_MEAN_VAR_BUFFER_NUM;
    int64_t saveMeanRstdSize = aFactor * static_cast<int64_t>(sizeof(float)) * MEAN_VAR_BUFFER_NUM;
    int64_t betaGammaSize = aFactor * elemSize * BETA_GAMMA_BUFFER_NUM;
    int64_t xSizePerR = aFactor * elemSize * DOUBLE_BUFFER_NUM;
    int64_t ySizePerR = aFactor * elemSize * DOUBLE_BUFFER_NUM;
    int64_t tmpMeanM2PerR = aFactor * static_cast<int64_t>(sizeof(float)) * MEAN_VAR_BUFFER_NUM;
    int64_t tmpCountPerR = sizeof(float);

    int64_t ubSizeCanUse = aicoreParams_.ubSize - runningMeanVarSize - saveMeanRstdSize - betaGammaSize;
    int64_t rFactor = ubSizeCanUse / (xSizePerR + ySizePerR + tmpMeanM2PerR + tmpCountPerR);
    rFactor = Ops::Base::FloorAlign(rFactor, RA_BINARY_ADD_THRESHOLD);
    OP_CHECK_IF(rFactor == 0, OP_LOGE(context_->GetNodeName(), "rfactor is 0."), return ge::GRAPH_FAILED);
    int64_t rFactorAlign =
        Ops::Base::CeilAlign(static_cast<int64_t>(rFactor * sizeof(float)), blockSize) / sizeof(float);
    if ((rFactor != rFactorAlign) &&
        ((rFactor * (xSizePerR + ySizePerR + tmpMeanM2PerR) + rFactorAlign * tmpCountPerR) > ubSizeCanUse)) {
        rFactor -= RA_BINARY_ADD_THRESHOLD;
    }
    if (rFactor > r1) {
        rFactor = Ops::Base::FloorAlign(r1, RA_BINARY_ADD_THRESHOLD);
    }
    batchNormV3TilingData.set_rFactor(rFactor);

    int64_t binaryQuotient = RA_BINARY_ADD_THRESHOLD;
    while (binaryQuotient < rFactor) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;
    batchNormV3TilingData.set_binaryAddQuotient(binaryQuotient);
    int64_t binaryAddNum = binaryQuotient / RA_BINARY_ADD_THRESHOLD;
    int64_t binaryAddK = 0;
    int64_t curBinaryAddNum = 1;
    while (curBinaryAddNum < binaryAddNum) {
        binaryAddK++;
        curBinaryAddNum *= BINARY_ADD_COEF_FOUR;
    }
    if (curBinaryAddNum == binaryAddNum) {
        batchNormV3TilingData.set_binaryAddK(binaryAddK);
        batchNormV3TilingData.set_binaryAddLast(0);
    } else if (curBinaryAddNum == binaryAddNum * BINARY_ADD_COEF) {
        batchNormV3TilingData.set_binaryAddK(binaryAddK - 1);
        batchNormV3TilingData.set_binaryAddLast(1);
    } else {
        OP_LOGE(context_->GetNodeName(), "Binary add calculate error.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RAWelfordTilingBase::DoOpTiling()
{
    SetInputInfo();
    // core num
    int64_t elemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
    }
    int64_t theLeastAPerCore = blockSize / elemSize;
    int64_t blockFactor = Ops::Base::CeilDiv(a, static_cast<int64_t>(aicoreParams_.blockDim));
    if (blockFactor < theLeastAPerCore) {
        blockFactor = theLeastAPerCore;
    }
    blockFactor = Ops::Base::CeilAlign(blockFactor * elemSize, blockSize) / elemSize;
    blockNum = Ops::Base::CeilDiv(a, blockFactor);
    batchNormV3TilingData.set_aBlockFactor(blockFactor);
    batchNormV3TilingData.set_blockNum(blockNum);
    int64_t aFactor = vl * A_UB_FACTOR_COEF;
    if (blockFactor <= vl * A_UB_FACTOR_COEF) {
        aFactor = Ops::Base::CeilAlign(blockFactor * elemSize, blockSize) / elemSize;
    }
    batchNormV3TilingData.set_aFactor(aFactor);

    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < r1) {
        powerOfTwoForR *= BINARY_ADD_COEF;
    }
    batchNormV3TilingData.set_powerOfTwoForR(powerOfTwoForR);

    return BinaryAddTiling(elemSize, aFactor);
}

uint64_t BatchNormV3RAWelfordTilingBase::GetTilingKey() const
{
    return TILINGKEY_RA_WELFORD;
}

ge::graphStatus BatchNormV3RAWelfordTilingBase::PostTiling()
{
    context_->SetBlockDim(blockNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        batchNormV3TilingData.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
            batchNormV3TilingData.GetDataSize(), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    batchNormV3TilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(batchNormV3TilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3RAWelfordTilingBase, 15000);
} // namespace optiling