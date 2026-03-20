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
 * \file batch_norm_tiling_block_split_r_arch35.cpp
 * \brief
 */
#include "batch_norm_tiling.h"

using namespace ge;

namespace
{
constexpr int64_t TILINGKEY_BLOCK_SPLIT_R = 600000;

constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t GAMMA_BETA_NODE_NUM = 2;
constexpr int64_t RUNNING_MEAN_VAR_NODE_NUM = 4;
constexpr int64_t BATCH_MEAN_VAR_NODE_NUM = 2;
constexpr int64_t X_IN_OUT_NODE_NUM = 2;
constexpr int64_t TBUF_NODE_NUM = 3;
constexpr int64_t MEAN_AND_VAR_NODE_NUM = 2;

constexpr int64_t RUNNING_MEAN_INPUT_IDX = 3;

constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t RA_BINARY_ADD_THRESHOLD = 4;
constexpr int64_t WSP_RESERVED_SIZE = 16L * 1024L * 1024L;
constexpr int64_t CACHE_LINE_SIZE = 256;
constexpr int64_t SPLIT_R_TEMPLATE_A_THRESHOLD = 512;

}  // namespace
static int64_t FindBinaryQuotient(int64_t len)
{
    int64_t binaryQuotient = 1;
    while (binaryQuotient <= len) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;
    return binaryQuotient;
}

namespace optiling
{
class BatchNormBlockSplitRTiling : public BatchNormRegbaseTilingBase
{
public:
    explicit BatchNormBlockSplitRTiling(gert::TilingContext* context) : BatchNormRegbaseTilingBase(context)
    {
    }
    ~BatchNormBlockSplitRTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormRegbaseTilingBase::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        // 只支持ra 场景
        if (r0_ != CONST_ONE) {
            return false;
        }

        if (a_ > SPLIT_R_TEMPLATE_A_THRESHOLD) {
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
    bool BinaryAddTiling(const int64_t binaryAddNum, int64_t& binaryAddK, int64_t& binaryAddLast);

private:
    BatchNormBlockSplitRTilingData batchNormTilingData;
};

void BatchNormBlockSplitRTiling::SetInputInfo()
{
    // dim
    batchNormTilingData.set_patternR(r1_);
    batchNormTilingData.set_patternA(a_);

    // attr
    batchNormTilingData.set_epsilon(epsilon_);
    batchNormTilingData.set_momentum(exponentialAvgFactor_);
    batchNormTilingData.set_momentumReverse(1 - exponentialAvgFactor_);
}

bool BatchNormBlockSplitRTiling::BinaryAddTiling(const int64_t binaryAddNum, int64_t& binaryAddK,
                                                 int64_t& binaryAddLast)
{
    binaryAddK = 0;
    int64_t curBinaryAddNum = 1;
    while (curBinaryAddNum < binaryAddNum) {
        binaryAddK++;
        curBinaryAddNum *= BINARY_ADD_COEF_FOUR;
    }
    if (curBinaryAddNum == binaryAddNum) {
        binaryAddLast = 0;
    } else if (curBinaryAddNum == binaryAddNum * BINARY_ADD_COEF) {
        binaryAddK = binaryAddK - 1;
        binaryAddLast = 1;
    } else {
        OP_LOGI(context_->GetNodeName(), "BinaryAddTiling binaryAddNum %ld case not supported", binaryAddNum);
        return false;
    }
    return true;
}

ge::graphStatus BatchNormBlockSplitRTiling::DoOpTiling()
{
    SetInputInfo();
    int64_t elemSize = FLOAT32_BYTES;
    int64_t gammaBetaNodeNum = GAMMA_BETA_NODE_NUM;
    int64_t runningMeanVarNodeNum = RUNNING_MEAN_VAR_NODE_NUM;
    int64_t inOutNodeNum = X_IN_OUT_NODE_NUM * DOUBLE_BUFFER_NUM;
    if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        elemSize = FLOAT16_BYTES;
        gammaBetaNodeNum = gammaBetaNodeNum / (FLOAT32_BYTES / FLOAT16_BYTES);
        inOutNodeNum = inOutNodeNum / (FLOAT32_BYTES / FLOAT16_BYTES);
    }

    int64_t patternAAlign = Ops::Base::CeilAlign(a_, static_cast<int64_t>(blockSize_ / elemSize));
    int64_t aUbFactor = CACHE_LINE_SIZE / elemSize;
    int64_t aUbLoop = Ops::Base::CeilDiv(a_, aUbFactor);
    int64_t aUbTail = a_ - (aUbLoop - 1) * aUbFactor;
    if (aUbLoop == 1) {
        aUbFactor = patternAAlign;
    }
    batchNormTilingData.set_patternAAlign(patternAAlign);
    batchNormTilingData.set_aUbFactor(aUbFactor);
    batchNormTilingData.set_aUbLoop(aUbLoop);
    batchNormTilingData.set_aUbTail(aUbTail);
    int64_t aAlignSize = aUbFactor * FLOAT32_BYTES;
    int64_t fp32EleNumPerBlock = blockSize_ / FLOAT32_BYTES;
    int64_t rFactorMaxAlignSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(aicoreParams_.ubSize) / (aAlignSize * (inOutNodeNum + TBUF_NODE_NUM)),
                       fp32EleNumPerBlock) *
        FLOAT32_BYTES;
    int64_t blockDimAlignSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(aicoreParams_.blockDim), fp32EleNumPerBlock) * FLOAT32_BYTES;
    int64_t ubSizeCanUse = aicoreParams_.ubSize - blockDimAlignSize - rFactorMaxAlignSize -
                           aAlignSize * (runningMeanVarNodeNum + gammaBetaNodeNum + BATCH_MEAN_VAR_NODE_NUM);
    int64_t rUbFactor = Ops::Base::FloorDiv(ubSizeCanUse, aAlignSize * (inOutNodeNum + TBUF_NODE_NUM));
    rUbFactor = std::min(rUbFactor, batchNormTilingData.get_patternR());
    rUbFactor = Ops::Base::FloorAlign(rUbFactor, RA_BINARY_ADD_THRESHOLD);
    OP_CHECK_IF(rUbFactor == 0, OP_LOGI(context_->GetNodeName(), "BatchNormBlockSplitRTiling rUbFactor == 0"),
                return ge::GRAPH_PARAM_INVALID);
    int64_t rGroups = Ops::Base::FloorDiv(batchNormTilingData.get_patternR(), rUbFactor);
    usedCoreNums_ = std::min(rGroups, static_cast<int64_t>(aicoreParams_.blockDim));
    int64_t tBufUbFactor = std::max(rUbFactor, usedCoreNums_);
    // LastFinalize tbuf无法复用，重新计算rFactor
    while (((rUbFactor * inOutNodeNum + tBufUbFactor * TBUF_NODE_NUM) * aAlignSize > ubSizeCanUse)) {
        rUbFactor -= RA_BINARY_ADD_THRESHOLD;
        rUbFactor = std::min(rUbFactor, batchNormTilingData.get_patternR());
        rUbFactor = Ops::Base::FloorAlign(rUbFactor, RA_BINARY_ADD_THRESHOLD);
        rGroups = Ops::Base::FloorDiv(batchNormTilingData.get_patternR(), rUbFactor);
        usedCoreNums_ = std::min(rGroups, static_cast<int64_t>(aicoreParams_.blockDim));
        tBufUbFactor = std::max(rUbFactor, usedCoreNums_);
    }
    OP_CHECK_IF(rUbFactor <= 0, OP_LOGI(context_->GetNodeName(), "BatchNormBlockSplitRTiling rUbFactor == 0"),
                return ge::GRAPH_PARAM_INVALID);
    int64_t formerCoreBlockFactor = Ops::Base::CeilDiv(rGroups, usedCoreNums_);
    int64_t tailCoreBlockFactor = formerCoreBlockFactor - 1;
    int64_t tailCoreNums = formerCoreBlockFactor * usedCoreNums_ - rGroups;
    int64_t formerCoreNums = usedCoreNums_ - tailCoreNums;
    int64_t tailR = batchNormTilingData.get_patternR() - rGroups * rUbFactor;
    int64_t binaryAddQuotient = FindBinaryQuotient(rUbFactor);
    int64_t binaryAddNum = binaryAddQuotient / RA_BINARY_ADD_THRESHOLD;
    int64_t binaryAddK = 0;
    int64_t binaryAddLast = 0;
    auto res0 = BinaryAddTiling(binaryAddNum, binaryAddK, binaryAddLast);
    int64_t lastBinaryAddQuotient = FindBinaryQuotient(usedCoreNums_);
    int64_t lastBinaryAddK = 0;
    int64_t lastBinaryAddLast = 0;
    auto res1 = BinaryAddTiling(lastBinaryAddQuotient, lastBinaryAddK, lastBinaryAddLast);
    OP_CHECK_IF(res0 == false || res1 == false,
                OP_LOGI(context_->GetNodeName(), "BatchNormBlockSplitRTiling BinaryAddTiling param invalid"),
                return ge::GRAPH_PARAM_INVALID);
    batchNormTilingData.set_tBufUbFactor(tBufUbFactor);
    batchNormTilingData.set_rUbFactor(rUbFactor);
    batchNormTilingData.set_formerCoreBlockFactor(formerCoreBlockFactor);
    batchNormTilingData.set_tailCoreBlockFactor(tailCoreBlockFactor);
    batchNormTilingData.set_formerCoreNums(formerCoreNums);
    batchNormTilingData.set_tailCoreNums(tailCoreNums);
    batchNormTilingData.set_tailR(tailR);
    batchNormTilingData.set_binaryAddQuotient(binaryAddQuotient);
    batchNormTilingData.set_binaryAddK(binaryAddK);
    batchNormTilingData.set_binaryAddLast(binaryAddLast);
    batchNormTilingData.set_lastBinaryAddQuotient(lastBinaryAddQuotient);
    batchNormTilingData.set_lastBinaryAddK(lastBinaryAddK);
    batchNormTilingData.set_lastBinaryAddLast(lastBinaryAddLast);

    batchNormTilingData.set_useRunningMeanVar(useRunningMeanVar_);

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormBlockSplitRTiling::GetTilingKey() const
{
    return TILINGKEY_BLOCK_SPLIT_R;
}

ge::graphStatus BatchNormBlockSplitRTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = WSP_RESERVED_SIZE + usedCoreNums_ * MEAN_AND_VAR_NODE_NUM *
                                                  batchNormTilingData.get_patternAAlign() * FLOAT32_BYTES;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(batchNormTilingData.GetDataSize() > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(),
                    "actual tiling data size %zu > context tiling data size %zu",
                    batchNormTilingData.GetDataSize(), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    batchNormTilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(batchNormTilingData.GetDataSize());
    uint32_t batch_mode = 1U;
    auto ret = context_->SetScheduleMode(batch_mode);

    return ret;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm, BatchNormBlockSplitRTiling, 12000);
}  // namespace optiling