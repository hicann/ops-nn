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
 * \file batch_norm_v3_tiling_block_split_r_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "batch_norm_v3_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_BLOCK_SPLIT_R = 600000;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;

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
constexpr int64_t SPLIT_R_TEMPLATE_A_THRESHOLD = 256;

} // namespace
static int64_t FindBinaryQuotient(int64_t len)
{
    int64_t binaryQuotient = 1;
    while (binaryQuotient <= len) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;
    return binaryQuotient;
}

namespace optiling {
class BatchNormV3BlockSplitRTiling : public BatchNormV3RegbaseTilingBase {
public:
    explicit BatchNormV3BlockSplitRTiling(gert::TilingContext* context) : BatchNormV3RegbaseTilingBase(context)
    {}
    ~BatchNormV3BlockSplitRTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormV3RegbaseTilingBase::Reset(context);
        usedCoreNum = 0;
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
        if (a > SPLIT_R_TEMPLATE_A_THRESHOLD) {
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
    int64_t usedCoreNum;
    BatchNormV3BlockSplitRTilingData batchNormV3TilingData;
};

void BatchNormV3BlockSplitRTiling::SetInputInfo()
{
    // dim
    batchNormV3TilingData.set_patternR(r1);
    batchNormV3TilingData.set_patternA(a);

    // attr
    batchNormV3TilingData.set_epsilon(epsilon);
    batchNormV3TilingData.set_momentum(momentum);
    batchNormV3TilingData.set_momentumReverse(1 - momentum);
}

bool BatchNormV3BlockSplitRTiling::BinaryAddTiling(
    const int64_t binaryAddNum, int64_t& binaryAddK, int64_t& binaryAddLast)
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

ge::graphStatus BatchNormV3BlockSplitRTiling::DoOpTiling()
{
    SetInputInfo();
    int64_t elemSize = FP32_BYTE;
    int64_t gammaBetaNodeNum = GAMMA_BETA_NODE_NUM;
    int64_t runningMeanVarNodeNum = RUNNING_MEAN_VAR_NODE_NUM;
    int64_t inOutNodeNum = X_IN_OUT_NODE_NUM * DOUBLE_BUFFER_NUM;
    auto runningMeanDesc = context_->GetInputDesc(RUNNING_MEAN_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningMeanDesc);
    auto runningMeanDataType = runningMeanDesc->GetDataType();
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
        gammaBetaNodeNum = gammaBetaNodeNum / (FP32_BYTE / FP16_BYTE);
        inOutNodeNum = inOutNodeNum / (FP32_BYTE / FP16_BYTE);
    }
    if (runningMeanDataType == ge::DT_FLOAT16 || runningMeanDataType == ge::DT_BF16) {
        runningMeanVarNodeNum = runningMeanVarNodeNum / (FP32_BYTE / FP16_BYTE);
    }
    int64_t patternAAlign = Ops::Base::CeilAlign(a, (blockSize / elemSize));
    int64_t aUbFactor = CACHE_LINE_SIZE / elemSize;
    int64_t aUbLoop = Ops::Base::CeilDiv(a, aUbFactor);
    int64_t aUbTail = a - (aUbLoop - 1) * aUbFactor;
    if (aUbLoop == 1) {
        aUbFactor = patternAAlign;
    }
    batchNormV3TilingData.set_patternAAlign(patternAAlign);
    batchNormV3TilingData.set_aUbFactor(aUbFactor);
    batchNormV3TilingData.set_aUbLoop(aUbLoop);
    batchNormV3TilingData.set_aUbTail(aUbTail);
    int64_t aAlignSize = aUbFactor * FP32_BYTE;
    int64_t fp32EleNumPerBlock = blockSize / FP32_BYTE;
    int64_t rFactorMaxAlignSize =
        Ops::Base::CeilAlign(
            static_cast<int64_t>(aicoreParams_.ubSize) / (aAlignSize * (inOutNodeNum + TBUF_NODE_NUM)),
            fp32EleNumPerBlock) *
        FP32_BYTE;
    int64_t blockDimAlignSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(aicoreParams_.blockDim), fp32EleNumPerBlock) * FP32_BYTE;
    int64_t ubSizeCanUse = aicoreParams_.ubSize - blockDimAlignSize - rFactorMaxAlignSize -
                           aAlignSize * (runningMeanVarNodeNum + gammaBetaNodeNum + BATCH_MEAN_VAR_NODE_NUM);
    int64_t rUbFactor = Ops::Base::FloorDiv(ubSizeCanUse, aAlignSize * (inOutNodeNum + TBUF_NODE_NUM));
    rUbFactor = std::min(rUbFactor, batchNormV3TilingData.get_patternR());
    rUbFactor = Ops::Base::FloorAlign(rUbFactor, RA_BINARY_ADD_THRESHOLD);
    OP_CHECK_IF(
        rUbFactor == 0, OP_LOGI(context_->GetNodeName(), "BatchNormV3BlockSplitRTiling rUbFactor == 0"),
        return ge::GRAPH_PARAM_INVALID);
    int64_t rGroups = Ops::Base::FloorDiv(batchNormV3TilingData.get_patternR(), rUbFactor);
    usedCoreNum = std::min(rGroups, static_cast<int64_t>(aicoreParams_.blockDim));
    int64_t tBufUbFactor = std::max(rUbFactor, usedCoreNum);
    // LastFinalize tbuf无法复用，重新计算rFactor
    while (((rUbFactor * inOutNodeNum + tBufUbFactor * TBUF_NODE_NUM) * aAlignSize > ubSizeCanUse)) {
        rUbFactor -= RA_BINARY_ADD_THRESHOLD;
        rUbFactor = std::min(rUbFactor, batchNormV3TilingData.get_patternR());
        rUbFactor = Ops::Base::FloorAlign(rUbFactor, RA_BINARY_ADD_THRESHOLD);
        rGroups = Ops::Base::FloorDiv(batchNormV3TilingData.get_patternR(), rUbFactor);
        usedCoreNum = std::min(rGroups, static_cast<int64_t>(aicoreParams_.blockDim));
        tBufUbFactor = std::max(rUbFactor, usedCoreNum);
    }
    OP_CHECK_IF(
        rUbFactor <= 0, OP_LOGI(context_->GetNodeName(), "BatchNormV3BlockSplitRTiling rUbFactor == 0"),
        return ge::GRAPH_PARAM_INVALID);
    int64_t formerCoreBlockFactor = Ops::Base::CeilDiv(rGroups, usedCoreNum);
    int64_t tailCoreBlockFactor = formerCoreBlockFactor - 1;
    int64_t tailCoreNums = formerCoreBlockFactor * usedCoreNum - rGroups;
    int64_t formerCoreNums = usedCoreNum - tailCoreNums;
    int64_t tailR = batchNormV3TilingData.get_patternR() - rGroups * rUbFactor;
    int64_t binaryAddQuotient = FindBinaryQuotient(rUbFactor);
    int64_t binaryAddNum = binaryAddQuotient / RA_BINARY_ADD_THRESHOLD;
    int64_t binaryAddK = 0;
    int64_t binaryAddLast = 0;
    auto res0 = BinaryAddTiling(binaryAddNum, binaryAddK, binaryAddLast);
    int64_t lastBinaryAddQuotient = FindBinaryQuotient(usedCoreNum);
    int64_t lastBinaryAddK = 0;
    int64_t lastBinaryAddLast = 0;
    auto res1 = BinaryAddTiling(lastBinaryAddQuotient, lastBinaryAddK, lastBinaryAddLast);
    OP_CHECK_IF(
        res0 == false || res1 == false,
        OP_LOGI(context_->GetNodeName(), "BatchNormV3BlockSplitRTiling BinaryAddTiling param invalid"),
        return ge::GRAPH_PARAM_INVALID);
    batchNormV3TilingData.set_tBufUbFactor(tBufUbFactor);
    batchNormV3TilingData.set_rUbFactor(rUbFactor);
    batchNormV3TilingData.set_formerCoreBlockFactor(formerCoreBlockFactor);
    batchNormV3TilingData.set_tailCoreBlockFactor(tailCoreBlockFactor);
    batchNormV3TilingData.set_formerCoreNums(formerCoreNums);
    batchNormV3TilingData.set_tailCoreNums(tailCoreNums);
    batchNormV3TilingData.set_tailR(tailR);
    batchNormV3TilingData.set_binaryAddQuotient(binaryAddQuotient);
    batchNormV3TilingData.set_binaryAddK(binaryAddK);
    batchNormV3TilingData.set_binaryAddLast(binaryAddLast);
    batchNormV3TilingData.set_lastBinaryAddQuotient(lastBinaryAddQuotient);
    batchNormV3TilingData.set_lastBinaryAddK(lastBinaryAddK);
    batchNormV3TilingData.set_lastBinaryAddLast(lastBinaryAddLast);

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormV3BlockSplitRTiling::GetTilingKey() const
{
    return TILINGKEY_BLOCK_SPLIT_R;
}

ge::graphStatus BatchNormV3BlockSplitRTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] =
        WSP_RESERVED_SIZE + usedCoreNum * MEAN_AND_VAR_NODE_NUM * batchNormV3TilingData.get_patternAAlign() * FP32_BYTE;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        batchNormV3TilingData.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
            batchNormV3TilingData.GetDataSize(), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    batchNormV3TilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(batchNormV3TilingData.GetDataSize());
    uint32_t batch_mode = 1U;
    auto ret = context_->SetScheduleMode(batch_mode);

    return ret;
}

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3BlockSplitRTiling, 12000);
} // namespace optiling