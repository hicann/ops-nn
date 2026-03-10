/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_v3_tiling_rar_block_split_r_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "batch_norm_v3_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_RAR_BLOCK_SPLIT_R = 250000;
constexpr int64_t PRIORITY_RAR_BLOCK_SPLIT_R = 25000;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t BLOCK_SPLIT_R0_INDEX = 2;
constexpr int64_t BLOCK_SPLIT_R1_INDEX = 0;
constexpr int64_t UP_SPLIT_R0_INDEX = 2;
constexpr int64_t RUNNING_MEAN_INPUT_IDX = 3;

constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t MEAN_VAR_BUFFER_NUM = 2;
constexpr int64_t RUNNING_MEAN_VAR_BUFFER_NUM = 4;
constexpr int64_t BETA_GAMMA_BUFFER_NUM = 2;

constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t WSP_RESERVED_SIZE = 16L * 1024L * 1024L;
constexpr int64_t MEAN_AND_VAR_NODE_NUM = 2;
constexpr int64_t FIRST_VCADD_RESULT_MAX_NUM = 128;
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
class BatchNormV3RARBlockSplitRTiling : public BatchNormV3RegbaseTilingBase {
public:
    explicit BatchNormV3RARBlockSplitRTiling(gert::TilingContext* context) : BatchNormV3RegbaseTilingBase(context)
    {}
    ~BatchNormV3RARBlockSplitRTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormV3RegbaseTilingBase::Reset(context);
        usedCoreNum = 0;
    }

protected:
    bool IsCapable() override
    {
        OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling IsCapable: enter IsCapable function to judge.");
        if (format != FORMAT_NCHW && format != FORMAT_NCDHW) {
            return false;
        }
        if (a * NUM_TWO >= static_cast<int64_t>(aicoreParams_.numBlocks)) {
            OP_LOGD(context_->GetNodeName(),
                "BatchNormV3RARBlockSplitRTiling IsCapable info: a * NUM_TWO:%lu, aicoreParams_.numBlocks: %lu.", a * NUM_TWO, aicoreParams_.numBlocks);
            return false;
        }
        OP_LOGD(context_->GetNodeName(),
            "BatchNormV3RARBlockSplitRTiling IsCapable true! ready to use BatchNormV3RARBlockSplitRTiling (tilingkey: %lu) !",
            TILINGKEY_RAR_BLOCK_SPLIT_R);
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
    int64_t UbSplit(int64_t r1BlockInner, int64_t r0BlockInner, int64_t ubFactor, int64_t& ubSplitAxis, int64_t& ubOuter, int64_t& ubInner, int64_t eleNumPerBlock);
private:
    int64_t usedCoreNum;
    BatchNormV3RARBlockSplitRTilingData batchNormV3TilingData;
};

void BatchNormV3RARBlockSplitRTiling::SetInputInfo()
{
    // dim
    batchNormV3TilingData.set_patternR1(r1);
    batchNormV3TilingData.set_patternA(a);
    batchNormV3TilingData.set_patternR0(r0);

    // attr
    batchNormV3TilingData.set_epsilon(epsilon);
    batchNormV3TilingData.set_momentum(momentum);
    batchNormV3TilingData.set_momentumReverse(1 - momentum);
}

bool BatchNormV3RARBlockSplitRTiling::BinaryAddTiling(
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
        OP_LOGD(context_->GetNodeName(), "BinaryAddTiling binaryAddNum %ld case not supported", binaryAddNum);
        return false;
    }
    return true;
}

int64_t BatchNormV3RARBlockSplitRTiling::UbSplit(int64_t r1BlockInner, int64_t r0BlockInner, int64_t ubFactor, 
                                                 int64_t& ubSplitAxis, int64_t& ubOuter, int64_t& ubInner, int64_t eleNumPerBlock)
{
    int64_t r0BlockInnerAlign = Ops::Base::CeilAlign(r0BlockInner, eleNumPerBlock);
    if (ubFactor <= r0BlockInnerAlign) {
        // 1. ub只切分到R0轴

        ubSplitAxis = UP_SPLIT_R0_INDEX;
        ubInner = Ops::Base::FloorAlign(ubFactor, eleNumPerBlock); // 考虑r0需要对齐，避免ub用超
        ubOuter = Ops::Base::CeilDiv(r0BlockInner, ubInner);
        ubInner = Ops::Base::CeilDiv(r0BlockInner, ubOuter);
 	    ubInner = Ops::Base::CeilAlign(ubInner, eleNumPerBlock);

        return ubInner;
    } else if (ubFactor <= r0BlockInnerAlign * a || r1BlockInner == 1) {
        // 2. A轴参与切分
        // 考虑r0需要对齐，且ubFactor大小的限制（ubFactor如果对于r0BlockInner对齐后的空间不够的话，ub切分轴需要在a轴）

        ubSplitAxis = 1;
        ubInner = std::min(Ops::Base::FloorDiv(ubFactor, r0BlockInnerAlign), a); // 考虑r0需要对齐，避免ub用超
        ubOuter = Ops::Base::CeilDiv(a, ubInner);  

        return r0BlockInner;
    } else {
        // 3. ub切分到R1轴

        ubSplitAxis = 0;
        // 计算在ubFactor限制下，最大的对齐R空间，并对齐到32B
        int64_t maxAlignedRNum = Ops::Base::FloorDiv(ubFactor, a);
        maxAlignedRNum = Ops::Base::FloorAlign(maxAlignedRNum, eleNumPerBlock);

        // 继续计算上述32B对齐的最大R可用空间对应的ubInner
        int64_t maxUbInnerFromAligned = Ops::Base::FloorDiv(maxAlignedRNum, r0BlockInner);

        ubInner = std::min(maxUbInnerFromAligned, r1BlockInner); 

        // 确保至少要处理1个元素
        ubInner = ubInner > 1 ? ubInner : 1;

        ubOuter = Ops::Base::CeilDiv(r1BlockInner, ubInner);
        ubInner = Ops::Base::CeilDiv(r1BlockInner, ubOuter);

        return ubInner * r0BlockInner;
    }
}

ge::graphStatus BatchNormV3RARBlockSplitRTiling::DoOpTiling()
{
    SetInputInfo();

    int64_t blockSplitAxis = 0;
    int64_t formerBlockOuter = 0;
    int64_t tailBlockOuter = 0;
    int64_t blockInner = 0;

    // 多核切分R轴：采用均分方式，划分formerCore和tailCore
    if (r1 < static_cast<int64_t>(aicoreParams_.numBlocks)) {
        // R0参与多核切分
        blockSplitAxis = BLOCK_SPLIT_R0_INDEX;
        int64_t blockOuter = std::min(Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.numBlocks), r1), r0);
        blockInner = Ops::Base::CeilDiv(r0, blockOuter);
        tailBlockOuter = blockInner * blockOuter - r0;
        formerBlockOuter = blockOuter - tailBlockOuter;
        usedCoreNum = r1 * blockOuter;
    } else {
        // R1即满足多核
        blockSplitAxis = BLOCK_SPLIT_R1_INDEX;
        blockInner = Ops::Base::CeilDiv(r1, static_cast<int64_t>(aicoreParams_.numBlocks));
        int64_t blockOuter = Ops::Base::CeilDiv(r1, blockInner);
        tailBlockOuter = blockInner * blockOuter - r1;
        formerBlockOuter = blockOuter - tailBlockOuter;
        usedCoreNum = blockOuter;
    }

    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: blockSplitAxis is %lu,"
            " formerBlockOuter is %lu, tailBlockOuter is %lu, blockInner is %lu, usedCoreNum is %lu",
            blockSplitAxis, formerBlockOuter, tailBlockOuter, blockInner, usedCoreNum);

    // 计算ubfactor
    // Step1: ubsize减去固定ubffer大小部分，得到ubSizeCanUse
    int64_t elemSize = FP32_BYTE;
    int64_t weightElemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
    }
    if (weightDataType == ge::DT_FLOAT16 || weightDataType == ge::DT_BF16) {
        weightElemSize = FP16_BYTE;
    }
    auto runningMeanDesc = context_->GetInputDesc(RUNNING_MEAN_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningMeanDesc);
    auto runningMeanDataType = runningMeanDesc->GetDataType();
    int64_t runningMeanElemSize = FP32_BYTE;
    if (runningMeanDataType == ge::DT_FLOAT16 || runningMeanDataType == ge::DT_BF16) {
        runningMeanElemSize = FP16_BYTE;
    }

    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: class member blockSize is %lu !", blockSize);
    int64_t fp32EleNumPerBlock = blockSize / FP32_BYTE;
    int64_t patternAAlign = Ops::Base::CeilAlign(a, fp32EleNumPerBlock);

    int64_t runningMeanVarSize = Ops::Base::CeilAlign(a, blockSize / runningMeanElemSize) * runningMeanElemSize * RUNNING_MEAN_VAR_BUFFER_NUM;
    int64_t saveMeanRstdSize =  patternAAlign * static_cast<int64_t>(sizeof(float)) * MEAN_VAR_BUFFER_NUM;
    int64_t betaGammaSize =  Ops::Base::CeilAlign(a, blockSize / weightElemSize) * weightElemSize * BETA_GAMMA_BUFFER_NUM;

    int64_t blockDimAlignSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(aicoreParams_.numBlocks), fp32EleNumPerBlock) * FP32_BYTE;
    int64_t tmpbuffer3AlignSize =
        std::max(static_cast<int64_t>(aicoreParams_.numBlocks) * patternAAlign,
                 static_cast<int64_t>(Ops::Base::CeilAlign(FIRST_VCADD_RESULT_MAX_NUM, fp32EleNumPerBlock))) * FP32_BYTE;
    
    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: runningMeanVarSize is %lu,"
            " saveMeanRstdSize is %lu, betaGammaSize is %lu, blockDimAlignSize is %lu, tmpbuffer3AlignSize is %lu",
            runningMeanVarSize, saveMeanRstdSize, betaGammaSize, blockDimAlignSize, tmpbuffer3AlignSize);
    
    int64_t xSizePerUbFactor = elemSize * DOUBLE_BUFFER_NUM;
    int64_t ySizePerUbFactor = elemSize * DOUBLE_BUFFER_NUM;
    int64_t tmpMeanVarPerUbFactor = static_cast<int64_t>(sizeof(float)) * MEAN_VAR_BUFFER_NUM;
    int64_t countBuffer1PerUbFactor = static_cast<int64_t>(sizeof(float));

    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: xSizePerUbFactor is %lu,"
            " ySizePerUbFactor is %lu, tmpMeanVarPerUbFactor is %lu, countBuffer1PerUbFactor is %lu",
            xSizePerUbFactor, ySizePerUbFactor, tmpMeanVarPerUbFactor, countBuffer1PerUbFactor);

    int64_t ubSizeCanUse = static_cast<int64_t>(aicoreParams_.ubSize) - runningMeanVarSize - saveMeanRstdSize - betaGammaSize - blockDimAlignSize - tmpbuffer3AlignSize;
    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: ubSizeCanUse is %lu !", ubSizeCanUse);
    OP_CHECK_IF(
        ubSizeCanUse <= 0, OP_LOGI(context_->GetNodeName(), "ubSizeCanUse is not a positive number."),
        return ge::GRAPH_PARAM_INVALID);

    // Step2: 根据ubSizeCanUse，算出ubFactor，并进行对齐操作
    int64_t ubFactor = ubSizeCanUse / (xSizePerUbFactor + ySizePerUbFactor + tmpMeanVarPerUbFactor + countBuffer1PerUbFactor);

    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: direct obtained ubFactor is %lu !", ubFactor);

    ubFactor = Ops::Base::FloorAlign(ubFactor, blockSize / elemSize);

    OP_CHECK_IF(ubFactor == 0, OP_LOGI(context_->GetNodeName(), "ubFactor is 0."), return ge::GRAPH_PARAM_INVALID);
    batchNormV3TilingData.set_ubFactor(ubFactor);
    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: ubFactor is %lu !\n", ubFactor);

    // 根据ubFactor计算ub切分参数
    int64_t formerCoreR1BlockInner = 0;
    int64_t formerCoreR0BlockInner = 0;
    int64_t tailCoreR1BlockInner = 0;
    int64_t tailCoreR0BlockInner = 0;

    int64_t formerCoreUbSplitAxis = 0;
    int64_t formerCoreUbOuter = 0;
    int64_t formerCoreUbInner = 0;
    int64_t tailCoreUbSplitAxis = 0;
    int64_t tailCoreUbOuter = 0;
    int64_t tailCoreUbInner = 0;

    if (blockSplitAxis == BLOCK_SPLIT_R1_INDEX) {
        formerCoreR1BlockInner = blockInner;
        tailCoreR1BlockInner = blockInner - 1;
        formerCoreR0BlockInner = r0;
        tailCoreR0BlockInner = r0;
    } else {
        formerCoreR1BlockInner = 1;
        tailCoreR1BlockInner = 1;
        formerCoreR0BlockInner = blockInner;
        tailCoreR0BlockInner = blockInner - 1;
    }
    int64_t formerCoreProcessRNum = 0;
    int64_t tailCoreProcessRNum = 0;
    formerCoreProcessRNum = 
        UbSplit(formerCoreR1BlockInner, formerCoreR0BlockInner, ubFactor, formerCoreUbSplitAxis, formerCoreUbOuter, formerCoreUbInner, blockSize / elemSize);
    tailCoreProcessRNum = 
        UbSplit(tailCoreR1BlockInner, tailCoreR0BlockInner, ubFactor, tailCoreUbSplitAxis, tailCoreUbOuter, tailCoreUbInner, blockSize / elemSize);

    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: formerCoreUbSplitAxis is %lu,"
            " formerCoreUbOuter is %lu, formerCoreUbInner is %lu, tailCoreUbSplitAxis is %lu, tailCoreUbOuter is %lu, tailCoreUbInner is %lu, "
            " formerCoreProcessRNum is %lu, tailCoreProcessRNum is %lu", formerCoreUbSplitAxis, formerCoreUbOuter, formerCoreUbInner, tailCoreUbSplitAxis,
            tailCoreUbOuter, tailCoreUbInner, formerCoreProcessRNum, tailCoreProcessRNum);

    int64_t formerCoreBinaryAddQuotient = FindBinaryQuotient(formerCoreProcessRNum);
    int64_t tailCoreBinaryAddQuotient = FindBinaryQuotient(tailCoreProcessRNum);

    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling: formerCoreBinaryAddQuotient is %lu, tailCoreBinaryAddQuotient is %lu",
            formerCoreBinaryAddQuotient, tailCoreBinaryAddQuotient);

    int64_t lastBinaryAddQuotient = FindBinaryQuotient(usedCoreNum);
    int64_t lastBinaryAddK = 0;
    int64_t lastBinaryAddLast = 0;
    auto res = BinaryAddTiling(lastBinaryAddQuotient, lastBinaryAddK, lastBinaryAddLast);
    OP_CHECK_IF(
        res == false,
        OP_LOGI(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling BinaryAddTiling param invalid"),
        return ge::GRAPH_PARAM_INVALID);

    batchNormV3TilingData.set_patternAAlign(patternAAlign);
    batchNormV3TilingData.set_blockSplitAxis(blockSplitAxis);
    batchNormV3TilingData.set_formerBlockOuter(formerBlockOuter);
    batchNormV3TilingData.set_tailBlockOuter(tailBlockOuter);
    batchNormV3TilingData.set_blockInner(blockInner);
    batchNormV3TilingData.set_formerCoreUbSplitAxis(formerCoreUbSplitAxis);
    batchNormV3TilingData.set_formerCoreUbOuter(formerCoreUbOuter);
    batchNormV3TilingData.set_formerCoreUbInner(formerCoreUbInner);
    batchNormV3TilingData.set_tailCoreUbSplitAxis(tailCoreUbSplitAxis);
    batchNormV3TilingData.set_tailCoreUbOuter(tailCoreUbOuter);
    batchNormV3TilingData.set_tailCoreUbInner(tailCoreUbInner);
    batchNormV3TilingData.set_formerCoreBinaryAddQuotient(formerCoreBinaryAddQuotient);
    batchNormV3TilingData.set_tailCoreBinaryAddQuotient(tailCoreBinaryAddQuotient);
    batchNormV3TilingData.set_lastBinaryAddQuotient(lastBinaryAddQuotient);
    batchNormV3TilingData.set_lastBinaryAddK(lastBinaryAddK);
    batchNormV3TilingData.set_lastBinaryAddLast(lastBinaryAddLast);

    OP_LOGD(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling DoOpTiling SUCCESS!");
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormV3RARBlockSplitRTiling::GetTilingKey() const
{
    OP_LOGI(context_->GetNodeName(), "BatchNormV3RARBlockSplitRTiling GetTilingKey tilingkey is %lu !", TILINGKEY_RAR_BLOCK_SPLIT_R);
    return TILINGKEY_RAR_BLOCK_SPLIT_R;
}

ge::graphStatus BatchNormV3RARBlockSplitRTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = WSP_RESERVED_SIZE + usedCoreNum * MEAN_AND_VAR_NODE_NUM * batchNormV3TilingData.get_patternAAlign() * FP32_BYTE;
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

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3RARBlockSplitRTiling, PRIORITY_RAR_BLOCK_SPLIT_R);
} // namespace optiling