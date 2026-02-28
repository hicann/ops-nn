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
 * \file batch_norm_grad_v3_tiling_infer.cc
 * \brief
 */

#include "batch_norm_grad_v3_tiling.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "batch_norm_grad_v3_base_tiling.h"

using namespace ge;

namespace optiling {
using namespace Ops::NN::OpTiling;

class BatchNormGradV3SplitLoadCrossCoreTiling : public BatchNormGradV3Base {
    static constexpr int64_t MAX_CHANNEL_SIZE = 1024;
    static constexpr int64_t BLOCK_SIZE = 8;
    static constexpr int64_t TWO = 2;
    static constexpr int64_t CORSSCORE_LIMIT = 20000;
    static constexpr int64_t CORSSCORE_TILINKEY_OFFSET = 10;

public:
    explicit BatchNormGradV3SplitLoadCrossCoreTiling(gert::TilingContext* context) : BatchNormGradV3Base(context)
    {}
    ~BatchNormGradV3SplitLoadCrossCoreTiling() override = default;

protected:
    bool IsCapable() override
    {
        if (IsRegbaseSocVersion(context_)) {
            return false;
        }

        CalcBasicInfo();

        if (bAlign < CORSSCORE_LIMIT) {
            return false;
        }

        if (fusedALen_ % aicoreParams_.numBlocks == 0) {
            return false;
        }

        OP_LOGD(
            context_->GetNodeName(),
            "BatchNormGradV3SplitLoadCrossCoreTiling BAB template is capable, , fused shape: (%ld, %ld, %ld)",
            fusedB0Len_, fusedALen_, fusedB1Len_);
        return true;
    }

    int64_t CalcBubBlock(int64_t tmpChannelNum);

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    bool CalcGroupInfo(int64_t tmpALen_, int64_t allBlockNum);
    bool CalcSplitHwTilingData(int64_t tmpALen_);
    bool CalcSplitNTilingData(int64_t tmpALen_);
    void FillTilingData();

private:
    BatchNormGradV3SplitLoadCrossCoreTilingData tilingData_;
    int64_t morehalfChannel = 0;
    int64_t moreMultiChannel = 0;

    int64_t eachCoreChannel = 0;

    int64_t bUbBlock = 0;
    int64_t b0UbBlock = 1;
    int64_t bUbLoop = 0;
    int64_t bUbBlockTail = 0;
    int64_t b0UbBlockTail = 0;
    int64_t bUbTailLoop = 0;
    int64_t b0UbTailBlockTail = 0;

    // 如果HW较大场景，搬运可以连续搬运，建议每个C进行分给少的核处理
    int64_t groupCore = 0;
    int64_t groupBlockNum = 0;
    int64_t groupBlockNumTail = 0;

    int64_t bUbLoopHalf = 0;
    int64_t b0UbBlockTailHalf = 0;
    int64_t bUbTailLoopHalf = 0;
    int64_t b0UbTailBlockTailHalf = 0;
    int64_t groupBlockNumHalf = 0;
    int64_t groupBlockNumHalfTail = 0;

    int64_t bUbLoopMulti = 0;
    int64_t b0UbBlockTailMulti = 0;
};

int64_t BatchNormGradV3SplitLoadCrossCoreTiling::CalcBubBlock(int64_t tmpChannelNum)
{
    tmpChannelNum = GetAlignValue(tmpChannelNum);
    int64_t ubTensorRelateChannelBlock = 0;
    int64_t ubTensorNotRelateChannelBlock = 0;
    int64_t relateBlockNum = 4;
    int64_t noRelateBlockNum = 3;
    ubTensorRelateChannelBlock = sizeof(float) * relateBlockNum;
    ubTensorNotRelateChannelBlock = tmpChannelNum * noRelateBlockNum * sizeof(float);
    return (aicoreParams_.ubSize - ubTensorNotRelateChannelBlock) / ubTensorRelateChannelBlock;
}

bool BatchNormGradV3SplitLoadCrossCoreTiling::CalcGroupInfo(int64_t tmpALen_, int64_t allBlockNum)
{
    if (tmpALen_ == 0) {
        return false;
    }
    // 对核按照C分组，如果每组核数超过总分块数，则取每组核数为总分块数
    groupCore = aicoreParams_.numBlocks / tmpALen_;
    groupCore = groupCore > allBlockNum ? allBlockNum : groupCore;
    if (groupCore == 0) {
        return false;
    }
    // 计算每个核计算的block数量
    groupBlockNum = std::ceil((float)allBlockNum / groupCore);
    if (groupBlockNum == 0) {
        return false;
    }
    groupCore = std::ceil((float)allBlockNum / groupBlockNum);
    groupBlockNumTail = allBlockNum - (groupCore - 1) * groupBlockNum;
    // 计算需要的总核数
    needCoreNum = groupCore * tmpALen_;
    return true;
}

bool BatchNormGradV3SplitLoadCrossCoreTiling::CalcSplitHwTilingData(int64_t tmpALen_)
{
    if (tmpALen_ == 0) {
        return false;
    }
    tilingKeyOffset_ += 0;
    b0UbBlock = 1;
    bUbLoop = (fusedB1Len_ + bUbBlock - 1) / bUbBlock; // 每个HW在核内循环次数
    bUbBlockTail = fusedB1Len_ - (bUbLoop - 1) * bUbBlock;
    bUbLoopMulti = bUbLoop;
    b0UbBlockTailMulti = bUbBlockTail;

    // 每个C的总分块数
    auto allBlockNum = bUbLoop * fusedB0Len_;

    int64_t coreMultiple = aicoreParams_.numBlocks / tmpALen_;
    if (allBlockNum > TWO && coreMultiple < TWO) {
        // C > 20 || C > 24， 前20个C每两个核处理，后面再走均分逻辑
        morehalfChannel = 1;
        tmpALen_ -= aicoreParams_.numBlocks / TWO;

        groupCore = TWO;
        // 计算每个核计算的block数量
        groupBlockNumHalf = std::ceil((float)allBlockNum / groupCore);
        groupBlockNumHalfTail = allBlockNum - (groupCore - 1) * groupBlockNumHalf;
    }
    return CalcGroupInfo(tmpALen_, allBlockNum);
}

bool BatchNormGradV3SplitLoadCrossCoreTiling::CalcSplitNTilingData(int64_t tmpALen_)
{
    // 如果HW较小，C也较小，应该分给越多核处理越合适
    tilingKeyOffset_ += 1;
    b0UbBlock = bUbBlock / fusedB1Align;
    bUbLoopMulti = (fusedB0Len_ + b0UbBlock - 1) / b0UbBlock;
    b0UbBlockTailMulti = fusedB0Len_ - (bUbLoopMulti - 1) * b0UbBlock;

    // 每个C的总分块数
    auto allBlockNum = fusedB0Len_;
    if (tmpALen_ == 0) {
        return false;
    }

    int64_t coreMultiple = aicoreParams_.numBlocks / tmpALen_;
    if (allBlockNum > TWO && coreMultiple < TWO) {
        // C > 20 || C > 24， 前20个C每两个核处理，后面再走均分逻辑
        morehalfChannel = 1;
        tmpALen_ -= aicoreParams_.numBlocks / TWO;

        groupCore = TWO;
        // 计算每个核计算的block数量
        groupBlockNumHalf = std::ceil((float)allBlockNum / groupCore);
        groupBlockNumHalfTail = allBlockNum - (groupCore - 1) * groupBlockNumHalf;

        bUbLoopHalf = (groupBlockNumHalf + b0UbBlock - 1) / b0UbBlock;
        b0UbBlockTailHalf = groupBlockNumHalf - (bUbLoopHalf - 1) * b0UbBlock;

        bUbTailLoopHalf = (groupBlockNumHalfTail + b0UbBlock - 1) / b0UbBlock;
        b0UbTailBlockTailHalf = groupBlockNumHalfTail - (bUbTailLoopHalf - 1) * b0UbBlock;
    }
    if (!CalcGroupInfo(tmpALen_, allBlockNum)) {
        return false;
    }
    bUbLoop = (groupBlockNum + b0UbBlock - 1) / b0UbBlock;
    b0UbBlockTail = groupBlockNum - (bUbLoop - 1) * b0UbBlock;
    bUbTailLoop = (groupBlockNumTail + b0UbBlock - 1) / b0UbBlock;
    b0UbTailBlockTail = groupBlockNumTail - (bUbTailLoop - 1) * b0UbBlock;
    return true;
}

ge::graphStatus BatchNormGradV3SplitLoadCrossCoreTiling::DoOpTiling()
{
    eachCoreChannel = fusedALen_ / aicoreParams_.numBlocks; // 每个核处理的channel个数

    bUbBlock = CalcBubBlock(eachCoreChannel);
    if (dyDtype_ == ge::DT_FLOAT) {
        bUbBlock = bUbBlock / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
    } else {
        bUbBlock = bUbBlock / HALF_BLOCK_SIZE * HALF_BLOCK_SIZE;
    }
    if (bUbBlock == 0) {
        return ge::GRAPH_FAILED;
    }

    tilingKeyOffset_ += CORSSCORE_TILINKEY_OFFSET;

    int64_t tmpALen_ = fusedALen_;

    if (fusedALen_ > aicoreParams_.numBlocks) {
        moreMultiChannel = 1;
        tmpALen_ -= eachCoreChannel * aicoreParams_.numBlocks;
    }

    if (bUbBlock < fusedB1Align * TWO) {
        CalcSplitHwTilingData(tmpALen_);
    } else if (bUbBlock >= fusedB1Align * TWO) {
        CalcSplitNTilingData(tmpALen_);
    }

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

void BatchNormGradV3SplitLoadCrossCoreTiling::FillTilingData()
{
    tilingData_.set_b1Dim(fusedB1Len_);
    tilingData_.set_aDim(fusedALen_);
    tilingData_.set_b0Dim(fusedB0Len_);
    tilingData_.set_bAlign(bAlign);
    tilingData_.set_coreChannelNum(eachCoreChannel);
    tilingData_.set_bUbBlock(bUbBlock);
    tilingData_.set_b0UbBlock(b0UbBlock);
    tilingData_.set_bUbLoop(bUbLoop);
    tilingData_.set_bUbBlockTail(bUbBlockTail);
    tilingData_.set_b0UbBlockTail(b0UbBlockTail);
    tilingData_.set_needCoreNum(needCoreNum);
    tilingData_.set_groupCore(groupCore);
    tilingData_.set_groupBlockNum(groupBlockNum);
    tilingData_.set_groupBlockNumTail(groupBlockNumTail);
    tilingData_.set_morehalfChannel(morehalfChannel);
    tilingData_.set_groupBlockNumHalf(groupBlockNumHalf);
    tilingData_.set_groupBlockNumHalfTail(groupBlockNumHalfTail);
    tilingData_.set_bUbTailLoop(bUbTailLoop);
    tilingData_.set_b0UbTailBlockTail(b0UbTailBlockTail);
    tilingData_.set_bUbLoopHalf(bUbLoopHalf);
    tilingData_.set_b0UbBlockTailHalf(b0UbBlockTailHalf);
    tilingData_.set_bUbTailLoopHalf(bUbTailLoopHalf);
    tilingData_.set_b0UbTailBlockTailHalf(b0UbTailBlockTailHalf);
    tilingData_.set_moreMultiChannel(moreMultiChannel);
    tilingData_.set_bUbLoopMulti(bUbLoopMulti);
    tilingData_.set_b0UbBlockTailMulti(b0UbBlockTailMulti);
    tilingData_.set_coreNum(aicoreParams_.numBlocks);
    tilingData_.set_epsilon(epsilon_);
}

uint64_t BatchNormGradV3SplitLoadCrossCoreTiling::GetTilingKey() const
{
    if (!isTraining) {
        return (TILINGKEY_INFER_SPLITLOAD_BASE + tilingKeyOffset_);
    } else {
        return (TILINGKEY_TRAIN_SPLITLOAD_BASE + tilingKeyOffset_);
    }
}

ge::graphStatus BatchNormGradV3SplitLoadCrossCoreTiling::PostTiling()
{
    int64_t coreNum = moreMultiChannel || morehalfChannel ? aicoreParams_.numBlocks : needCoreNum;
    context_->SetBlockDim(coreNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    auto tmpWsSize = coreNum * 8 * sizeof(float) * 2;
    auto needWSSize = workspaceSize_ + tmpWsSize;
    currentWorkspace[0] = needWSSize;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3SplitLoadCrossCoreTiling, 1050); // traing = false
} // namespace optiling
