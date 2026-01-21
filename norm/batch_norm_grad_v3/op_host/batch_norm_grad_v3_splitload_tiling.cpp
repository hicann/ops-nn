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
#include "op_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "batch_norm_grad_v3_base_tiling.h"

using namespace ge;

namespace optiling {
class BatchNormGradV3SplitLoadTiling : public BatchNormGradV3Base {
    static constexpr int64_t MAX_CHANNEL_SIZE = 1024;
    static constexpr int64_t BLOCK_SIZE = 8;
    static constexpr int64_t TWO = 2;

public:
    explicit BatchNormGradV3SplitLoadTiling(gert::TilingContext* context) : BatchNormGradV3Base(context)
    {}
    ~BatchNormGradV3SplitLoadTiling() override = default;

protected:
    bool IsCapable() override
    {
        if (socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
            return false;
        }

        CalcBasicInfo();

        OP_LOGD(
            context_->GetNodeName(),
            "BatchNormGradV3SplitLoadTiling BAB template is capable, , fused shape: (%ld, %ld, %ld)", fusedB0Len_,
            fusedALen_, fusedB1Len_);
        return true;
    }

    int64_t CalcBubBlock(int64_t tmpChannelNum);

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    BatchNormGradV3SplitLoadTilingData tilingData_;
};

int64_t BatchNormGradV3SplitLoadTiling::CalcBubBlock(int64_t tmpChannelNum)
{
    tmpChannelNum = GetAlignValue(tmpChannelNum);
    int64_t ubTensorRelateChannelBlock = 0;
    int64_t ubTensorNotRelateChannelBlock = 0;
    if (!isTraining) {
        int64_t relateBlockNum = 4;
        int64_t noRelateBlockNum = 3;
        ubTensorRelateChannelBlock = sizeof(float) * relateBlockNum;
        ubTensorNotRelateChannelBlock = tmpChannelNum * noRelateBlockNum * sizeof(float);
    } else {
        int64_t relateBlockNum = 4;
        int64_t noRelateBlockNum = 3;
        ubTensorRelateChannelBlock = sizeof(float) * relateBlockNum;
        ubTensorNotRelateChannelBlock = tmpChannelNum * noRelateBlockNum * sizeof(float);
    }
    return (aicoreParams_.ubSize - ubTensorNotRelateChannelBlock) / ubTensorRelateChannelBlock;
}

ge::graphStatus BatchNormGradV3SplitLoadTiling::DoOpTiling()
{
    int64_t eachCoreChannel = std::ceil((float)fusedALen_ / aicoreParams_.blockDim); // 每个核处理的channel个数
    if (eachCoreChannel == 0) {
        return ge::GRAPH_FAILED;
    }
    needCoreNum = std::ceil((float)fusedALen_ / eachCoreChannel);                   // 需要总核数
    int64_t eachCoreChannelTail = fusedALen_ - (needCoreNum - 1) * eachCoreChannel; // 尾块

    int64_t tmpChannelNum = eachCoreChannel > MAX_CHANNEL_SIZE ? MAX_CHANNEL_SIZE : eachCoreChannel;

    int64_t bUbBlock = CalcBubBlock(tmpChannelNum);
    if (dyDtype_ == ge::DT_FLOAT) {
        bUbBlock = bUbBlock / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
    } else {
        bUbBlock = bUbBlock / HALF_BLOCK_SIZE * HALF_BLOCK_SIZE;
    }
    if (bUbBlock == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t b0UbBlock = 1;
    int64_t bUbLoop = 0;
    int64_t bUbBlockTail = 0;
    int64_t b0UbBlockTail = 0;
    if (bUbBlock < fusedB1Align * TWO) {
        tilingKeyOffset_ += 0;
        b0UbBlock = 1;
        bUbLoop = (fusedB1Len_ + bUbBlock - 1) / bUbBlock;
        bUbBlockTail = fusedB1Len_ - (bUbLoop - 1) * bUbBlock;
    } else if (bUbBlock >= fusedB1Align * TWO) {
        tilingKeyOffset_ += 1;
        b0UbBlock = bUbBlock / fusedB1Align;
        bUbLoop = (fusedB0Len_ + b0UbBlock - 1) / b0UbBlock;
        b0UbBlockTail = fusedB0Len_ - (bUbLoop - 1) * b0UbBlock;
    }

    tilingData_.set_b1Dim(fusedB1Len_);
    tilingData_.set_aDim(fusedALen_);
    tilingData_.set_b0Dim(fusedB0Len_);
    tilingData_.set_bAlign(bAlign);
    tilingData_.set_coreChannelNum(eachCoreChannel);
    tilingData_.set_coreChannelNumTail(eachCoreChannelTail);
    tilingData_.set_bUbBlock(bUbBlock);
    tilingData_.set_b0UbBlock(b0UbBlock);
    tilingData_.set_bUbLoop(bUbLoop);
    tilingData_.set_bUbBlockTail(bUbBlockTail);
    tilingData_.set_b0UbBlockTail(b0UbBlockTail);
    tilingData_.set_needCoreNum(needCoreNum);
    tilingData_.set_epsilon(epsilon_);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradV3SplitLoadTiling::GetTilingKey() const
{
    if (!isTraining) {
        return (TILINGKEY_INFER_SPLITLOAD_BASE + tilingKeyOffset_);
    } else {
        return (TILINGKEY_TRAIN_SPLITLOAD_BASE + tilingKeyOffset_);
    }
}

ge::graphStatus BatchNormGradV3SplitLoadTiling::PostTiling()
{
    context_->SetBlockDim(needCoreNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3SplitLoadTiling, 1100); // traing = false
} // namespace optiling
