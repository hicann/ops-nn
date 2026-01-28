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
#include "tiling_base/tiling_util.h"
#include "batch_norm_grad_v3_base_tiling.h"

using namespace ge;

namespace optiling {
using namespace Ops::NN::OpTiling;

class BatchNormGradV3FullLoadTiling : public BatchNormGradV3Base {
public:
    explicit BatchNormGradV3FullLoadTiling(gert::TilingContext* context) : BatchNormGradV3Base(context)
    {}
    ~BatchNormGradV3FullLoadTiling() override = default;

protected:
    bool IsCapable() override
    {
        if (IsRegbaseSocVersion(context_)) {
            return false;
        }

        CalcBasicInfo();

        if (isTraining && bAlign > fullloadTrainLimit) {
            return false;
        } else if (!isTraining && bAlign > fullloadInferLimit) {
            return false;
        }

        OP_LOGD(
            context_->GetNodeName(),
            "BatchNormGradV3FullLoadTiling BAB template is capable, , fused shape: (%ld, %ld, %ld)", fusedB0Len_,
            fusedALen_, fusedB1Len_);
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    BatchNormGradV3FullLoadTilingData tilingData_;
    const int64_t fullloadTrainLimit = 12000;
    const int64_t fullloadInferLimit = 16000;
};

ge::graphStatus BatchNormGradV3FullLoadTiling::DoOpTiling()
{
    uint64_t eachCoreChannel = std::ceil((float)fusedALen_ / aicoreParams_.blockDim); // 每个核处理的channel个数
    if (eachCoreChannel == 0) {
        return ge::GRAPH_FAILED;
    }
    needCoreNum = std::ceil((float)fusedALen_ / eachCoreChannel);                    // 需要总核数
    uint64_t eachCoreChannelTail = fusedALen_ - (needCoreNum - 1) * eachCoreChannel; // 尾块

    uint64_t ubTensorRelateChannelBlock = 0;
    uint64_t ubTensorNotRelateChannelBlock = 0;
    if (!isTraining) {
        uint64_t noRelateBlockNum = 9 * 8 * 4;
        uint64_t relateBlockNum = 9 * 3;
        uint64_t relateAlignBlockNum = 3;
        ubTensorRelateChannelBlock = bAlign * sizeof(float) * relateAlignBlockNum + relateBlockNum * sizeof(float);
        ubTensorNotRelateChannelBlock = noRelateBlockNum * sizeof(float);
    } else {
        uint64_t noRelateBlockNum = 9 * 8 * 5;
        uint64_t relateBlockNum = 9 * 4;
        uint64_t relateAlignBlockNum = 4;
        ubTensorRelateChannelBlock = bAlign * sizeof(float) * relateAlignBlockNum + relateBlockNum * sizeof(float);
        ubTensorNotRelateChannelBlock = noRelateBlockNum * sizeof(float);
    }
    uint64_t cUbBlock = (aicoreParams_.ubSize - ubTensorNotRelateChannelBlock) / ubTensorRelateChannelBlock;

    tilingData_.set_b1Dim(fusedB1Len_);
    tilingData_.set_aDim(fusedALen_);
    tilingData_.set_b0Dim(fusedB0Len_);
    tilingData_.set_bAlign(bAlign);
    tilingData_.set_coreChannelNum(eachCoreChannel);
    tilingData_.set_coreChannelNumTail(eachCoreChannelTail);
    tilingData_.set_cUbBlock(cUbBlock);
    tilingData_.set_needCoreNum(needCoreNum);
    tilingData_.set_epsilon(epsilon_);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradV3FullLoadTiling::GetTilingKey() const
{
    if (!isTraining) {
        return (TILINGKEY_INFER_FULLLOAD_BASE);
    } else {
        return (TILINGKEY_TRAIN_FULLLOAD_BASE);
    }
}

ge::graphStatus BatchNormGradV3FullLoadTiling::PostTiling()
{
    context_->SetBlockDim(needCoreNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3FullLoadTiling, 1000); // traing = false
} // namespace optiling
