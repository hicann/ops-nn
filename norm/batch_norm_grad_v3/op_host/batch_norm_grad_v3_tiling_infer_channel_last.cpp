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
 * \file batch_norm_grad_v3_tiling_infer_channel_last.cpp
 * \brief
 */

#include "batch_norm_grad_v3_tiling.h"
#include "op_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "batch_norm_grad_v3_tiling_infer_base.h"

using namespace ge;

namespace optiling {
class BatchNormGradV3InferChannelLastTiling : public BatchNormGradV3InferBase {
public:
    explicit BatchNormGradV3InferChannelLastTiling(gert::TilingContext* context) : BatchNormGradV3InferBase(context)
    {}
    ~BatchNormGradV3InferChannelLastTiling() override = default;

protected:
    bool IsCapable() override
    {
        OP_TILING_CHECK(
            fusedB1Len_ != 1,
            OP_LOGD(
                context_->GetNodeName(),
                "BatchNormGradV3InferChannelLastTiling BA template is not capable, fused shape: (%ld, %ld, %ld)",
                fusedB0Len_, fusedALen_, fusedB1Len_),
            return false);

        CalcBasicInfo();

        OP_LOGD(
            context_->GetNodeName(),
            "BatchNormGradV3InferChannelLastTiling BA template is capable, fused shape: (%ld, %ld, %ld)", fusedB0Len_,
            fusedALen_, fusedB1Len_);
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    BatchNormGradV3InferChannelLastTilingData tilingData_;
};

ge::graphStatus BatchNormGradV3InferChannelLastTiling::DoOpTiling()
{
    // 切分A、B基本块， （B,A） -- >(Bouter, Aouter, Binner*Ainner*aTileBase_)
    int64_t aInner = 1;
    int64_t ubBufferSize =
        (aicoreParams_.ubSize / DOUBLE_BUFFER - (bytesPerWeight_ + bytesPerRunningVar_) * aInner * aTileBase_) /
        bytesPerDy_ / INPUT_OUTPUT_NUM;

    // 先按照B切分，再切A
    int64_t bFactorMax = ubBufferSize / aTileBase_;
    int64_t bInner = fusedB0Len_ <= bFactorMax ? fusedB0Len_ : bFactorMax;
    int64_t bOuter = Ops::Base::CeilDiv(fusedB0Len_, bInner);
    int64_t bTail = fusedB0Len_ % bInner;
    int64_t tileBlockBTail = bTail == 0 ? bInner : bTail;

    int64_t aFactorMax = aicoreParams_.ubSize / DOUBLE_BUFFER / aTileBase_ /
                         (bInner * INPUT_OUTPUT_NUM * bytesPerDy_ + bytesPerWeight_ + bytesPerRunningVar_);
    int64_t aInnerMax = fusedALen_ / aTileBase_;
    aInner = aInnerMax <= aFactorMax ? aInnerMax : aFactorMax;

    int64_t tileBlockALen = aInner == 0 ? aTileBase_ : aInner * aTileBase_;
    int64_t aOuter = Ops::Base::CeilDiv(fusedALen_, tileBlockALen);
    int64_t aTail = fusedALen_ % tileBlockALen;
    int64_t tileBlockATail = aTail == 0 ? tileBlockALen : aTail;
    int64_t tileBlockAPaddingNum = tileBlockALen - tileBlockATail;

    // 切核 （Bouter, Binner, Aouter, Ainner*aTileBase_） -- > (Bouter*Aouter, Binner, Ainner*aTileBase_)
    int64_t totalTiles = aOuter * bOuter;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    tilingData_.set_totalTiles(totalTiles);
    tilingData_.set_tilesPerCore(tilesPerCore);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_totalALen(fusedALen_);
    tilingData_.set_aOuter(aOuter);
    tilingData_.set_bOuter(bOuter);
    tilingData_.set_tileBlockALen(tileBlockALen);
    tilingData_.set_tileBlockATail(tileBlockATail);
    tilingData_.set_tileBlockAPaddingNum(tileBlockAPaddingNum);
    tilingData_.set_tileBlockBLen(bInner);
    tilingData_.set_tileBlockBTail(tileBlockBTail);
    tilingData_.set_epsilon(epsilon_);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradV3InferChannelLastTiling::GetTilingKey() const
{
    return (TILINGKEY_INFER_CHANNEL_LAST_BASE);
}

ge::graphStatus BatchNormGradV3InferChannelLastTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3InferChannelLastTiling, 90000);
} // namespace optiling
