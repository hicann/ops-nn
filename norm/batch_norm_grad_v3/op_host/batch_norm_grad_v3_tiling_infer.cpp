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
 * \file batch_norm_grad_v3_tiling_infer.cpp
 * \brief
 */

#include "batch_norm_grad_v3_tiling.h"
#include "op_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "batch_norm_grad_v3_tiling_infer_base.h"

using namespace ge;

namespace optiling {
class BatchNormGradV3InferTiling : public BatchNormGradV3InferBase {
public:
    explicit BatchNormGradV3InferTiling(gert::TilingContext* context) : BatchNormGradV3InferBase(context)
    {}
    ~BatchNormGradV3InferTiling() override = default;

protected:
    bool IsCapable() override
    {
        // B0AB1, B1==1时走BA模板
        OP_TILING_CHECK(
            fusedB1Len_ == 1,
            OP_LOGD(
                context_->GetNodeName(),
                "BatchNormGradV3InferTiling BAB template is not capable, fused shape: (%ld, %ld, %ld)", fusedB0Len_,
                fusedALen_, fusedB1Len_),
            return false);

        CalcBasicInfo();

        OP_LOGD(
            context_->GetNodeName(),
            "BatchNormGradV3InferTiling BAB template is capable, , fused shape: (%ld, %ld, %ld)", fusedB0Len_,
            fusedALen_, fusedB1Len_);
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    BatchNormGradV3InferTilingData tilingData_;
};

ge::graphStatus BatchNormGradV3InferTiling::DoOpTiling()
{
    // 切分A、B基本块， （B0,A,B1） -- >(b0Outer*aOuter*b1Outer, B0inner*Ainner*B1innerA(TileBase))
    int64_t aInner = 1;
    int64_t ubBufferSize =
        (aicoreParams_.ubSize / DOUBLE_BUFFER - (bytesPerWeight_ + bytesPerRunningVar_) * aInner * aTileBase_) /
        bytesPerDy_ / INPUT_OUTPUT_NUM;

    // 先按照B切分，再切A
    // UB可载入最大tile块数
    int64_t factorMax = ubBufferSize / aTileBase_;

    // 默认策略: 先按照B0, B1把UB切满
    int64_t b1FactorMax = Ops::Base::CeilDiv(fusedB1Len_, aTileBase_);
    int64_t b1Inner = factorMax <= b1FactorMax ? factorMax : b1FactorMax;
    int64_t b1Outer = Ops::Base::CeilDiv(fusedB1Len_, b1Inner * aTileBase_);

    factorMax = factorMax / b1Inner;
    int64_t b0FactorMax = fusedB0Len_;
    int64_t b0Inner = factorMax <= b0FactorMax ? factorMax : b0FactorMax;
    int64_t b0Outer = Ops::Base::CeilDiv(fusedB0Len_, b0Inner);

    factorMax = factorMax / b0Inner;
    int64_t aFactorMax = fusedALen_;
    aInner = factorMax <= aFactorMax ? factorMax : aFactorMax;
    int64_t aOuter = Ops::Base::CeilDiv(fusedALen_, aInner);

    int64_t totalTiles = b0Outer * aOuter * b1Outer;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    int64_t tileBlockB0Tail = fusedB0Len_ - b0Inner * (b0Outer - 1);
    int64_t tileBlockATail = fusedALen_ - aInner * (aOuter - 1);
    int64_t tileBlockB1Tail = fusedB1Len_ - b1Inner * aTileBase_ * (b1Outer - 1);

    tilingData_.set_totalTiles(totalTiles);
    tilingData_.set_tilesPerCore(tilesPerCore);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_totalB0Len(fusedB0Len_);
    tilingData_.set_totalALen(fusedALen_);
    tilingData_.set_totalB1Len(fusedB1Len_);
    tilingData_.set_b0Outer(b0Outer);
    tilingData_.set_aOuter(aOuter);
    tilingData_.set_b1Outer(b1Outer);
    tilingData_.set_tileBlockB0Len(b0Inner);
    tilingData_.set_tileBlockB0Tail(tileBlockB0Tail);
    tilingData_.set_tileBlockALen(aInner);
    tilingData_.set_tileBlockATail(tileBlockATail);
    tilingData_.set_tileBlockB1Len(b1Inner * aTileBase_);
    tilingData_.set_tileBlockB1Tail(tileBlockB1Tail);
    tilingData_.set_tileBlockAPaddingNum(0);
    tilingData_.set_epsilon(epsilon_);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradV3InferTiling::GetTilingKey() const
{
    return (TILINGKEY_INFER_BASE);
}

ge::graphStatus BatchNormGradV3InferTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3InferTiling, 91000);
} // namespace optiling
