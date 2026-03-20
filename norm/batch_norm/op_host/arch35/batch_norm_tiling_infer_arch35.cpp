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
 * \file batch_norm_tiling_infer_arch35.cpp
 * \brief
 */
#include "batch_norm_tiling.h"

using namespace ge;

namespace
{
constexpr int64_t TILINGKEY_INFER = 910000;
constexpr int64_t SMALL_SHAPE_NUM = 6;  // scale, offset, mean, var, outMean, outVar
constexpr int64_t BIG_SHAPE_NUM = 2;    // x, y
}  // namespace

namespace optiling
{
class BatchNormInferTiling : public BatchNormTilingInferBase
{
public:
    explicit BatchNormInferTiling(gert::TilingContext* context) : BatchNormTilingInferBase(context)
    {
    }
    ~BatchNormInferTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;

private:
    const char* opName = "BatchNormInfer";

    BatchNormInferTilingData tilingData_;
};

ge::graphStatus BatchNormInferTiling::DoOpTiling()
{
    aTileBase_ = vlFp16_;
    bytesPerElement_ = FLOAT16_BYTES;
    if (xDtype_ == ge::DT_FLOAT) {
        aTileBase_ = vlFp32_;
        bytesPerElement_ = FLOAT32_BYTES;
    }
    // 切分A、B基本块， （B0,A,B1） -- >(b0Outer*aOuter*b1Outer, B0inner*Ainner*B1innerA(TileBase))
    int64_t aInner = 1;
    int64_t ubBufferSize =
        (aicoreParams_.ubSize / DOUBLE_BUFFER - SMALL_SHAPE_NUM * FLOAT32_BYTES * aInner * aTileBase_) /
        bytesPerElement_ / BIG_SHAPE_NUM;

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
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.blockDim));
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

uint64_t BatchNormInferTiling::GetTilingKey() const
{
    return TILINGKEY_INFER;
}

ge::graphStatus BatchNormInferTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(tilingData_.GetDataSize() > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(),
                    "actual tiling data size %zu > context tiling data size %zu",
                    tilingData_.GetDataSize(), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm, BatchNormInferTiling, 91000);
}  // namespace optiling
