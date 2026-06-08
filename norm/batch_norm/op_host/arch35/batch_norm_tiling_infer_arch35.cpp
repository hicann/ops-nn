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
#include <algorithm>
#include "batch_norm_tiling.h"

using namespace ge;

namespace
{
constexpr int64_t TILINGKEY_INFER = 910000;
constexpr int64_t TILINGKEY_INFER_SMALL_AB1 = 911000;
constexpr int64_t SMALL_SHAPE_NUM = 6;  // scale, offset, mean, var, outMean, outVar
constexpr int64_t BIG_SHAPE_NUM = 2;    // x, y
constexpr int64_t SMALL_AB1_FACTOR_LIMIT = 32;
constexpr int64_t MIN_SMALL_AB1_B0_CORE_FACTOR = 2;
constexpr int64_t SMALL_AB1_CACHE_BUFFER_NUM = 5;
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

    bool isSmallAB1Mode_{false};
    BatchNormInferTilingData tilingData_;
};

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return value;
    }

    return (value + factor - 1) / factor;
}

inline static int64_t AlignUp(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}

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
    int64_t b1Inner = 1;
    int64_t b1Outer = 1;
    int64_t b0Inner = 1;
    int64_t b0Outer = 1;
    int64_t aOuter = 1;
    isSmallAB1Mode_ = false;
    int64_t minSmallAB1B0Len = std::max<int64_t>(1, static_cast<int64_t>(aicoreParams_.blockDim)) *
                               MIN_SMALL_AB1_B0_CORE_FACTOR;

    if (fusedALen_ > 0 && fusedB1Len_ > 0 && fusedALen_ * fusedB1Len_ <= SMALL_AB1_FACTOR_LIMIT &&
        fusedB0Len_ > minSmallAB1B0Len) {
        isSmallAB1Mode_ = true;
        aInner = fusedALen_;
        b1Inner = 1;
        b1Outer = 1;
        aOuter = 1;

        int64_t paramBytes = SMALL_SHAPE_NUM * FLOAT32_BYTES * aInner;
        int64_t abLen = fusedALen_ * fusedB1Len_;
        int64_t paramCacheElemLen = (vlFp32_ / abLen) * abLen;
        int64_t alignedParamCacheLen = CeilDiv(paramCacheElemLen, vlFp32_) * vlFp32_;
        int64_t smallAB1CacheBytes = SMALL_AB1_CACHE_BUFFER_NUM * alignedParamCacheLen * FLOAT32_BYTES;
        int64_t ubCanUseBytes = static_cast<int64_t>(aicoreParams_.ubSize) - DOUBLE_BUFFER * paramBytes -
                                smallAB1CacheBytes;
        int64_t bytesPerB0 = abLen * bytesPerElement_ * BIG_SHAPE_NUM * DOUBLE_BUFFER;
        b0Inner = bytesPerB0 == 0 ? 1 : ubCanUseBytes / bytesPerB0;
        b0Inner = std::max<int64_t>(1, std::min<int64_t>(b0Inner, fusedB0Len_));
        b0Outer = CeilDiv(fusedB0Len_, b0Inner);
    } else {
        int64_t ubBufferSize =
            (aicoreParams_.ubSize / DOUBLE_BUFFER - SMALL_SHAPE_NUM * FLOAT32_BYTES * aInner * aTileBase_) /
            bytesPerElement_ / BIG_SHAPE_NUM;

        // 先按照B切分，再切A
        // UB可载入最大tile块数
        int64_t factorMax = ubBufferSize / aTileBase_;

        // 默认策略: 先按照B0, B1把UB切满
        int64_t b1FactorMax = CeilDiv(fusedB1Len_, aTileBase_);
        b1Inner = std::min(factorMax, b1FactorMax);
        b1Outer = CeilDiv(fusedB1Len_, b1Inner * aTileBase_);

        factorMax = factorMax / b1Inner;
        int64_t b0FactorMax = fusedB0Len_;
        b0Inner = std::min(factorMax, b0FactorMax);
        b0Outer = CeilDiv(fusedB0Len_, b0Inner);

        factorMax = factorMax / b0Inner;
        int64_t aFactorMax = fusedALen_;
        aInner = std::min(factorMax, aFactorMax);
        int64_t maxAInnerByUb = aicoreParams_.ubSize / DOUBLE_BUFFER /
                                (BIG_SHAPE_NUM * b0Inner * b1Inner * aTileBase_ * bytesPerElement_ +
                                 SMALL_SHAPE_NUM * FLOAT32_BYTES);
        aInner = std::min(aInner, maxAInnerByUb);
        int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
        auto getAlignedUbSize = [this, b0Inner, b1Inner](int64_t aInnerCandidate) {
            int64_t tileBlockB1Len = b1Inner * aTileBase_;
            int64_t xyBufferSize = b0Inner * aInnerCandidate * tileBlockB1Len * bytesPerElement_;
            int64_t paramBufferSize = aInnerCandidate * FLOAT32_BYTES;
            return DOUBLE_BUFFER *
                   (BIG_SHAPE_NUM * AlignUp(xyBufferSize, blockSize_) +
                    SMALL_SHAPE_NUM * AlignUp(paramBufferSize, blockSize_));
        };
        while (aInner > 0 && getAlignedUbSize(aInner) > ubSize) {
            --aInner;
        }
        OP_CHECK_IF(aInner <= 0,
                    OP_LOGE(context_->GetNodeName(),
                        "Invalid tiling params, aInner: %ld, b0Inner: %ld, b1Inner: %ld, aTileBase: %ld, "
                        "bytesPerElement: %ld, ubSize: %lu",
                        aInner, b0Inner, b1Inner, aTileBase_, bytesPerElement_, aicoreParams_.ubSize),
                    return ge::GRAPH_FAILED);
        aOuter = CeilDiv(fusedALen_, aInner);
    }

    int64_t totalTiles = b0Outer * aOuter * b1Outer;
    int64_t tilesPerCore = CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.blockDim));
    usedCoreNums_ = CeilDiv(totalTiles, tilesPerCore);

    int64_t tileBlockB0Tail = fusedB0Len_ - b0Inner * (b0Outer - 1);
    int64_t tileBlockATail = fusedALen_ - aInner * (aOuter - 1);
    int64_t tileBlockB1Len = isSmallAB1Mode_ ? fusedB1Len_ : b1Inner * aTileBase_;
    int64_t tileBlockB1Tail = fusedB1Len_ - tileBlockB1Len * (b1Outer - 1);

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
    tilingData_.set_tileBlockB1Len(tileBlockB1Len);
    tilingData_.set_tileBlockB1Tail(tileBlockB1Tail);
    tilingData_.set_tileBlockAPaddingNum(0);
    tilingData_.set_epsilon(epsilon_);

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormInferTiling::GetTilingKey() const
{
    return isSmallAB1Mode_ ? TILINGKEY_INFER_SMALL_AB1 : TILINGKEY_INFER;
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
