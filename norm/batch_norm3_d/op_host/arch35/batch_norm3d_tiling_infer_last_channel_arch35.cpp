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
 * \file batch_norm3d_tiling_infer_last_channel_arch35.cpp
 * \brief
 */
#include "batch_norm3d_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_INFER_LAST_CHANNEL = 900000;
constexpr int64_t TILINGKEY_INFER_LAST_CHANNEL_SMALL_A = 902000;
constexpr int64_t TILINGKEY_INFER_LAST_CHANNEL_CONTINUOUS_A = 901000;
constexpr int64_t MAX_SMALL_A = 8;
constexpr int64_t MIN_CONTINUOUS_A_LEN = 64;
constexpr int64_t MAX_CONTINUOUS_A_OUTER = 6;
constexpr int64_t MAX_CONTINUOUS_A_OUTER_FP16 = 3;
constexpr int64_t MIN_SMALL_A_B_LEN = 65536;
constexpr int64_t SMALL_SHAPE_NUM = 6;       // scale, offset, mean, var, outMean, outVar
constexpr int64_t BIG_SHAPE_NUM = 2;         // x, y
constexpr int64_t MEAN_VAR_OUTPUT_COUNT = 2; // mean, var
constexpr int64_t SMALL_LAST_CHANNEL_CACHE_BUFFER_NUM = 4;
} // namespace

namespace optiling {
class BatchNorm3DInferLastChannelTiling : public BatchNorm3DTilingInferBase {
public:
    explicit BatchNorm3DInferLastChannelTiling(gert::TilingContext* context) : BatchNorm3DTilingInferBase(context) {}
    ~BatchNorm3DInferLastChannelTiling() override = default;

protected:
    bool IsCapable() override
    {
        if (fusedB1Len_ != 1) {
            OP_LOGD(context_, "BatchNorm3D Infer channel_last template not support fused shape(%ld, %ld, %ld).",
                    fusedB0Len_, fusedALen_, fusedB1Len_);
            return false;
        }
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus FillLastChannelTilingForBSplit(int64_t paramBytes, int64_t cacheBytes);
    int64_t GetBaseTilingAOuter() const;

    const char* opName = "BatchNorm3DInferLastChannel";
    bool isSmallLastChannel_{false};
    bool isContinuousLastChannel_{false};
    BatchNorm3DInferLastChannelTilingData tilingData;
};

inline static int64_t AlignUp(int64_t value, int64_t base) { return (value + base - 1) / base * base; }

int64_t BatchNorm3DInferLastChannelTiling::GetBaseTilingAOuter() const
{
    if (aTileBase_ <= 0 || bytesPerElement_ <= 0 || aicoreParams_.ubSize <= 0) {
        return 0;
    }
    int64_t paramBytes = SMALL_SHAPE_NUM * FLOAT32_BYTES * aTileBase_;
    int64_t ubBufferSize = (static_cast<int64_t>(aicoreParams_.ubSize) / DOUBLE_BUFFER - paramBytes) /
                           bytesPerElement_ / BIG_SHAPE_NUM;
    int64_t bFactorMax = ubBufferSize / aTileBase_;
    if (bFactorMax <= 0) {
        return 0;
    }
    int64_t bInner = fusedB0Len_ <= bFactorMax ? fusedB0Len_ : bFactorMax;
    int64_t elemBytes = bInner * BIG_SHAPE_NUM * bytesPerElement_ + SMALL_SHAPE_NUM * FLOAT32_BYTES;
    if (elemBytes <= 0) {
        return 0;
    }
    int64_t aFactorMax = static_cast<int64_t>(aicoreParams_.ubSize) / DOUBLE_BUFFER / aTileBase_ / elemBytes;
    int64_t aInnerMax = fusedALen_ / aTileBase_;
    int64_t aInner = aInnerMax <= aFactorMax ? aInnerMax : aFactorMax;
    int64_t tileBlockALen = aInner == 0 ? aTileBase_ : aInner * aTileBase_;
    return Ops::Base::CeilDiv(fusedALen_, tileBlockALen);
}

ge::graphStatus BatchNorm3DInferLastChannelTiling::FillLastChannelTilingForBSplit(int64_t paramBytes,
                                                                                  int64_t cacheBytes)
{
    int64_t perElemBytes = BIG_SHAPE_NUM * DOUBLE_BUFFER * bytesPerElement_;
    int64_t elemFactorMax = (static_cast<int64_t>(aicoreParams_.ubSize) - paramBytes - cacheBytes) / perElemBytes;
    int64_t bInner = elemFactorMax / fusedALen_;
    bInner = bInner <= 0 ? 1 : bInner;
    bInner = fusedB0Len_ <= bInner ? fusedB0Len_ : bInner;
    while ((paramBytes + cacheBytes + bInner * fusedALen_ * BIG_SHAPE_NUM * DOUBLE_BUFFER * bytesPerElement_ >
            static_cast<int64_t>(aicoreParams_.ubSize)) &&
           bInner > 1) {
        bInner--;
    }
    int64_t bOuter = Ops::Base::CeilDiv(fusedB0Len_, bInner);
    int64_t bTail = fusedB0Len_ % bInner;
    int64_t tileBlockBTail = bTail == 0 ? bInner : bTail;
    int64_t totalTiles = bOuter;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.blockDim));
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    tilingData.totalTiles = totalTiles;
    tilingData.tilesPerCore = tilesPerCore;
    tilingData.usedCoreNums = usedCoreNums_;
    tilingData.totalALen = fusedALen_;
    tilingData.aOuter = 1;
    tilingData.bOuter = bOuter;
    tilingData.tileBlockALen = fusedALen_;
    tilingData.tileBlockATail = fusedALen_;
    tilingData.tileBlockAPaddingNum = 0;
    tilingData.tileBlockBLen = bInner;
    tilingData.tileBlockBTail = tileBlockBTail;
    tilingData.epsilon = epsilon_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNorm3DInferLastChannelTiling::DoOpTiling()
{
    aTileBase_ = vlFp16_;
    bytesPerElement_ = FLOAT16_BYTES;
    if (xDtype_ == ge::DT_FLOAT) {
        aTileBase_ = vlFp32_;
        bytesPerElement_ = FLOAT32_BYTES;
    }

    isSmallLastChannel_ = fusedALen_ > 0 && fusedALen_ <= MAX_SMALL_A && fusedB0Len_ > MIN_SMALL_A_B_LEN;
    if (isSmallLastChannel_) {
        int64_t meanVarOutBytes = MEAN_VAR_OUTPUT_COUNT * AlignUp(FLOAT32_BYTES * fusedALen_, blockSize_);
        int64_t paramBytes = DOUBLE_BUFFER * SMALL_SHAPE_NUM * FLOAT32_BYTES * fusedALen_ + meanVarOutBytes;
        int64_t paramCacheElemLen = (vlFp32_ / fusedALen_) * fusedALen_;
        int64_t offsetBytes = AlignUp(paramCacheElemLen * static_cast<int64_t>(sizeof(uint32_t)), blockSize_);
        int64_t cacheBytes = offsetBytes + SMALL_LAST_CHANNEL_CACHE_BUFFER_NUM *
                                               AlignUp(paramCacheElemLen * FLOAT32_BYTES, blockSize_);
        return FillLastChannelTilingForBSplit(paramBytes, cacheBytes);
    }

    int64_t baseAOuter = GetBaseTilingAOuter();
    int64_t maxContinuousAOuter = xDtype_ == ge::DT_FLOAT ? MAX_CONTINUOUS_A_OUTER : MAX_CONTINUOUS_A_OUTER_FP16;
    isContinuousLastChannel_ = fusedALen_ > MIN_CONTINUOUS_A_LEN && fusedB0Len_ > MIN_SMALL_A_B_LEN && baseAOuter > 1 &&
                               baseAOuter <= maxContinuousAOuter;
    if (isContinuousLastChannel_) {
        int64_t paramAlignLen = AlignUp(fusedALen_, vlFp32_);
        int64_t meanVarOutBytes = MEAN_VAR_OUTPUT_COUNT * AlignUp(FLOAT32_BYTES * fusedALen_, blockSize_);
        int64_t paramBytes = DOUBLE_BUFFER * SMALL_SHAPE_NUM * FLOAT32_BYTES * paramAlignLen + meanVarOutBytes;
        int64_t paramCacheBytes = SMALL_SHAPE_NUM * FLOAT32_BYTES * paramAlignLen;
        return FillLastChannelTilingForBSplit(paramBytes, paramCacheBytes);
    }

    // 切分A、B基本块， （B,A） -- >(Bouter, Aouter, Binner*Ainner*ATileBase)
    int64_t aInner = 1;
    int64_t ubBufferSize = (aicoreParams_.ubSize / DOUBLE_BUFFER -
                            SMALL_SHAPE_NUM * FLOAT32_BYTES * aInner * aTileBase_) /
                           bytesPerElement_ / BIG_SHAPE_NUM;

    // 先按照B切分，再切A
    int64_t bFactorMax = ubBufferSize / aTileBase_;
    int64_t bInner = fusedB0Len_ <= bFactorMax ? fusedB0Len_ : bFactorMax;
    int64_t bOuter = Ops::Base::CeilDiv(fusedB0Len_, bInner);
    int64_t bTail = fusedB0Len_ % bInner;
    int64_t tileBlockBTail = bTail == 0 ? bInner : bTail;

    int64_t aFactorMax = aicoreParams_.ubSize / DOUBLE_BUFFER / aTileBase_ /
                         (bInner * BIG_SHAPE_NUM * bytesPerElement_ + SMALL_SHAPE_NUM * FLOAT32_BYTES);
    int64_t aInnerMax = fusedALen_ / aTileBase_;
    aInner = aInnerMax <= aFactorMax ? aInnerMax : aFactorMax;

    int64_t tileBlockALen = aInner == 0 ? aTileBase_ : aInner * aTileBase_;
    int64_t aOuter = Ops::Base::CeilDiv(fusedALen_, tileBlockALen);
    int64_t aTail = fusedALen_ % tileBlockALen;
    int64_t tileBlockATail = aTail == 0 ? tileBlockALen : aTail;
    int64_t tileBlockAPaddingNum = tileBlockALen - tileBlockATail;

    // 切核 （Bouter, Binner, Aouter, Ainner*ATileBase） -- > (Bouter*Aouter, Binner, Ainner*ATileBase)
    int64_t totalTiles = aOuter * bOuter;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, (int64_t)aicoreParams_.blockDim);
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    tilingData.totalTiles = totalTiles;
    tilingData.tilesPerCore = tilesPerCore;
    tilingData.usedCoreNums = usedCoreNums_;
    tilingData.totalALen = fusedALen_;
    tilingData.aOuter = aOuter;
    tilingData.bOuter = bOuter;
    tilingData.tileBlockALen = tileBlockALen;
    tilingData.tileBlockATail = tileBlockATail;
    tilingData.tileBlockAPaddingNum = tileBlockAPaddingNum;
    tilingData.tileBlockBLen = bInner;
    tilingData.tileBlockBTail = tileBlockBTail;
    tilingData.epsilon = epsilon_;

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNorm3DInferLastChannelTiling::GetTilingKey() const
{
    if (isSmallLastChannel_) {
        return TILINGKEY_INFER_LAST_CHANNEL_SMALL_A;
    }
    if (isContinuousLastChannel_) {
        return TILINGKEY_INFER_LAST_CHANNEL_CONTINUOUS_A;
    }
    return TILINGKEY_INFER_LAST_CHANNEL;
}

ge::graphStatus BatchNorm3DInferLastChannelTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto* tilingDataOut = context_->GetTilingData<BatchNorm3DInferLastChannelTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataOut);
    *tilingDataOut = tilingData;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm3D, BatchNorm3DInferLastChannelTiling, 90000);
} // namespace optiling
