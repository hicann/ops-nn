/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fake_quant_with_min_max_vars_per_channel_gradient_tiling.cpp
 * \brief
 */

#include "fake_quant_with_min_max_vars_per_channel_gradient_tiling.h"

namespace optiling {
constexpr uint64_t TILING_KEY_FP32 = 1001;
constexpr uint32_t VL_FP32 = 64;
constexpr uint32_t RESERVED_UB = 8 * 1024; // 8 KB scratchpad
// 9 FP32 buffers all sized by dTileLen: nudgedMin/Max + bpMinAcc/Max + bpMinComp/MaxComp + g + x + bpx = 9 * 4 bytes.
constexpr uint32_t BYTES_PER_DTILE_ELEM = 36;
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t SPLIT_MODE_ROWS = 0;
constexpr uint32_t SPLIT_MODE_DCHUNKS = 1;
constexpr size_t ATTR_IDX_NUM_BITS = 0;
constexpr size_t ATTR_IDX_NARROW_RANGE = 1;
constexpr size_t INPUT_IDX_X = 1;

class FakeQuantWithMinMaxVarsPerChannelGradientTiling {
public:
    explicit FakeQuantWithMinMaxVarsPerChannelGradientTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    FakeQuantWithMinMaxVarsPerChannelGradientTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t usedCoreNum_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t tailRows_ = 0;
    uint32_t totalRows_ = 0;
    uint32_t channelNum_ = 0;
    uint32_t tileLen_ = 0;
    int32_t quantMin_ = 0;
    int32_t quantMax_ = 0;
    uint32_t dTileLen_ = 0;
    uint32_t numDChunks_ = 0;
    uint32_t splitMode_ = 0;
    uint32_t chunksPerCore_ = 0;
    uint32_t tailChunks_ = 0;
    uint64_t ubSizePlatForm_ = 0;
    uint32_t coreNumPlatform_ = 0;
};

ge::graphStatus FakeQuantWithMinMaxVarsPerChannelGradientTiling::Init()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling init begin.");

    auto compileInfo = tilingContext->GetCompileInfo<FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo>();
    OP_CHECK_IF((compileInfo == nullptr), OP_LOGE(tilingContext, "compileInfo is nullptr"), return ge::GRAPH_FAILED);
    coreNumPlatform_ = static_cast<uint32_t>(compileInfo->totalCoreNum);
    ubSizePlatForm_ = compileInfo->ubSizePlatForm;
    if (coreNumPlatform_ == 0 || ubSizePlatForm_ == 0) {
        OP_LOGE(tilingContext, "platform info invalid: coreNum=%u, ubSize=%lu", coreNumPlatform_, ubSizePlatForm_);
        return ge::GRAPH_FAILED;
    }

    auto xShapePtr = tilingContext->GetInputShape(INPUT_IDX_X);
    OP_CHECK_IF((xShapePtr == nullptr), OP_LOGE(tilingContext, "x shape is nullptr"), return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();
    size_t dimNum = xShape.GetDimNum();
    if (dimNum == 0) {
        OP_LOGE(tilingContext, "x rank is 0");
        return ge::GRAPH_FAILED;
    }
    int64_t numel = xShape.GetShapeSize();
    int64_t channelNum = xShape.GetDim(static_cast<int64_t>(dimNum) - 1);
    if (channelNum <= 0) {
        OP_LOGE(tilingContext, "channelNum <= 0");
        return ge::GRAPH_FAILED;
    }
    if (numel <= 0 || (numel % channelNum) != 0) {
        OP_LOGE(tilingContext, "x numel %ld not divisible by channelNum %ld", numel, channelNum);
        return ge::GRAPH_FAILED;
    }
    int64_t totalRows = numel / channelNum;
    if (totalRows <= 0) {
        OP_LOGE(tilingContext, "totalRows <= 0");
        return ge::GRAPH_FAILED;
    }
    totalRows_ = static_cast<uint32_t>(totalRows);
    channelNum_ = static_cast<uint32_t>(channelNum);

    // Attrs
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_IF((attrs == nullptr), OP_LOGE(tilingContext, "attrs is nullptr"), return ge::GRAPH_FAILED);
    int32_t numBits = 8;
    bool narrowRange = false;
    const int32_t* numBitsPtr = attrs->GetAttrPointer<int32_t>(ATTR_IDX_NUM_BITS);
    if (numBitsPtr != nullptr) {
        numBits = *numBitsPtr;
    }
    const bool* narrowPtr = attrs->GetAttrPointer<bool>(ATTR_IDX_NARROW_RANGE);
    if (narrowPtr != nullptr) {
        narrowRange = *narrowPtr;
    }
    if (numBits < 2 || numBits > 16) {
        OP_LOGE(tilingContext, "num_bits %d out of range [2,16]", numBits);
        return ge::GRAPH_FAILED;
    }
    quantMin_ = narrowRange ? 1 : 0;
    quantMax_ = (1 << numBits) - 1;

    // D-chunk size: every UB-resident buffer is sized by dTileLen, no longer by D.
    // freeBytes = ubSize - RESERVED_UB; dTileLen = floor(freeBytes / 28), align down to VL=64.
    int64_t freeBytes = static_cast<int64_t>(ubSizePlatForm_) - static_cast<int64_t>(RESERVED_UB);
    if (freeBytes <= 0) {
        OP_LOGE(tilingContext, "UB %lu too small (reserved %u)", ubSizePlatForm_, RESERVED_UB);
        return ge::GRAPH_FAILED;
    }
    int64_t rawDTileLen = freeBytes / static_cast<int64_t>(BYTES_PER_DTILE_ELEM);
    int64_t alignedDTileLen = (rawDTileLen / VL_FP32) * VL_FP32;
    if (alignedDTileLen <= 0) {
        OP_LOGE(tilingContext, "dTileLen too small after alignment");
        return ge::GRAPH_FAILED;
    }
    // Cap dTileLen at ceil(D, VL): no benefit to going larger than D itself.
    uint32_t dCap = ((channelNum_ + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    if (alignedDTileLen > static_cast<int64_t>(dCap)) {
        alignedDTileLen = dCap;
    }
    dTileLen_ = static_cast<uint32_t>(alignedDTileLen);
    numDChunks_ = (channelNum_ + dTileLen_ - 1) / dTileLen_;
    // Keep tileLen_ (legacy field) identical to dTileLen_ — the kernel now uses dTileLen
    // everywhere, but old field is preserved in TilingData for compatibility.
    tileLen_ = dTileLen_;

    // Multi-core split strategy:
    //   N > 1                       -> split rows across cores (split_mode=0)
    //   N == 1 and numDChunks_ > 1  -> split d-chunks across cores (split_mode=1)
    //   otherwise                   -> single core
    if (totalRows_ > 1) {
        splitMode_ = SPLIT_MODE_ROWS;
        usedCoreNum_ = totalRows_ < coreNumPlatform_ ? totalRows_ : coreNumPlatform_;
        rowsPerCore_ = totalRows_ / usedCoreNum_;
        tailRows_ = totalRows_ % usedCoreNum_;
        chunksPerCore_ = numDChunks_;
        tailChunks_ = 0;
    } else {
        splitMode_ = SPLIT_MODE_DCHUNKS;
        usedCoreNum_ = numDChunks_ < coreNumPlatform_ ? numDChunks_ : coreNumPlatform_;
        if (usedCoreNum_ == 0) {
            usedCoreNum_ = 1;
        }
        chunksPerCore_ = numDChunks_ / usedCoreNum_;
        tailChunks_ = numDChunks_ % usedCoreNum_;
        rowsPerCore_ = totalRows_; // every core processes the single row
        tailRows_ = 0;
    }
    if (usedCoreNum_ == 0) {
        OP_LOGE(tilingContext, "usedCoreNum is 0");
        return ge::GRAPH_FAILED;
    }

    // WHY: split_mode=0 cross-core reduce per channel needs a 2*usedCoreNum*ceil(D,VL) workspace
    // for bp_min/bp_max merge; split_mode=1 partitions disjoint d-chunks across cores so each
    // core writes its own channel range directly to bp_min/bp_max GM, no cross-core reduce
    // workspace is required (avoids GB-scale workspace when channelNum is huge).
    size_t userWorkspaceBytes = 0;
    if (splitMode_ == SPLIT_MODE_ROWS) {
        uint32_t wsStride = ((channelNum_ + VL_FP32 - 1) / VL_FP32) * VL_FP32;
        userWorkspaceBytes = static_cast<size_t>(2) * usedCoreNum_ * wsStride * sizeof(float);
    }
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = SYS_WORKSPACE_SIZE + userWorkspaceBytes;

    tilingContext->SetTilingKey(TILING_KEY_FP32);
    OP_LOGD(tilingContext->GetNodeName(),
            "Tiling init done. totalRows=%u channelNum=%u usedCore=%u splitMode=%u "
            "rowsPerCore=%u tailRows=%u dTileLen=%u numDChunks=%u chunksPerCore=%u tailChunks=%u "
            "qmin=%d qmax=%d",
            totalRows_, channelNum_, usedCoreNum_, splitMode_, rowsPerCore_, tailRows_, dTileLen_, numDChunks_,
            chunksPerCore_, tailChunks_, quantMin_, quantMax_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxVarsPerChannelGradientTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "RunKernelTiling start.");
    tilingContext->SetBlockDim(usedCoreNum_);
    tilingData.set_usedCoreNum(usedCoreNum_);
    tilingData.set_rowsPerCore(rowsPerCore_);
    tilingData.set_tailRows(tailRows_);
    tilingData.set_totalRows(totalRows_);
    tilingData.set_channelNum(channelNum_);
    tilingData.set_tileLen(tileLen_);
    tilingData.set_quantMin(quantMin_);
    tilingData.set_quantMax(quantMax_);
    tilingData.set_dTileLen(dTileLen_);
    tilingData.set_numDChunks(numDChunks_);
    tilingData.set_splitMode(splitMode_);
    tilingData.set_chunksPerCore(chunksPerCore_);
    tilingData.set_tailChunks(tailChunks_);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(tilingContext->GetNodeName(), "RunKernelTiling end.");
    return ge::GRAPH_SUCCESS;
}

void FakeQuantWithMinMaxVarsPerChannelGradientTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext->GetNodeName(), "usedCoreNum: %u", usedCoreNum_);
    OP_LOGD(tilingContext->GetNodeName(), "rowsPerCore: %u", rowsPerCore_);
    OP_LOGD(tilingContext->GetNodeName(), "tailRows: %u", tailRows_);
    OP_LOGD(tilingContext->GetNodeName(), "totalRows: %u", totalRows_);
    OP_LOGD(tilingContext->GetNodeName(), "channelNum: %u", channelNum_);
    OP_LOGD(tilingContext->GetNodeName(), "tileLen: %u", tileLen_);
    OP_LOGD(tilingContext->GetNodeName(), "quantMin: %d", quantMin_);
    OP_LOGD(tilingContext->GetNodeName(), "quantMax: %d", quantMax_);
    OP_LOGD(tilingContext->GetNodeName(), "dTileLen: %u", dTileLen_);
    OP_LOGD(tilingContext->GetNodeName(), "numDChunks: %u", numDChunks_);
    OP_LOGD(tilingContext->GetNodeName(), "splitMode: %u", splitMode_);
    OP_LOGD(tilingContext->GetNodeName(), "chunksPerCore: %u", chunksPerCore_);
    OP_LOGD(tilingContext->GetNodeName(), "tailChunks: %u", tailChunks_);
}

static ge::graphStatus TilingFakeQuantWithMinMaxVarsPerChannelGradient(gert::TilingContext* context)
{
    FakeQuantWithMinMaxVarsPerChannelGradientTiling tilingObject(context);
    auto status = tilingObject.Init();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus TilingPrepareForFakeQuantWithMinMaxVarsPerChannelGradient(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForFakeQuantWithMinMaxVarsPerChannelGradient start.");
    auto compileInfo = context->GetCompiledInfo<FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF((compileInfo->ubSizePlatForm == 0), OP_LOGE(context, "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0), OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "TilingPrepareForFakeQuantWithMinMaxVarsPerChannelGradient end. core=%d ub=%lu",
            compileInfo->totalCoreNum, compileInfo->ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FakeQuantWithMinMaxVarsPerChannelGradient)
    .Tiling(TilingFakeQuantWithMinMaxVarsPerChannelGradient)
    .TilingParse<FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo>(
        TilingPrepareForFakeQuantWithMinMaxVarsPerChannelGradient);
} // namespace optiling
