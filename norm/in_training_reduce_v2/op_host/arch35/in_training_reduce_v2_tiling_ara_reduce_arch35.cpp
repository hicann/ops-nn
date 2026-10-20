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
 * \file in_training_reduce_v2_tiling_ara_reduce_arch35.cpp
 * \brief ARA reduce（NHWC/NDHWC）：按 C 切核，R 超 UB 时按 rFactor 分载。
 */

#include <algorithm>
#include <limits>
#include "in_training_reduce_v2_tiling.h"

using namespace ge;

namespace {
constexpr uint64_t TILINGKEY_ARA_REDUCE = 400000;
constexpr uint64_t RESERVE_FOR_ALIGN = 512;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t FP16_BYTES = sizeof(uint16_t);
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t OUTPUT_NUM = 2;
constexpr uint64_t ACC_COMPONENTS = 2;
constexpr uint64_t A0_UB_FACTOR = 2;
constexpr uint64_t PARALLEL_CHANNEL_MIN_REDUCE = 1024;
constexpr uint64_t CONTIGUOUS_GATHER_MAX_CHANNELS = 16;
constexpr uint64_t GATHER_TILE_MAX_CHANNELS = 8;
constexpr uint64_t MAX_COPY_BLOCK_COUNT = 4095;
constexpr uint64_t MAX_GM_STRIDE_BYTES = (1UL << 40) - 1;
constexpr int64_t ARA_VECTOR_BYTES = 256;
constexpr int64_t ARA_UB_BLOCK_BYTES = 32;
} // namespace

namespace optiling {
bool INTrainingReduceV2ARAReduceTiling::IsCapable()
{
    return (format == FORMAT_NHWC || format == FORMAT_NDHWC) && r > 0;
}

ge::graphStatus INTrainingReduceV2ARAReduceTiling::DoOpTiling()
{
    const uint64_t elemSize = dataType == ge::DT_FLOAT16 ? FP16_BYTES : FP32_BYTES;
    OP_CHECK_IF(static_cast<uint64_t>(a0) > MAX_GM_STRIDE_BYTES / elemSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetNodeName(), "x", Ops::Base::ToString(xStorageShape).c_str(),
                                                      "The input row must fit the 40-bit GM stride limit"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vectorLength != ARA_VECTOR_BYTES || ubBlockSize != ARA_UB_BLOCK_BYTES,
                OP_LOGE(GetNodeName(), "ARA requires 256-byte vectors and 32-byte UB blocks."),
                return ge::GRAPH_FAILED);
    const uint64_t accumulatorChannels = static_cast<uint64_t>(vlfp32) * A0_UB_FACTOR;
    uint64_t tileCLen = accumulatorChannels;
    if (static_cast<uint64_t>(r) >= PARALLEL_CHANNEL_MIN_REDUCE) {
        const uint64_t tilesPerInstance = std::max(uint64_t{1}, aicoreParams_.blockDim / static_cast<uint64_t>(a1));
        tileCLen = std::min(tileCLen, Ops::Base::CeilDiv(static_cast<uint64_t>(a0), tilesPerInstance));
    }
    // Kernel correction planes retain their fixed offset even when fewer
    // channels are assigned to a core. Do not shrink these two output buffers.
    const uint64_t outputBytes = accumulatorChannels * FP32_BYTES * OUTPUT_NUM * ACC_COMPONENTS;
    const uint64_t maxTileC = std::min(static_cast<uint64_t>(a0), tileCLen);
    // Small channel tiles share a short full row. Read that row contiguously
    // in each owner instead of issuing one narrow strided DMA block per R.
    const bool contiguousGatherInput = static_cast<uint64_t>(a0) <= CONTIGUOUS_GATHER_MAX_CHANNELS &&
                                       maxTileC <= GATHER_TILE_MAX_CHANNELS && r >= vlfp32;
    const uint64_t inputChannels = contiguousGatherInput ? static_cast<uint64_t>(a0) : maxTileC;
    const uint64_t inputRowBytes = Ops::Base::CeilAlign(inputChannels * elemSize, static_cast<uint64_t>(ubBlockSize));
    const uint64_t vectorRowBytes = Ops::Base::CeilAlign(inputChannels, static_cast<uint64_t>(vlfp32)) * elemSize;
    // 尾行的非掩码 LoadAlign 仍读取整个 vector；清零无效 lane 前必须保证地址落在已分配 UB 内。
    const uint64_t inputTailBytes = vectorRowBytes - inputRowBytes;
    const uint64_t fixedBytes = RESERVE_FOR_ALIGN + outputBytes + inputTailBytes * DOUBLE_BUFFER;

    OP_CHECK_IF(aicoreParams_.ubSize <= fixedBytes,
                OP_LOGE(GetNodeName(), "ARA tiling cannot fit output accumulators in UB."), return ge::GRAPH_FAILED);

    // 小 C 按实际 DMA 行宽预算，避免为每行预留完整的 128 个通道。
    uint64_t rFactor = (aicoreParams_.ubSize - fixedBytes) / (inputRowBytes * DOUBLE_BUFFER);
    rFactor = std::min(rFactor, MAX_COPY_BLOCK_COUNT);
    rFactor = std::min(rFactor, static_cast<uint64_t>(r));
    OP_CHECK_IF(rFactor == 0, OP_LOGE(GetNodeName(), "ARA rFactor is 0."), return ge::GRAPH_FAILED);

    const uint64_t cOuter = Ops::Base::CeilDiv(static_cast<uint64_t>(a0), tileCLen);
    const uint64_t tileCTail = static_cast<uint64_t>(a0) - (cOuter - 1) * tileCLen;
    OP_CHECK_IF(static_cast<uint64_t>(a1) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / cOuter,
                OP_LOGE(GetNodeName(), "ARA tile count exceeds int64 range."), return ge::GRAPH_FAILED);
    const uint64_t totalTiles = static_cast<uint64_t>(a1) * cOuter;
    const uint64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, aicoreParams_.blockDim);
    blockNum_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    td_.totalRows = totalRows_;
    td_.totalElements = totalElements_;
    td_.inputBufferBytes = rFactor * inputRowBytes + inputTailBytes;
    td_.outputBufferBytes = accumulatorChannels * FP32_BYTES * ACC_COMPONENTS;
    td_.numN = a1;
    td_.numR = r;
    td_.numC = a0;
    td_.totalTiles = totalTiles;
    td_.tilesPerCore = tilesPerCore;
    td_.cOuter = cOuter;
    td_.tileCLen = tileCLen;
    td_.tileCTail = tileCTail;
    td_.rFactor = rFactor;
    td_.numChunks = Ops::Base::CeilDiv(static_cast<uint64_t>(r), rFactor);
    return ge::GRAPH_SUCCESS;
}

uint64_t INTrainingReduceV2ARAReduceTiling::GetTilingKey() const { return TILINGKEY_ARA_REDUCE; }

ge::graphStatus INTrainingReduceV2ARAReduceTiling::PostTiling()
{
    OP_CHECK_IF(context_->SetBlockDim(static_cast<uint32_t>(blockNum_)) != ge::GRAPH_SUCCESS,
                OP_LOGE(GetNodeName(), "Set blockDim failed"), return ge::GRAPH_FAILED);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(td_),
                OP_LOGE(GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(td_),
                        rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    OP_CHECK_IF(memcpy_s(ptrData, rawTilingData->GetCapacity(), &td_, sizeof(td_)) != 0,
                OP_LOGE(GetNodeName(), "Set tiling data failed."), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(INTrainingReduceV2, INTrainingReduceV2ARAReduceTiling,
                             IN_TRAINING_REDUCE_V2_ARA_REDUCE_PRIORITY);
} // namespace optiling
