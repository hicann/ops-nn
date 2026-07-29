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
 * \file apply_gradient_descent_tiling.cpp
 * \brief apply_gradient_descent classic (ascend910b) tiling.
 *
 * TilingKey selection (dtype): 1 = float16, 2 = float32, 3 = bfloat16.
 */

#include "apply_gradient_descent_tiling.h"

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

// Per-core distribution granularity. 256 elems keeps every core's GM start on a >=512B boundary
// (fp32 1024B, fp16/bf16 512B), the alignment MTE2 prefers for full-width read bursts on this
// memory-bound op. The kernel consumes tiling.blockElems dynamically, so only this host constant moves.
constexpr uint32_t BLOCK_ELEMS = 256;
constexpr uint64_t TILE_ALIGN = 128;   // UB tile alignment (covers vector repeat rounding)
constexpr uint64_t MAX_TILE = 8192;    // cap so per-copy byte length fits DataCopyParams blockLen
constexpr uint64_t UB_RESERVE = 16384; // reserved UB for tiling struct / sync / scalar bufs
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t FP16_BF16_SIZE = 2;
constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

constexpr uint64_t TILING_KEY_FLOAT16 = 1;
constexpr uint64_t TILING_KEY_FLOAT32 = 2;
constexpr uint64_t TILING_KEY_BFLOAT16 = 3;

ge::graphStatus ApplyGradientDescentTiling::CheckDtype()
{
    auto varDesc = context_->GetInputDesc(0);
    auto alphaDesc = context_->GetInputDesc(1);
    auto deltaDesc = context_->GetInputDesc(2);
    auto outDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alphaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, deltaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);

    varDtype_ = varDesc->GetDataType();
    ge::DataType alphaDtype = alphaDesc->GetDataType();
    ge::DataType deltaDtype = deltaDesc->GetDataType();
    ge::DataType outDtype = outDesc->GetDataType();

    OP_CHECK_IF(varDtype_ != ge::DT_FLOAT && varDtype_ != ge::DT_FLOAT16 && varDtype_ != ge::DT_BF16,
                OP_LOGE(context_, "var dtype %d is invalid, only float32/float16/bfloat16 are supported.",
                        static_cast<int32_t>(varDtype_)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(alphaDtype != varDtype_ || deltaDtype != varDtype_ || outDtype != varDtype_,
                OP_LOGE(context_, "dtypes of var/alpha/delta/output var must all be the same."),
                return ge::GRAPH_FAILED);

    if (varDtype_ == ge::DT_FLOAT) {
        dtypeSize_ = FP32_SIZE;
        tilingKey_ = TILING_KEY_FLOAT32;
    } else if (varDtype_ == ge::DT_FLOAT16) {
        dtypeSize_ = FP16_BF16_SIZE;
        tilingKey_ = TILING_KEY_FLOAT16;
    } else {
        dtypeSize_ = FP16_BF16_SIZE;
        tilingKey_ = TILING_KEY_BFLOAT16;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyGradientDescentTiling::CheckShape()
{
    auto varShape = context_->GetInputShape(0);
    auto deltaShape = context_->GetInputShape(2);
    auto outShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, deltaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outShape);

    uint64_t varSize = static_cast<uint64_t>(varShape->GetStorageShape().GetShapeSize());
    uint64_t deltaSize = static_cast<uint64_t>(deltaShape->GetStorageShape().GetShapeSize());
    uint64_t outSize = static_cast<uint64_t>(outShape->GetStorageShape().GetShapeSize());
    OP_CHECK_IF(varSize != deltaSize || varSize != outSize,
                OP_LOGE(context_, "shapes of input var, delta and output var must be the same."),
                return ge::GRAPH_FAILED);

    totalDataCount_ = varSize;
    return ge::GRAPH_SUCCESS;
}

void ApplyGradientDescentTiling::SplitCore()
{
    blockElems_ = BLOCK_ELEMS;
    if (totalDataCount_ == 0) {
        needCoreNum_ = 1;
        blocksPerCore_ = 0;
        remCoreNum_ = 0;
        return;
    }
    uint64_t totalBlocks = (totalDataCount_ + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
    uint64_t used = coreNum_;
    if (used == 0) {
        used = 1;
    }
    if (used > totalBlocks) {
        used = totalBlocks;
    }
    needCoreNum_ = static_cast<uint32_t>(used);
    blocksPerCore_ = totalBlocks / used;
    remCoreNum_ = static_cast<uint32_t>(totalBlocks % used);
}

void ApplyGradientDescentTiling::CalcTileDataCount()
{
    uint64_t ubAvail = (ubSize_ > UB_RESERVE) ? (ubSize_ - UB_RESERVE) : (ubSize_ / 2);
    // per-element UB bytes:
    //   fp32     : 3 T queues * BUFFER_NUM(2) * 4 bytes = 24
    //   fp16/bf16: 3 T queues * 2 * dtypeSize + 2 fp32 temp buffers * 4 = 20
    uint64_t perElemBytes = (varDtype_ == ge::DT_FLOAT) ? (3ULL * 2ULL * FP32_SIZE) :
                                                          (3ULL * 2ULL * dtypeSize_ + 2ULL * FP32_SIZE);
    uint64_t maxTile = ubAvail / perElemBytes;
    maxTile = maxTile / TILE_ALIGN * TILE_ALIGN;
    if (maxTile < TILE_ALIGN) {
        maxTile = TILE_ALIGN;
    }
    if (maxTile > MAX_TILE) {
        maxTile = MAX_TILE;
    }
    tileDataCount_ = maxTile;
}

void ApplyGradientDescentTiling::SetTilingData(ApplyGradientDescentTilingData* tilingData)
{
    tilingData->totalDataCount = totalDataCount_;
    tilingData->tileDataCount = tileDataCount_;
    tilingData->blocksPerCore = blocksPerCore_;
    tilingData->needCoreNum = needCoreNum_;
    tilingData->blockElems = blockElems_;
    tilingData->remCoreNum = remCoreNum_;
    tilingData->reserved = 0;

    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(needCoreNum_);

    size_t* workspaceSize = context_->GetWorkspaceSizes(1);
    if (workspaceSize != nullptr) {
        workspaceSize[0] = SYS_WORKSPACE_SIZE;
    }
}

ge::graphStatus ApplyGradientDescentTiling::RunTiling()
{
    auto compileInfo = reinterpret_cast<const ApplyGradientDescentCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check dtype failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape failed."), return ge::GRAPH_FAILED);

    SplitCore();
    CalcTileDataCount();

    ApplyGradientDescentTilingData* tilingData = context_->GetTilingData<ApplyGradientDescentTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    SetTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ApplyGradientDescent(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4ApplyGradientDescent (ascend910b classic) is running.");
    ApplyGradientDescentTiling tilingObject(context);
    return tilingObject.RunTiling();
}

static ge::graphStatus TilingPrepareForApplyGradientDescent(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ApplyGradientDescentCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF((compileInfo->coreNum == 0) || (compileInfo->ubSize == 0),
                OP_LOGE(context, "failed to get core num or ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyGradientDescent)
    .Tiling(Tiling4ApplyGradientDescent)
    .TilingParse<ApplyGradientDescentCompileInfo>(TilingPrepareForApplyGradientDescent);

} // namespace optiling
