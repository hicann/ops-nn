/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unique_dim_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling {

using namespace Ops::Base;

constexpr int64_t TILE_SIZE = 512;
constexpr int64_t BLOCK_DIM_X = 256;
constexpr int64_t MAGIC_GM_PAGE_SIZE = 128;
constexpr int64_t ALIGNMENT = 128;
constexpr int32_t SHAPE_LEN = 27;

constexpr int64_t TILING_KEY_EMPTY = 0;
constexpr int64_t TILING_KEY_BASE = 1; // non-empty (flags read from tiling data at runtime)

static inline int64_t CeilAlign(int64_t size, int64_t alignment)
{
    return (alignment == 0) ? size : (size + alignment - 1) / alignment * alignment;
}

bool UniqueDimTilingHelper::DoTiling()
{
    OP_CHECK_IF(!GetPlatformInfo(), OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed."), return false);
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "GetShapeInfo failed."), return false);
    OP_CHECK_IF(!ReadAttrs(), OP_LOGE(context_->GetNodeName(), "ReadAttrs failed."), return false);
    OP_CHECK_IF(!DoBlockTiling(), OP_LOGE(context_->GetNodeName(), "DoBlockTiling failed."), return false);
    OP_CHECK_IF(!ComputeWorkspaces(), OP_LOGE(context_->GetNodeName(), "ComputeWorkspaces failed."), return false);
    return true;
}

bool UniqueDimTilingHelper::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(nullptr == platformInfo, OP_LOGE(context_->GetNodeName(), "platform info is null"), return false);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    blockSize_ = GetUbBlockSize(context_);
    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGI("GetPlatformInfo", "aivCoreNum: %u, ubSize: %lu, blockSize: %u", aivCoreNum_, ubSize_, blockSize_);
    return true;
}

bool UniqueDimTilingHelper::GetShapeInfo()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE("GetShapeInfo", "input desc is null"), return false);

    dataTypeX_ = inputDesc->GetDataType();
    dtSizeX_ = GetSizeByDataType(dataTypeX_);

    auto inputShape = context_->GetInputShape(0)->GetStorageShape();
    totalElements_ = inputShape.GetShapeSize();
    int64_t dimNum = inputShape.GetDimNum();

    OP_CHECK_IF(dimNum > 8,
        OP_LOGE(context_->GetNodeName(), "UniqueDim only supports up to 8 dimensions, got %ld.", dimNum),
        return false);

    // Read dim from attrs (attr index 0)
    auto attrs = context_->GetAttrs();
    int64_t rawDim = 0;
    if (attrs != nullptr && attrs->GetAttrNum() >= 1) {
        const int64_t *dimPtr = attrs->GetAttrPointer<int64_t>(0);
        if (dimPtr != nullptr) {
            rawDim = *dimPtr;
        }
    }
    dim_ = (rawDim < 0) ? (rawDim + dimNum) : rawDim;

    numInp_ = inputShape.GetDim(dim_);
    rowLen_ = (numInp_ > 0) ? totalElements_ / numInp_ : 0;

    outerSize_ = 1;
    for (int64_t i = 0; i < dim_; i++) {
        outerSize_ *= inputShape.GetDim(i);
    }
    innerSize_ = (numInp_ > 0) ? rowLen_ / outerSize_ : 0;

    inputDimNum_ = dimNum;
    for (int64_t i = 0; i < dimNum; i++) {
        inputDims_[i] = inputShape.GetDim(i);
    }

    OP_LOGI("GetShapeInfo", "dataType=%d, dtSize=%ld, dim=%ld, numInp=%ld, rowLen=%ld, outerSize=%ld, innerSize=%ld",
            static_cast<int32_t>(dataTypeX_), dtSizeX_, dim_, numInp_, rowLen_, outerSize_, innerSize_);
    return true;
}

bool UniqueDimTilingHelper::ReadAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() >= 4) {
        const bool *ri = attrs->GetAttrPointer<bool>(2);
        const bool *rc = attrs->GetAttrPointer<bool>(3);
        if (ri != nullptr) {
            returnInverse_ = *ri;
        }
        if (rc != nullptr) {
            returnCounts_ = *rc;
        }
    }
    OP_LOGI("ReadAttrs", "returnInverse=%d, returnCounts=%d", returnInverse_, returnCounts_);
    return true;
}

bool UniqueDimTilingHelper::DoBlockTiling()
{
    if (numInp_ <= 0) {
        usedCoreNum_ = 1;
        elementsPerCore_ = 0;
        tailCoreElements_ = 0;
        numTiles_ = 0;
        return true;
    }

    // Always use all available cores for maximum parallelism in
    // transpose, cross-core merge, gather, and other cooperative phases.
    usedCoreNum_ = aivCoreNum_;

    elementsPerCore_ = CeilDiv(numInp_, usedCoreNum_);
    tailCoreElements_ = numInp_ - (usedCoreNum_ - 1) * elementsPerCore_;
    if (tailCoreElements_ < 0) {
        tailCoreElements_ = 0;
    }

    numTiles_ = CeilDiv(numInp_, TILE_SIZE);
    tileSize_ = TILE_SIZE;
    blockDimX_ = BLOCK_DIM_X;

    OP_LOGI("DoBlockTiling", "usedCoreNum=%ld, elementsPerCore=%ld, tailCoreElements=%ld, numTiles=%ld",
            usedCoreNum_, elementsPerCore_, tailCoreElements_, numTiles_);
    return true;
}

bool UniqueDimTilingHelper::ComputeWorkspaces()
{
    int64_t N = numInp_;
    int64_t CN = usedCoreNum_;
    constexpr int64_t IDX_SIZE = sizeof(uint32_t); // 4 bytes, saves 50% workspace vs int64_t

    indicesOffset_ = 0;
    sortBufOffset_ = CeilAlign(N * IDX_SIZE, ALIGNMENT);
    flagsOffset_ = CeilAlign(sortBufOffset_ + N * IDX_SIZE, ALIGNMENT);

    int64_t nextOffset;
    if (returnCounts_) {
        positionsOffset_ = CeilAlign(flagsOffset_ + N * IDX_SIZE, ALIGNMENT);
        nextOffset = positionsOffset_ + (N + 1) * IDX_SIZE;
    } else {
        positionsOffset_ = 0;
        nextOffset = flagsOffset_ + N * IDX_SIZE;
    }

    partialSumOffset_ = CeilAlign(nextOffset, ALIGNMENT);
    globalPrefixOffset_ = CeilAlign(partialSumOffset_ + CN * MAGIC_GM_PAGE_SIZE, ALIGNMENT);
    shapeOutOffset_ = CeilAlign(globalPrefixOffset_ + (CN + 1) * IDX_SIZE, ALIGNMENT);
    transposeDstOffset_ = CeilAlign(shapeOutOffset_ + SHAPE_LEN * sizeof(uint64_t), ALIGNMENT);
    if (dim_ != 0) {
        workspaceSize_ = CeilAlign(transposeDstOffset_ + totalElements_ * dtSizeX_, ALIGNMENT);
    } else {
        workspaceSize_ = transposeDstOffset_;
    }

    OP_LOGI("ComputeWorkspaces", "workspaceSize=%ld", workspaceSize_);
    return true;
}

void UniqueDimTilingHelper::SetTilingData(UniqueDimTilingData *tiling)
{
    tiling->set_numInp(numInp_);
    tiling->set_rowLen(rowLen_);
    tiling->set_returnInverse(returnInverse_ ? 1 : 0);
    tiling->set_returnCounts(returnCounts_ ? 1 : 0);
    tiling->set_usedCoreNum(usedCoreNum_);
    tiling->set_elementsPerCore(elementsPerCore_);
    tiling->set_tailCoreElements(tailCoreElements_);
    tiling->set_blockDimX(blockDimX_);
    tiling->set_inputDtypeSize(dtSizeX_);
    tiling->set_tileSize(tileSize_);
    tiling->set_numTiles(numTiles_);

    tiling->set_indicesOffset(indicesOffset_);
    tiling->set_sortBufOffset(sortBufOffset_);
    tiling->set_flagsOffset(flagsOffset_);
    tiling->set_positionsOffset(positionsOffset_);
    tiling->set_partialSumOffset(partialSumOffset_);
    tiling->set_globalPrefixOffset(globalPrefixOffset_);
    tiling->set_shapeOutOffset(shapeOutOffset_);
    tiling->set_workspaceSize(workspaceSize_);

    tiling->set_dim(dim_);
    tiling->set_outerSize(outerSize_);
    tiling->set_innerSize(innerSize_);
    tiling->set_transposeDstOffset(transposeDstOffset_);

    tiling->set_inputDimNum(inputDimNum_);
    tiling->set_inputDim0(inputDims_[0]);
    tiling->set_inputDim1(inputDims_[1]);
    tiling->set_inputDim2(inputDims_[2]);
    tiling->set_inputDim3(inputDims_[3]);
    tiling->set_inputDim4(inputDims_[4]);
    tiling->set_inputDim5(inputDims_[5]);
    tiling->set_inputDim6(inputDims_[6]);
    tiling->set_inputDim7(inputDims_[7]);

    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    usedCoreNum_ = (usedCoreNum_ > 0) ? usedCoreNum_ : 1;

    uint64_t tilingKey = (numInp_ > 0) ? TILING_KEY_BASE : TILING_KEY_EMPTY;

    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize_) + static_cast<size_t>(workspaceSize_);

    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);

    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    OP_LOGI("SetTilingData", "tilingKey=%lu, usedCoreNum=%ld, workspaceSize=%ld", tilingKey, usedCoreNum_,
            workspaceSize_);
}

static ge::graphStatus TilingPrepare4UniqueDim(gert::TilingParseContext *context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepare4UniqueDim start");
    auto compileInfo = context->GetCompiledInfo<UniqueDimCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    compileInfo->ubSize = ubSizePlatform;
    compileInfo->blockSize = GetUbBlockSize(context);
    OP_LOGI(context->GetNodeName(), "aivCoreNum %u, ubSize %lu, blockSize %u, sysWorkspaceSize %u",
            compileInfo->aivCoreNum, compileInfo->ubSize, compileInfo->blockSize, compileInfo->sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4UniqueDim(gert::TilingContext *context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("Tiling4UniqueDim", "context is NULL"), return ge::GRAPH_FAILED);

    UniqueDimTilingData tiling;
    OP_LOGI(context->GetNodeName(), "Begin to do Tiling4UniqueDim");

    UniqueDimTilingHelper helper(context);

    bool status = helper.DoTiling();
    OP_CHECK_IF(!status, OP_LOGE(context->GetNodeName(), "DoTiling failed."), return ge::GRAPH_FAILED);
    helper.SetTilingData(&tiling);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UniqueDim)
    .Tiling(Tiling4UniqueDim)
    .TilingParse<UniqueDimCompileInfo>(TilingPrepare4UniqueDim);

} // namespace optiling
