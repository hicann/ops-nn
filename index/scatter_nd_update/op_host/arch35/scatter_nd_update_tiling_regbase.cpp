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
 * \file scatter_nd_update_tiling.cc
 * \brief ascendc scatter nd update tiling cpp
 */

#include "scatter_nd_update_tiling_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/util/math_util.h"

using namespace AscendC;
namespace optiling {
static constexpr uint16_t INPUT_IDX_INDICES = 1;
static constexpr uint16_t INPUT_IDX_UPDATES = 2;
static constexpr uint16_t OUTPUT_IDX_SHAPE = 0;
static constexpr uint16_t RANK_MIN_VALUE = 1;
static constexpr uint16_t RANK_MAX_VALUE = 7;
static constexpr uint64_t MIN_TILING_SIZE = 128;
static constexpr uint32_t DCACHE_SIZE = 32U * 1024U;
static constexpr uint32_t RESERVED_WORKSPACE_SIZE = 16U * 1024U * 1024U;
static constexpr uint32_t INPUT_ADDRESS_IN_INT32 = 100;
static constexpr uint32_t INPUT_ADDRESS_IN_INT64 = 200;
static constexpr uint32_t THREE = 3;
static constexpr uint32_t SIMT_SORT_USED_QUENUM = 5;

static constexpr uint64_t DB_BUFFER = 2;
static constexpr uint64_t RESERVE_SIZE = 256;
static constexpr int64_t ALIGN_SIZE = 32;
static constexpr int64_t MIN_HANDLE_SIZE = 128;
static constexpr int64_t MIN_SIZE_SIMD_NONDETERMINSTIC = 128;
static constexpr int64_t INDICES_MIN_BLOCK_SIZE = 1024;
static constexpr int64_t INT32_BYTES = 4;
static constexpr int64_t FP32_BYTES = 4;
static constexpr int64_t SIMT_SORT_LIMIT = 1;
static constexpr int64_t TWO = 2;
static constexpr int64_t MASK_CORE = 1000;
static constexpr int64_t MASK_VAR = 5;
static constexpr int64_t MASK_AFTER = 19;
static constexpr int64_t ONE = 1;
static constexpr int64_t ROW_THRESH_SIZE = 4096;
static constexpr float PARTIAL_UB = 0.1;
static constexpr int64_t MIN_THREAD_NUM = 128;
static constexpr int64_t MIN_SIZE_SIMD_DETERMINSTIC = 128;

static const std::set<ge::DataType> DETERMIN_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16};

ge::graphStatus ScatterNdUpdateTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (aivNum <= 0), OP_LOGE(opName, "ScatterNdUpdateTiling fail to get totalCoreNum_."), return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(
        (ubSizePlatForm <= DCACHE_SIZE), OP_LOGE(opName, "ub size less than Dcache Size. please check"),
        return ge::GRAPH_FAILED);
    // UB Size Need reserve space for Dcache / CCEC Compile Stack.
    ubSize_ = ubSizePlatForm - DCACHE_SIZE;
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF(
        (res != ge::GRAPH_SUCCESS), OP_LOGE(opName, "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::GetShapeAttrsInfo()
{
    auto var = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, var);
    auto varShapeSize = var->GetShapeSize();
    OP_CHECK_IF(
        (varShapeSize <= 0), OP_LOGE(opName, "var shape size is invalid(%ld)", varShapeSize), return ge::GRAPH_FAILED);
    auto varDesc = context_->GetInputDesc(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    auto varDtype = varDesc->GetDataType();
    varTypeSize_ = ge::GetSizeByDataType(varDtype);

    auto indices = context_->GetInputTensor(INPUT_IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    indiceShapeSize = indices->GetShapeSize();
    OP_CHECK_IF(
        (indiceShapeSize < 0UL), OP_LOGE(opName, "indices shape size is invalid(%ld)", indiceShapeSize),
        return ge::GRAPH_FAILED);
    auto indicesDesc = context_->GetInputDesc(INPUT_IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    indiceDtype_ = indicesDesc->GetDataType();
    indicesTypeSize_ = ge::GetSizeByDataType(indiceDtype_);

    auto indiceShape = indices->GetStorageShape();
    auto indiceDims = indiceShape.GetDimNum();
    rankSize_ = indiceShape.GetDim(indiceDims - 1);
    OP_CHECK_IF(
        (indiceDims < TWO),
        OP_LOGE(opName, "indiceDims %lu less than %u, please check.", indiceDims, static_cast<uint16_t>(TWO)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (RANK_MIN_VALUE > static_cast<uint16_t>(rankSize_) || static_cast<uint16_t>(rankSize_) > RANK_MAX_VALUE),
        OP_LOGE(opName, "rankSize_ %u out of range[1, 7], please check.", rankSize_), return ge::GRAPH_FAILED);

    auto updates = context_->GetInputTensor(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updates);
    updateShapeSize = updates->GetShapeSize();
    OP_CHECK_IF(
        (updateShapeSize < 0UL), OP_LOGE(opName, "update shape size is invalid(%ld)", updateShapeSize),
        return ge::GRAPH_FAILED);

    auto updateDesc = context_->GetInputDesc(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updateDesc);
    auto updateShape = updates->GetStorageShape();
    updateDtype_ = updateDesc->GetDataType();
    OP_CHECK_IF(
        (updateDtype_ != varDtype),
        OP_LOGE(
            opName, "updates [%s] and var [%s] must have the same dtype.", Ops::Base::ToString(updateDtype_).c_str(),
            Ops::Base::ToString(varDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto shapeValue = outputShape->GetStorageShape();
    uint64_t shapeRank = shapeValue.GetDimNum();
    OP_CHECK_IF(
        (shapeRank < rankSize_),
        OP_LOGE(opName, "varShapeRank %lu less than rank %u, please check.", shapeRank, rankSize_),
        return ge::GRAPH_FAILED);

    for (uint64_t idx = 0; idx < shapeRank; idx++) {
        outPutShape[idx] = shapeValue.GetDim(idx);
        outputShapeSize *= outPutShape[idx];
    }

    if (indiceShapeSize == 0UL || updateShapeSize == 0UL) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        CheckScatterNdUpdateTensorShape(indiceShape, updateShape, shapeValue),
        OP_LOGE(opName, "the dim of updateRank and outputRank should be consistent, please check."),
        return ge::GRAPH_FAILED);

    // indicesAxis_ equal updatesInAxis
    indicesAxis_ = static_cast<int64_t>(indiceShapeSize / rankSize_);
    afterAxis_ = static_cast<int64_t>(updateShapeSize) / indicesAxis_;
    varInAxis_ = varShapeSize / afterAxis_;
    sliceSize = static_cast<uint64_t>(afterAxis_);

    if (context_->GetDeterministic() == 1) {
        isDeterminstic_ = 1;
        context_->SetScheduleMode(1);
    }
    if (isDeterminstic_ != 1 && afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_NONDETERMINSTIC) {
        isSimdNonDeterminstic_ = 1;
    }
    if (indicesAxis_ / varInAxis_ >= SIMT_SORT_LIMIT) {
        isSimtWithSort_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::CheckScatterNdUpdateTensorShape(
    const gert::Shape& indiceShape, const gert::Shape& updateShape, const gert::Shape& outputShape)
{
    int64_t indiceDims = indiceShape.GetDimNum();
    int64_t updateDims = updateShape.GetDimNum();
    int64_t outputDims = outputShape.GetDimNum();

    int64_t outputAxisDims = outputDims - static_cast<int64_t>(rankSize_);
    int64_t updateAxisDims = updateDims - (indiceDims - 1);
    if (outputAxisDims != updateAxisDims) {
        return ge::GRAPH_FAILED;
    }

    for (int64_t idx = 0; idx < outputAxisDims; idx++) {
        int64_t updateDim = updateShape.GetDim(idx + indiceDims - 1);
        int64_t outputDim = outputShape.GetDim(idx + rankSize_);
        if (updateDim != outputDim) {
            return ge::GRAPH_FAILED;
        }
    }

    for (int64_t idx = 0; idx < indiceDims - 1; idx++) {
        int64_t updateDim = updateShape.GetDim(idx);
        int64_t indiceDim = indiceShape.GetDim(idx);
        if (indiceDim != updateDim) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void ScatterNdUpdateTiling::BlockTiling()
{
    auto typeSize = ge::GetSizeByDataType(updateDtype_);
    OP_CHECK_IF(typeSize == 0, OP_LOGE(opName, "typeSize is 0"), return);
    alignFactor = Ops::Base::GetUbBlockSize(context_) / typeSize;
    auto blockFactor = Ops::Base::CeilDiv(updateShapeSize, static_cast<uint64_t>(totalCoreNum_));
    auto blockAlignFactor = Ops::Base::CeilDiv(blockFactor, alignFactor) * alignFactor;
    blockTilingSize = std::max(static_cast<uint64_t>(blockAlignFactor), MIN_TILING_SIZE);
    blockNum = Ops::Base::CeilDiv(updateShapeSize, blockTilingSize);
    tailBlockTilingSize = updateShapeSize - blockTilingSize * (blockNum - 1UL);
    OP_LOGD(
        opName,
        "updateShapeSize = %lld, blockFactor = %lld, blockAlignFactor = %lld,"
        "blockTilingSize = %d, tailBlockTilingSize = %d",
        updateShapeSize, blockFactor, blockAlignFactor, blockTilingSize, tailBlockTilingSize);
}

ge::graphStatus ScatterNdUpdateTiling::UbTiling()
{
    if (indiceShapeSize == 0UL || updateShapeSize == 0UL) {
        return ge::GRAPH_SUCCESS;
    }
    // halfUbSize for double buffer
    auto halfUbSize = ubSize_ / DB_BUFFER;
    auto indiceNum = indiceShapeSize / rankSize_;
    sliceSize = updateShapeSize / indiceNum;
    OP_CHECK_IF(
        sliceSize == static_cast<uint64_t>(0), OP_LOGE(opName, "sliceSize %lu is zero. please check.", sliceSize),
        return ge::GRAPH_FAILED);
    auto updateTypeSize = ge::GetSizeByDataType(updateDtype_);
    indiceDtype_ = context_->GetInputDesc(INPUT_IDX_INDICES)->GetDataType();
    auto indiceTypeSize = ge::GetSizeByDataType(indiceDtype_);
    // sliceUb : the required size of UB for one scatter operation;
    auto sliceUb = sliceSize * updateTypeSize + rankSize_ * indiceTypeSize;
    sliceUb = Ops::Base::CeilDiv(static_cast<uint64_t>(sliceUb), alignFactor) * alignFactor;
    OP_CHECK_IF(updateTypeSize == 0, OP_LOGE(opName, "updateTypeSize is 0"), return ge::GRAPH_FAILED);
    if (sliceUb > halfUbSize) {
        // for scatter operator. At least  rank size index need to be move in UB.
        ubTilingSize = (halfUbSize - rankSize_ * indiceTypeSize) / updateTypeSize;
    } else {
        // calculate the size of updates that need to be move in UB
        auto maxIndiceCnt = halfUbSize / sliceUb;
        ubTilingSize = maxIndiceCnt * sliceSize;
    }
    OP_LOGD(opName, "sliceUb = %lu, halfUbSize = %u, ubTilingSize = %u", sliceUb, halfUbSize, ubTilingSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::SortTiling()
{
    if (indiceShapeSize == static_cast<uint64_t>(0) || updateShapeSize == static_cast<uint64_t>(0)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        sliceSize == static_cast<uint64_t>(0), OP_LOGE(opName, "sliceSize %lu is zero. please check.", sliceSize),
        return ge::GRAPH_FAILED);
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context_);

    // 分核策略：每个核平分行数
    uint64_t rows = indiceShapeSize / rankSize_;
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(rows) + 1;
    int64_t mid = 0;
    int64_t sortTmpSize = 0;
    while (end - start > 1) {
        mid = (end + start) / TWO;
        int64_t totalIndexSize = Ops::Base::CeilAlign(mid * rankSize_ * indicesTypeSize_, ubBlockSize) + // indice
                                 Ops::Base::CeilAlign(mid * indicesTypeSize_, ubBlockSize) +             // outOfsetBuf
                                 Ops::Base::CeilAlign(mid * indicesTypeSize_, ubBlockSize) +
                                 TWO * ubBlockSize +                                                    // sortIndiceBuf
                                 Ops::Base::CeilAlign(mid * indicesTypeSize_, ubBlockSize) +            // updateOrigin
                                 Ops::Base::CeilAlign((mid + 1) * indicesTypeSize_, ubBlockSize) +      // uniqeIdCount
                                 Ops::Base::CeilAlign(RANK_MAX_VALUE * indicesTypeSize_, ubBlockSize) + // strideBuf
                                 MIN_HANDLE_SIZE * FP32_BYTES;                                          // maxScore
        sortTmpSize = GetSortTmpSize(indiceDtype_, mid, false);
        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, ubBlockSize);
        int64_t tmpToTalSize = totalIndexSize + sortTmpSize + static_cast<int64_t>(MIN_TILING_SIZE);
        if (tmpToTalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }

    ubTilingSize = static_cast<uint32_t>(start);
    uint64_t totalLoop = Ops::Base::CeilDiv(rows, static_cast<uint64_t>(ubTilingSize));
    uint64_t eachCoreLoop = Ops::Base::CeilDiv(totalLoop, static_cast<uint64_t>(totalCoreNum_));
    blockNum = Ops::Base::CeilDiv(totalLoop, eachCoreLoop);

    while (blockNum < static_cast<uint64_t>(totalCoreNum_ / TWO) && ubTilingSize > static_cast<uint32_t>(1)) {
        ubTilingSize = ubTilingSize / static_cast<uint32_t>(TWO);
        totalLoop = Ops::Base::CeilDiv(rows, static_cast<uint64_t>(ubTilingSize));
        eachCoreLoop = Ops::Base::CeilDiv(totalLoop, static_cast<uint64_t>(totalCoreNum_));
        blockNum = Ops::Base::CeilDiv(totalLoop, eachCoreLoop);
    }
    blockTilingSize = eachCoreLoop * ubTilingSize;
    tailBlockTilingSize = rows - blockTilingSize * (blockNum - 1UL);
    OP_LOGD(
        opName,
        "rows = %lld, blockTilingSize = %lld, tailBlockTilingSize = %lld,"
        "blockNum = %d ,eachCoreLoop = %d ,",
        rows, blockTilingSize, tailBlockTilingSize, blockNum, eachCoreLoop);
    return ge::GRAPH_SUCCESS;
}

uint32_t ScatterNdUpdateTiling::GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
{
    std::vector<int64_t> shapeVec = {lastAxisNum};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTilingForAscendC", "Need tmp buffer %u byte for ac sort api", maxValue);
    return maxValue;
}

int64_t ScatterNdUpdateTiling::GetRestAvailableSize(
    int64_t sampleNum, int64_t valueTypeBytes, int64_t originalSize, int64_t postAxisSize, ge::DataType idType)
{
    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t occupy = Ops::Base::CeilAlign(sampleNum * rankSize_ * indicesTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * indicesTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * (indicesTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * INT32_BYTES, ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * (INT32_BYTES * TWO), ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * indicesTypeSize_, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((varTypeSize_)*postAxisSize, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((FP32_BYTES)*postAxisSize, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((FP32_BYTES)*postAxisSize, ubBlock) +
                     GetSortTmpSize(idType, sampleNum, false);
    return originalSize - occupy;
}

void ScatterNdUpdateTiling::ComputeCoreSplitAfterAxis()
{
    eachCoreAfterAxisCount_ = Ops::Base::CeilDiv(afterAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(afterAxis_, eachCoreAfterAxisCount_);
    tailCoreAfterAxisCount_ = afterAxis_ - eachCoreAfterAxisCount_ * (usedCoreNumBefore_ - 1);
}

void ScatterNdUpdateTiling::InitFactors(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum)
{
    afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
    indicesFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
}

void ScatterNdUpdateTiling::HandleIndicesFactorGtOne(
    int64_t halfUbSize, int64_t indicesSize, int64_t alignNum, int64_t ubBlock)
{
    int64_t oneBlockSize = indicesSize + varTypeSize_ * eachCoreAfterAxisCount_;
    indicesFactor_ = halfUbSize / oneBlockSize;
    int64_t occupy =
        Ops::Base::CeilAlign(rankSize_ * indicesTypeSize_, ubBlock) + Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +
        Ops::Base::CeilAlign(indicesTypeSize_ + TWO * ALIGN_SIZE, ubBlock) +
        Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + Ops::Base::CeilAlign(INT32_BYTES + 1, ubBlock) +
        Ops::Base::CeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) + GetSortTmpSize(indiceDtype_, 1, false);
    if (occupy > halfUbSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesSize);
        indicesFactor_ = Ops::Base::CeilAlign(indicesUbSize, ALIGN_SIZE) / indicesSize;
        afterAxisFactor_ = (halfUbSize - indicesFactor_ * indicesSize) / indicesFactor_ / varTypeSize_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
        indicesFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --indicesFactor_;
            restSize =
                halfUbSize - (Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                              Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ubBlock) +
                              Ops::Base::CeilAlign(indicesFactor_ * (indicesTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                              Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) +
                              Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES + 1), ubBlock) +
                              indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_)*eachCoreAfterAxisCount_, ubBlock) +
                              GetSortTmpSize(indiceDtype_, indicesFactor_, false));
            if (indicesFactor_ > indicesAxis_) {
                indicesFactor_ = indicesAxis_;
                break;
            }
        }
    }
}

void ScatterNdUpdateTiling::HandleIndicesFactorLeOne(
    int64_t halfUbSize, int64_t indicesSize, int64_t alignNum, int64_t ubBlock)
{
    int64_t roughMaxElemByUb =
        (halfUbSize > indicesSize) ? (halfUbSize - indicesSize) / (varTypeSize_ + FP32_BYTES) : 0;
    int64_t initAfterAxis = std::min(eachCoreAfterAxisCount_, roughMaxElemByUb);
    afterAxisFactor_ = Ops::Base::FloorAlign(initAfterAxis, alignNum);
    indicesFactor_ = RoughMaxIdxByUb(afterAxisFactor_, halfUbSize, indicesSize);
    indicesFactor_ = indicesFactor_ < 1 ? 1 : indicesFactor_;
    indicesFactor_ = indicesFactor_ > indicesAxis_ ? indicesAxis_ : indicesFactor_;
    bool ok = false;
    while (true) {
        int64_t unitIdxOne = UnitIdxAligned(1, ubBlock);
        int64_t uintUpOne = UnitUpdAligned(afterAxisFactor_, ubBlock);
        int64_t maxIdxByAligned = 0;
        if (unitIdxOne + uintUpOne > 0) {
            maxIdxByAligned = (halfUbSize - GetSortTmpSize(indiceDtype_, 1, false)) / (unitIdxOne + uintUpOne);
        }
        int64_t tryIdx = std::max<int64_t>(1, std::min({indicesFactor_, maxIdxByAligned, indicesAxis_}));
        while (tryIdx >= 1) {
            int64_t occ = OccupyTotal(tryIdx, afterAxisFactor_, ubBlock);
            if (occ < halfUbSize) {
                indicesFactor_ = tryIdx;
                ok = true;
                break;
            }
            --tryIdx;
        }
        if (ok) {
            break;
        }
        afterAxisFactor_ -= alignNum;
        indicesFactor_ = RoughMaxIdxByUb(afterAxisFactor_, halfUbSize, indicesSize);
    }
}

int64_t ScatterNdUpdateTiling::UnitIdxAligned(int64_t idxFactor, int64_t ubBlock)
{
    return Ops::Base::CeilAlign(idxFactor * static_cast<int64_t>(rankSize_) * indicesTypeSize_, ubBlock) +
           Ops::Base::CeilAlign(idxFactor * indicesTypeSize_, ubBlock) +
           Ops::Base::CeilAlign(idxFactor * (indicesTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
           Ops::Base::CeilAlign(idxFactor * INT32_BYTES, ubBlock) +
           Ops::Base::CeilAlign(idxFactor * (INT32_BYTES + 1), ubBlock);
}

int64_t ScatterNdUpdateTiling::UnitUpdAligned(int64_t afterAxisFactor, int64_t ubBlock)
{
    return Ops::Base::CeilAlign(varTypeSize_ * afterAxisFactor, ubBlock) +
           Ops::Base::CeilAlign(FP32_BYTES * afterAxisFactor, ubBlock);
}

int64_t ScatterNdUpdateTiling::OccupyTotal(int64_t idxFactor, int64_t afterAxisFactor, int64_t ubBlock)
{
    int64_t indicesPart = UnitIdxAligned(idxFactor, ubBlock);
    int64_t updatesPart = idxFactor * UnitUpdAligned(afterAxisFactor, ubBlock);
    int64_t sortTmp = GetSortTmpSize(indiceDtype_, idxFactor, false);
    return indicesPart + updatesPart + sortTmp;
}

int64_t ScatterNdUpdateTiling::RoughMaxIdxByUb(int64_t afterAxisFactor, int64_t halfUbSize, int64_t indicesSize)
{
    int64_t denom = afterAxisFactor * (varTypeSize_ + FP32_BYTES) + indicesSize;
    if (denom <= 0) {
        return 1;
    }
    return halfUbSize / denom;
}

void ScatterNdUpdateTiling::DoOpTilingSplitAfter()
{
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;
    int64_t indicesSize =
        oneIndexSize + indicesTypeSize_ + (indicesTypeSize_ + TWO * ALIGN_SIZE) + INT32_BYTES + (INT32_BYTES + 1);
    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    ComputeCoreSplitAfterAxis();
    InitFactors(halfUbSize, indicesSize, alignNum);
    if (indicesFactor_ > 1) {
        HandleIndicesFactorGtOne(halfUbSize, indicesSize, alignNum, ubBlock);
    } else {
        HandleIndicesFactorLeOne(halfUbSize, indicesSize, alignNum, ubBlock);
    }
    /* 每个核分的indices相同 */
    indicesLoopSize_ = Ops::Base::CeilDiv(indicesAxis_, indicesFactor_);
    indiceTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * indicesFactor_;
    /* 主核循环次数 */
    updateLoopSize_ = Ops::Base::CeilDiv(eachCoreAfterAxisCount_, afterAxisFactor_);
    /* 主核尾loop处理afterAxis大小 */
    updateTailNum_ = eachCoreAfterAxisCount_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    /* 尾核循环次数 */
    tailUpdateLoopSize_ = Ops::Base::CeilDiv(tailCoreAfterAxisCount_, afterAxisFactor_);
    /* 尾核尾loop处理afterAxis大小 */
    tailUpdateTailNum_ = tailCoreAfterAxisCount_ - (tailUpdateLoopSize_ - 1) * afterAxisFactor_;
    isSplitAfterAxis_ = 1;
}

void ScatterNdUpdateTiling::DoOpTilingSimdSplitIndices()
{
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);

    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;

    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中:
     * indicesFactor_: indiecesQue + outOfsetBuf + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
     *                 (uniqueIdCntQue_ + 1)
     * indicesFactor_ * eachCoreAfterAxisCount_: updatesQue_
     */
    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t indicesAlignSize =
        Ops::Base::CeilAlign(oneIndexSize, ubBlock) + Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +
        Ops::Base::CeilAlign(indicesTypeSize_ + TWO * ALIGN_SIZE, ubBlock) +
        Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + Ops::Base::CeilAlign(INT32_BYTES + 1, ubBlock);

    int64_t updateAlignSize =
        Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) + GetSortTmpSize(indiceDtype_, 1, false);
    if (indicesAlignSize + updateAlignSize > halfUbSize) {
        int64_t indicesSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesAlignSize);
        /* indicesBuf_ + outOfstBuf_ */
        indicesFactor_ = Ops::Base::CeilAlign(indicesSize, ALIGN_SIZE) / indicesAlignSize;
        afterAxisFactor_ = (halfUbSize - indicesFactor_ * indicesAlignSize) / indicesFactor_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        indicesFactor_ = halfUbSize / (updateAlignSize + indicesAlignSize);
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --indicesFactor_;
            int64_t occupy = Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                             Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ubBlock) +
                             Ops::Base::CeilAlign(indicesFactor_ * (indicesTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                             Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) +
                             Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES + 1), ubBlock) +
                             indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_)*afterAxisFactor_, ubBlock) +
                             GetSortTmpSize(indiceDtype_, indicesFactor_, false);
            restSize = halfUbSize - occupy;
            if (indicesFactor_ > indicesAxis_) {
                indicesFactor_ = indicesAxis_;
                break;
            }
        }
    }
    /* 每个核分的update相同 */
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
}

void ScatterNdUpdateTiling::DoOpTilingForSimdNonDetermin()
{
    /* 优先分after */
    int64_t splitThresh = totalCoreNum_ * MIN_HANDLE_SIZE / varTypeSize_;
    if ((afterAxis_ > splitThresh) || (indicesAxis_ < (totalCoreNum_ / TWO))) {
        DoOpTilingSplitAfter();
        return;
    }
    DoOpTilingSimdSplitIndices();
    return;
}

void ScatterNdUpdateTiling::DoOpTilingForSimdMask()
{
    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t alignNum = ubBlock / varTypeSize_;
    uint64_t maskSize = static_cast<uint64_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(varInAxis_) * static_cast<int64_t>(sizeof(int8_t)), ubBlock));
    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - maskSize - RESERVE_SIZE) / DB_BUFFER);

    int64_t indicesAlignSize =
        Ops::Base::CeilAlign(oneIndexSize, ubBlock) + Ops::Base::CeilAlign(indicesTypeSize_, ubBlock);
    int64_t updateAlignSize = Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock);
    int64_t colTotalAlign = Ops::Base::CeilAlign(afterAxis_, alignNum);
    if (colTotalAlign * varTypeSize_ < ROW_THRESH_SIZE) {
        afterAxisFactor_ = colTotalAlign;
        indicesFactor_ = std::min(eachCoreIndexCount_, halfUbSize / (updateAlignSize + indicesAlignSize));
    } else {
        indicesFactor_ = ONE;
        afterAxisFactor_ = (halfUbSize - indicesAlignSize) / varTypeSize_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
        afterAxisFactor_ = std::min(colTotalAlign, afterAxisFactor_);
        isSplitOneLine_ = 1;
    }
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
}

void ScatterNdUpdateTiling::CalcDeterministicCoreSplit()
{
    calcMaskUsedCoreNum_ = Ops::Base::CeilDiv(indicesAxis_, MIN_THREAD_NUM);
    calcMaskUsedCoreNum_ = std::min(totalCoreNum_, calcMaskUsedCoreNum_);
    normCoreHandleIdx_ = Ops::Base::CeilDiv(indicesAxis_, calcMaskUsedCoreNum_);
    tailCoreHandleIdx_ = indicesAxis_ - normCoreHandleIdx_ * (calcMaskUsedCoreNum_ - 1);
    maskNormBlockLen_ = Ops::Base::FloorDiv(varInAxis_, calcMaskUsedCoreNum_);
    maskTailBlockLen_ = varInAxis_ - maskNormBlockLen_ * (calcMaskUsedCoreNum_ - 1);

    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);

    if (afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_DETERMINSTIC) {
        isDeterminSimt_ = 0;
    } else {
        isDeterminSimt_ = 1;
    }
}

void ScatterNdUpdateTiling::CalcDeterministicUpdateSplit(int64_t ubBlock)
{
    int64_t alignNum = ubBlock / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);

    int64_t updateAlignSize = Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock);
    int64_t colTotalAlign = Ops::Base::CeilAlign(afterAxis_, alignNum);
    if (colTotalAlign * varTypeSize_ < halfUbSize) {
        if (isDeterminSimt_) {
            indicesFactor_ = std::min(eachCoreIndexCount_, halfUbSize / (updateAlignSize));
            afterAxisFactor_ = afterAxis_ * indicesFactor_;
        } else {
            indicesFactor_ = ONE;
            afterAxisFactor_ = afterAxis_;
        }
    } else {
        indicesFactor_ = ONE;
        afterAxisFactor_ = halfUbSize / varTypeSize_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
        afterAxisFactor_ = std::min(colTotalAlign, afterAxisFactor_);
    }
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);

    // 一次搬多行场景
    if (afterAxis_ < afterAxisFactor_) {
        updateTailNum_ = afterAxisFactor_;
    } else {
        updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
    }
}

void ScatterNdUpdateTiling::CalcDeterministicIndicesSplit(int64_t ubBlock)
{
    uint64_t rows = indicesAxis_;
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(rows) + 1;
    int64_t mid = 0;
    int64_t sortTmpSize = 0;
    int64_t ubBlockSize = ubBlock;

    while (end - start > 1) {
        mid = (end + start) / TWO;
        int64_t totalIndexSize =
            Ops::Base::CeilAlign(mid * rankSize_ * indicesTypeSize_, ubBlockSize) +                 // indice
            Ops::Base::CeilAlign(mid * indicesTypeSize_, ubBlockSize) +                             // outOfsetBuf
            Ops::Base::CeilAlign(mid * indicesTypeSize_, ubBlockSize) + TWO * ubBlockSize +         // sortIndiceBuf
            Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(uint32_t)), ubBlockSize) +       // updateOrigin
            Ops::Base::CeilAlign((mid + 1) * static_cast<int64_t>(sizeof(uint32_t)), ubBlockSize) + // uniqeIdCount
            Ops::Base::CeilAlign(RANK_MAX_VALUE * indicesTypeSize_, ubBlockSize) +                  // strideBuf
            MIN_HANDLE_SIZE * FP32_BYTES;                                                           // maxScore
        sortTmpSize = GetSortTmpSize(indiceDtype_, mid, false);
        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, ubBlockSize);
        int64_t tmpToTalSize = totalIndexSize + sortTmpSize + static_cast<int64_t>(MIN_TILING_SIZE);
        if (tmpToTalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }

    indicesUbFactor_ = std::min(start, normCoreHandleIdx_);
    normBlockLoop_ = Ops::Base::CeilDiv(normCoreHandleIdx_, indicesUbFactor_);
    tailBlockLoop_ = Ops::Base::CeilDiv(tailCoreHandleIdx_, indicesUbFactor_);
    normBlockTail_ = normCoreHandleIdx_ - (normBlockLoop_ - 1) * indicesUbFactor_;
    tailBlockTail_ = tailCoreHandleIdx_ - (tailBlockLoop_ - 1) * indicesUbFactor_;
}

void ScatterNdUpdateTiling::DoOpTilingForDeterministic()
{
    CalcDeterministicCoreSplit();

    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    CalcDeterministicUpdateSplit(ubBlock);
    CalcDeterministicIndicesSplit(ubBlock);
}

void ScatterNdUpdateTiling::CalculateMask()
{
    int64_t eachCoreIndex = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    int64_t usedCoreNumMask = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndex);
    float ubBound = PARTIAL_UB * ubSize_;
    int64_t coreBound = MASK_CORE * usedCoreNumMask;
    int64_t varBound = MASK_VAR * varInAxis_;
    if ((varInAxis_ < ubBound) && (indicesAxis_ > varBound) && (indicesAxis_ > coreBound) &&
        (afterAxis_ > MASK_AFTER)) {
        isMask_ = 1;
    }
}
ge::graphStatus ScatterNdUpdateTiling::DoOpTiling()
{
    if (isSimdNonDeterminstic_ == 1) {
        CalculateMask();
        if (isMask_ == 1) {
            DoOpTilingForSimdMask();
        } else {
            DoOpTilingForSimdNonDetermin();
        }
    } else if (isDeterminstic_ == 1) {
        DoOpTilingForDeterministic();
    } else if (isSimtWithSort_ == 1) {
        ge::graphStatus res = SortTiling();
        if (res == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    } else {
        BlockTiling();
        ge::graphStatus res = UbTiling();
        if (res == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    ge::graphStatus status = SetStride();
    OP_CHECK_IF(ge::GRAPH_SUCCESS != status, OP_LOGE(opName, "SetStride failed."), return ge::GRAPH_FAILED);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterNdUpdateTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;

    if (indiceShapeSize < UINT32_MAX && updateShapeSize < UINT32_MAX && outputShapeSize < UINT32_MAX) {
        tilingKey = INPUT_ADDRESS_IN_INT32;
    } else {
        tilingKey = INPUT_ADDRESS_IN_INT64;
    }
    OP_LOGD(opName, "tilingKey = %lld.", tilingKey);
    return tilingKey;
}

ge::graphStatus ScatterNdUpdateTiling::GetWorkspaceSize()
{
    workspaceSize = RESERVED_WORKSPACE_SIZE;
    if (isDeterminstic_ == 1) {
        if (indiceShapeSize < UINT32_MAX && updateShapeSize < UINT32_MAX && outputShapeSize < UINT32_MAX) {
            workspaceSize = workspaceSize + (varInAxis_ + indicesAxis_ + 1) * sizeof(int32_t);
        } else {
            workspaceSize = workspaceSize + (varInAxis_ + indicesAxis_ + 1) * sizeof(int64_t);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    if (indiceShapeSize == 0UL || updateShapeSize == 0UL) {
        // 输入为空tensor时，设置blockNum为1，在kernel中直接返回
        context_->SetBlockDim(1);
    }
    if (isDeterminstic_ == 1 || isSimdNonDeterminstic_ == 1 || isMask_ == 1) {
        context_->SetBlockDim(totalCoreNum_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::SetStride()
{
    auto outPutShapePtr = context_->GetOutputShape(OUTPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outPutShapePtr);
    auto outPutShape = outPutShapePtr->GetStorageShape();
    strideList[rankSize_ - ONE] = static_cast<uint64_t>(1);
    for (int16_t dim = static_cast<int16_t>(rankSize_ - TWO); dim >= 0; --dim) {
        strideList[dim] = strideList[dim + 1] * outPutShape.GetDim(dim + 1);
    }
    return ge::GRAPH_SUCCESS;
}

void ScatterNdUpdateTiling::SetTilingData()
{
    ScatterNdUpdateRegBaseTilingData* tilingData = context_->GetTilingData<ScatterNdUpdateRegBaseTilingData>();

    tilingData->blockNum = blockNum;
    tilingData->blockTilingSize = blockTilingSize;
    tilingData->tailBlockTilingSize = tailBlockTilingSize;
    tilingData->ubTilingSize = ubTilingSize;
    tilingData->sliceSize = sliceSize;
    tilingData->rankSize = rankSize_;
    for (int32_t i = 0; i < MAX_RANK_COUNT; i++) {
        tilingData->strideList[i] = strideList[i];
    }
    for (int32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        tilingData->outPutShape[i] = outPutShape[i];
    }
    tilingData->varInAxis = varInAxis_;
    tilingData->indexRankSize = rankSize_;
    tilingData->afterAxis = afterAxis_;
    tilingData->usedCoreNumBefore = usedCoreNumBefore_;
    tilingData->usedCoreNumAfter = usedCoreNumAfter_;
    tilingData->eachCoreAfterAxisCount = eachCoreAfterAxisCount_;
    tilingData->tailCoreAfterAxisCount = tailCoreAfterAxisCount_;

    tilingData->updateLoopSize = updateLoopSize_;
    tilingData->updateTailNum = updateTailNum_;
    tilingData->indicesLoopSize = indicesLoopSize_;
    tilingData->indiceTailNum = indiceTailNum_;
    tilingData->tailUpdateLoopSize = tailUpdateLoopSize_;
    tilingData->tailUpdateAxisNum = tailUpdateTailNum_;
    tilingData->isSplitAfterAxis = isSplitAfterAxis_;
    tilingData->eachCoreIndexCount = eachCoreIndexCount_;
    tilingData->tailCoreIndexCount = tailCoreIndexCount_;
    tilingData->eachCoreVarCount = eachCoreVarCount_;
    tilingData->tailCoreVarCount = tailCoreVarCount_;
    tilingData->indicesFactor = indicesFactor_;
    tilingData->afterAxisFactor = afterAxisFactor_;
    tilingData->ubQuantaIndxFactor = ubQuantaIndxFactor_;
    tilingData->ubRowFactor = ubRowFactor_;
    tilingData->isDeterminstic = isDeterminstic_;
    tilingData->isSimtWithSort = isSimtWithSort_;
    tilingData->isSimdNonDeterminstic = isSimdNonDeterminstic_;
    tilingData->isMask = isMask_;
    tilingData->isSplitOneLine = isSplitOneLine_;
    tilingData->calcMaskUsedCoreNum = calcMaskUsedCoreNum_;
    tilingData->normCoreHandleIdx = normCoreHandleIdx_;
    tilingData->tailCoreHandleIdx = tailCoreHandleIdx_;
    tilingData->maskNormBlockLen = maskNormBlockLen_;
    tilingData->maskTailBlockLen = maskTailBlockLen_;
    tilingData->isDeterminSimt = isDeterminSimt_;

    tilingData->indicesUbFactor = indicesUbFactor_;
    tilingData->normBlockLoop = normBlockLoop_;
    tilingData->tailBlockLoop = tailBlockLoop_;
    tilingData->normBlockTail = normBlockTail_;
    tilingData->tailBlockTail = tailBlockTail_;
}

void ScatterNdUpdateTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "normCoreHandleIdx: " << normCoreHandleIdx_ << std::endl;
    info << "tailCoreHandleIdx: " << tailCoreHandleIdx_ << std::endl;
    info << "maskNormBlockLen: " << maskNormBlockLen_ << std::endl;
    info << "maskTailBlockLen: " << maskTailBlockLen_ << std::endl;
    info << "indicesFactor: " << indicesFactor_ << std::endl;
    info << "isDeterminSimt: " << isDeterminSimt_ << std::endl;
    info << "isDeterminstic: " << isDeterminstic_ << std::endl;
    info << "calcMaskUsedCoreNum: " << calcMaskUsedCoreNum_ << std::endl;
    info << "usedCoreNumBefore: " << usedCoreNumBefore_ << std::endl;
    info << "afterAxisFactor: " << afterAxisFactor_ << std::endl;
    info << "varInAxis: " << varInAxis_ << std::endl;
    info << "afterAxis: " << afterAxis_ << std::endl;
    info << "updateLoopSize: " << updateLoopSize_ << std::endl;
    info << "updateTailNum: " << updateTailNum_ << std::endl;
    info << "eachCoreIndexCount: " << eachCoreIndexCount_ << std::endl;
    info << "tailCoreIndexCount: " << tailCoreIndexCount_ << std::endl;
    info << "sliceSize: " << sliceSize << std::endl;
    info << "rankSize: " << rankSize_ << std::endl;
    OP_LOGI(opName, "Tiling info is: %s", info.str().c_str());
}

} // namespace optiling