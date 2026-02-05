/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_nd_add_tiling_base.cpp
 * \brief ascendc scatter nd add tiling cpp
 */

#include "scatter_nd_add_tiling_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

using namespace AscendC;

namespace optiling
{
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
static constexpr uint32_t ONE = 1;
static constexpr int64_t TWO = 2;
static constexpr uint32_t THREE = 3;

static constexpr uint64_t DB_BUFFER = 2;
static constexpr uint64_t RESERVE_SIZE = 256;
static constexpr int64_t ALIGN_SIZE = 32;
static constexpr int64_t MIN_HANDLE_SIZE = 128;
static constexpr int64_t MIN_SIZE_SIMD_NONDETERMINSTIC = 128;
static constexpr int64_t INDICES_MIN_BLOCK_SIZE = 1024;
static constexpr int64_t INT32_BYTES = 4;
static constexpr int64_t FP32_BYTES = 4;
static constexpr int64_t BASE_UPDATES_SIZE = 1024;
static constexpr int64_t SORT_LIMIT = 5;
static constexpr int64_t FULL_LOAD_LIMIT = 16;
static constexpr int64_t CORE_NUM = 64;
static constexpr float THRESHOLD_FACTOR = 0.1;


static const std::set<ge::DataType> DETERMIN_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16};

ge::graphStatus ScatterNdAddSimtTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"),
            return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((aivNum <= 0),
            OP_LOGE(opName, "ScatterNdAddSimtTiling fail to get totalCoreNum_."),
            return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF((ubSizePlatForm <= DCACHE_SIZE),
            OP_LOGE(opName, "ub size less than Dcache Size. please check"),
            return ge::GRAPH_FAILED);
    // UB Size Need reserve space for Dcache / CCEC Compile Stack.
    ubSize_ = ubSizePlatForm - DCACHE_SIZE;
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
            OP_LOGE(opName, "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdAddSimtTiling::GetShapeAttrsInfo()
{
    auto var = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, var);
    auto varShapeSize = var->GetShapeSize();
    OP_CHECK_IF((varShapeSize <= 0),
            OP_LOGE(opName, "var shape size is invalid(%ld)", varShapeSize),
            return ge::GRAPH_FAILED);
    auto varDesc = context_->GetInputDesc(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    auto varDtype = varDesc->GetDataType();
    varTypeSize_ = ge::GetSizeByDataType(varDtype);

    auto indices = context_->GetInputTensor(INPUT_IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    indiceShapeSize = indices->GetShapeSize();
    OP_CHECK_IF((indiceShapeSize < 0UL),
            OP_LOGE(opName,
            "update shape size is invalid(%ld)", indiceShapeSize), return ge::GRAPH_FAILED);
    auto indicesDesc = context_->GetInputDesc(INPUT_IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    indiceDtype_ = indicesDesc->GetDataType();
    indicesTypeSize_ = ge::GetSizeByDataType(indiceDtype_);
    
    auto indiceShape = indices->GetStorageShape();
    auto indiceDims = indiceShape.GetDimNum();
    rankSize_ = indiceShape.GetDim(indiceDims - 1);
    OP_CHECK_IF(
        (RANK_MIN_VALUE > static_cast<uint16_t>(rankSize_) || static_cast<uint16_t>(rankSize_) > RANK_MAX_VALUE),
        OP_LOGE(opName,
        "rankSize_ %u out of range[1, 7], please check.", rankSize_),
        return ge::GRAPH_FAILED);
    
    auto updates = context_->GetInputTensor(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updates);
    updateShapeSize = updates->GetShapeSize();
    OP_CHECK_IF((updateShapeSize < 0UL),
                    OP_LOGE(opName,
                    "update shape size is invalid(%ld)", updateShapeSize), return ge::GRAPH_FAILED);

    auto updateDesc = context_->GetInputDesc(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updateDesc);
    updateDtype_ = updateDesc->GetDataType();
    OP_CHECK_IF(
            (updateDtype_ != varDtype),
            OP_LOGE(opName, "updates [%s] and var [%s] must have the same dtype.",
                                          Ops::Base::ToString(updateDtype_).c_str(), Ops::Base::ToString(varDtype).c_str()),
          return ge::GRAPH_FAILED);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto shapeValue = outputShape->GetStorageShape();
    uint64_t shapeRank = shapeValue.GetDimNum();
    OP_CHECK_IF((shapeRank < rankSize_),
            OP_LOGE(opName,
            "shapeRank %lu less than rank %u, please check.", shapeRank, rankSize_), 
            return ge::GRAPH_FAILED);

    for (uint64_t idx = 0; idx < shapeRank; idx++) {
      outPutShape[idx] = shapeValue.GetDim(idx);
      outputShapeSize *= outPutShape[idx];
    }

    if (indiceShapeSize == 0 || updateShapeSize == 0) {
        return ge::GRAPH_SUCCESS;
    }
    // indicesAxis_ equal updatesInAxis
    indicesAxis_ = static_cast<int64_t>(indiceShapeSize / rankSize_);
    afterAxis_ = static_cast<int64_t>(updateShapeSize) / indicesAxis_;
    varInAxis_ = varShapeSize / afterAxis_;

    SelectTiling();
    return ge::GRAPH_SUCCESS;
}

void ScatterNdAddSimtTiling::SelectTiling(){
    if (context_->GetDeterministic() == 1 && (DETERMIN_DTYPE.find(updateDtype_) != DETERMIN_DTYPE.end())) {
        isDeterminstic_ = 1;
    }
    if (isDeterminstic_ != 1 && afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_NONDETERMINSTIC &&
        updateDtype_ != ge::DT_INT64) {
        isSimdNonDeterminstic_ = 1;
    }
    //是否排序
    if (indicesAxis_ / varInAxis_ >= SORT_LIMIT && updateDtype_ != ge::DT_INT64) {
        isSort_ = 1;
    }
}

void ScatterNdAddSimtTiling::BlockTiling()
{
    auto typeSize = ge::GetSizeByDataType(updateDtype_);
    alignFactor = Ops::Base::GetUbBlockSize(context_) / typeSize;
    auto blockFactor = Ops::Base::CeilDiv(updateShapeSize, static_cast<uint64_t>(totalCoreNum_));
    auto blockAlignFactor = Ops::Base::CeilDiv(blockFactor, alignFactor) * alignFactor;
    blockTilingSize = std::max(static_cast<uint64_t>(blockAlignFactor), MIN_TILING_SIZE);
    blockNum = Ops::Base::CeilDiv(updateShapeSize, blockTilingSize);
    tailBlockTilingSize = updateShapeSize - blockTilingSize * (blockNum - 1UL);
    OP_LOGD(opName,
               "updateShapeSize = %lld, blockFactor = %lld, blockAlignFactor = %lld,"
               "blockTilingSize = %d, tailBlockTilingSize = %d", updateShapeSize,
               blockFactor, blockAlignFactor, blockTilingSize, tailBlockTilingSize);
}

ge::graphStatus ScatterNdAddSimtTiling::UbTiling()
{
    if (indiceShapeSize == static_cast<uint64_t>(0) || updateShapeSize == static_cast<uint64_t>(0)) {
        return ge::GRAPH_SUCCESS;
    }
    // halfUbSize for double buffer
    auto halfUbSize = ubSize_ / DB_BUFFER;
    auto indiceNum = indiceShapeSize / rankSize_;
    sliceSize = updateShapeSize / indiceNum;
    OP_CHECK_IF(sliceSize == static_cast<uint64_t>(0),
                    OP_LOGE(opName, "sliceSize %lu is zero. please check.", sliceSize),
                    return ge::GRAPH_FAILED);
    auto updateTypeSize = ge::GetSizeByDataType(updateDtype_);
    indiceDtype_ = context_->GetInputDesc(INPUT_IDX_INDICES)->GetDataType();
    auto indiceTypeSize = ge::GetSizeByDataType(indiceDtype_);
    // sliceUb : the required size of UB for one scatter operation;
    auto sliceUb = sliceSize * updateTypeSize + rankSize_ * indiceTypeSize;
    sliceUb = Ops::Base::CeilDiv(static_cast<uint64_t>(sliceUb), alignFactor) * alignFactor;
    if (sliceUb > halfUbSize) {
        // for scatter operator. At least  rank size index need to be move in UB.
        ubTilingSize = (halfUbSize - rankSize_ * indiceTypeSize) / updateTypeSize;
    } else {
        // calculate the size of updates that need to be move in UB
        auto maxIndiceCnt = halfUbSize / sliceUb;
        ubTilingSize = maxIndiceCnt * sliceSize;
    }
    OP_LOGD(opName, "sliceUb = %lu, halfUbSize = %u, ubTilingSize = %u", sliceUb, halfUbSize,
                ubTilingSize);
    return ge::GRAPH_SUCCESS;
}

uint32_t ScatterNdAddSimtTiling::GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
{
    std::vector<int64_t> shapeVec = { lastAxisNum };
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

int64_t ScatterNdAddSimtTiling::GetRestAvailableSize(int64_t sampleNum, int64_t valueTypeBytes,
                              int64_t originalSize, int64_t postAxisSize, ge::DataType idType)
{
    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t occupy = Ops::Base::CeilAlign(sampleNum * rankSize_ * indicesTypeSize_, ubBlock) +
                    Ops::Base::CeilAlign(sampleNum * indicesTypeSize_, ubBlock) +
                    Ops::Base::CeilAlign(sampleNum * (indicesTypeSize_ + TWO * ALIGN_SIZE), ubBlock) + 
                    Ops::Base::CeilAlign(sampleNum * INT32_BYTES, ubBlock) + 
                    Ops::Base::CeilAlign(sampleNum * (INT32_BYTES * TWO), ubBlock) +
                    Ops::Base::CeilAlign(sampleNum * indicesTypeSize_, ubBlock) + 
                    sampleNum * Ops::Base::CeilAlign((varTypeSize_) * postAxisSize, ubBlock) + 
                    sampleNum * Ops::Base::CeilAlign((FP32_BYTES) * postAxisSize, ubBlock) + 
                    sampleNum * Ops::Base::CeilAlign((FP32_BYTES) * postAxisSize, ubBlock) + 
                    GetSortTmpSize(idType, sampleNum, false);
    return originalSize - occupy;
}

void ScatterNdAddSimtTiling::DoOpTilingSplitAfter()
{
    int64_t UbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE));
    UbSize = UbSize - MIN_HANDLE_SIZE * FP32_BYTES;//maxScore
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    /* split afterAxis */
    eachCoreAfterAxisCount_ = Ops::Base::CeilDiv(afterAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(afterAxis_, eachCoreAfterAxisCount_);
    tailCoreAfterAxisCount_ = afterAxis_ - eachCoreAfterAxisCount_ * (usedCoreNumBefore_ - 1);

    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中:
    * indicesFactor_: outOfsetBuf + indiecesQue + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
    *                 (uniqueIdCntQue_ + 1) + updateSumIdxQue_,
    * indicesFactor_ * eachCoreAfterAxisCount_: updatesQue_ + updateSumQue_
    */
    int64_t indicesSize = Ops::Base::CeilAlign(rankSize_ * indicesTypeSize_, ubBlock) +
                          Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +
                          Ops::Base::CeilAlign(indicesTypeSize_ + TWO * ALIGN_SIZE, ubBlock) + 
                          Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + 
                          Ops::Base::CeilAlign(INT32_BYTES * TWO, ubBlock) +
                          Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +
                          GetSortTmpSize(indiceDtype_, 1, false);
    int64_t oneBlockSize = indicesSize + (varTypeSize_ + FP32_BYTES) * eachCoreAfterAxisCount_;
    indicesFactor_ = UbSize / oneBlockSize;

    int64_t occupy = Ops::Base::CeilAlign(rankSize_ * indicesTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(indicesTypeSize_ + TWO * ALIGN_SIZE, ubBlock) + 
                     Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + 
                     Ops::Base::CeilAlign(INT32_BYTES * TWO, ubBlock) +
                     Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) + 
                     Ops::Base::CeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) + 
                     Ops::Base::CeilAlign(FP32_BYTES * eachCoreAfterAxisCount_, ubBlock) + 
                     GetSortTmpSize(indiceDtype_, 1, false);
    if (occupy > UbSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesSize);
        /* indicesBuf_ + outOfstBuf_ */
        indicesFactor_ = Ops::Base::CeilAlign(indicesUbSize, ALIGN_SIZE) / indicesSize;
        afterAxisFactor_ = (UbSize - indicesFactor_ * indicesSize) / indicesFactor_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
        indicesFactor_ = UbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
        
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --indicesFactor_;
            restSize = UbSize - (Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                                    Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ubBlock) +
                                    Ops::Base::CeilAlign(indicesFactor_ * (indicesTypeSize_ + TWO * ALIGN_SIZE), ubBlock) + 
                                    Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) + 
                                    Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES * TWO), ubBlock) +
                                    Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ubBlock) + 
                                    indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_) * eachCoreAfterAxisCount_, ubBlock) + 
                                    indicesFactor_ * Ops::Base::CeilAlign((FP32_BYTES) * eachCoreAfterAxisCount_, ubBlock) + 
                                    GetSortTmpSize(indiceDtype_, indicesFactor_, false));
            if (indicesFactor_ > indicesAxis_) {
                indicesFactor_ = indicesAxis_;
                break;
            }
        }
        // indicesFactor_ = Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ALIGN_SIZE) / indicesTypeSize_;
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

void ScatterNdAddSimtTiling::DoOpTilingForDeterminsticSplitIndices()
{
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);
    int64_t afterAxisAlignFp32 = Ops::Base::CeilAlign(afterAxis_ * FP32_BYTES, ALIGN_SIZE) / FP32_BYTES;
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;

    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);

    /* varAxis分核，用于最后计算反量化 */
    eachCoreVarCount_ = Ops::Base::CeilDiv(varInAxis_, totalCoreNum_);
    usedCoreNumAfter_ = Ops::Base::CeilDiv(varInAxis_, eachCoreVarCount_);
    tailCoreVarCount_ = varInAxis_ - eachCoreVarCount_ * (usedCoreNumAfter_ - 1);

    /* first step:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中:
      * indicesFactor_: outOfsetBuf + indiecesQue + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
      *                 (uniqueIdCntQue_ + 1) + updateSumIdxQue_,
      * indicesFactor_ * afterAxisAlign: updatesQue_ + updatesCastQue_ + updateSumQue_
    */
    int64_t indicesSize = oneIndexSize + indicesTypeSize_ + (indicesTypeSize_ + TWO * ALIGN_SIZE) + INT32_BYTES +
                          (INT32_BYTES * TWO) + indicesTypeSize_;
    int64_t oneBlockSize = indicesSize + (varTypeSize_ + FP32_BYTES + FP32_BYTES) * afterAxis_;
    indicesFactor_ = halfUbSize / oneBlockSize;
    afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_ * varTypeSize_, ALIGN_SIZE) / varTypeSize_;

    int64_t restSize = static_cast<int64_t>(-1);
    while (restSize <= 0) {
        --indicesFactor_;
        restSize = GetRestAvailableSize(indicesFactor_, varTypeSize_, halfUbSize, afterAxis_, indiceDtype_);
    }
    if (indicesFactor_ > eachCoreIndexCount_) {
        indicesFactor_ = eachCoreIndexCount_;
    }

    /* second step */
    /* sumIdx, sumQue + rValueQue + sumQuantaQue */
    oneBlockSize = indicesTypeSize_ + (FP32_BYTES + FP32_BYTES + INT32_BYTES) * afterAxis_;
    ubQuantaIndxFactor_ = halfUbSize / oneBlockSize;
    
    restSize = static_cast<int64_t>(-1);
    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    while (restSize <= 0) {
        --ubQuantaIndxFactor_;
        int64_t occupy = Ops::Base::CeilAlign(ubQuantaIndxFactor_ * indicesTypeSize_, ubBlock) +
                        ubQuantaIndxFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) + 
                        ubQuantaIndxFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) + 
                        ubQuantaIndxFactor_ * Ops::Base::CeilAlign(INT32_BYTES * afterAxis_, ubBlock);
        restSize = halfUbSize - occupy;
    }
    if (ubQuantaIndxFactor_ > eachCoreVarCount_) {
        ubQuantaIndxFactor_ = eachCoreVarCount_;   // quantaFactor Align
    }
    
    /* third step */
    /* sumQuanToIntQue + rValueQue_ + invQuanDataQue_ */
    oneBlockSize = (INT32_BYTES + FP32_BYTES + FP32_BYTES) * afterAxisAlignFp32;
    ubRowFactor_ = halfUbSize / oneBlockSize;
    restSize = static_cast<int64_t>(-1);
    while (restSize <= 0) {
        --ubRowFactor_;
        int64_t occupy = ubRowFactor_ * Ops::Base::CeilAlign(INT32_BYTES * afterAxis_, ubBlock) + 
                        ubRowFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) + 
                        ubRowFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock);
        restSize = halfUbSize - occupy;
    }
    if (ubRowFactor_ > eachCoreVarCount_) {
        ubRowFactor_ = eachCoreVarCount_;
    }

    float xValue = static_cast<float>(indicesAxis_) / varInAxis_;
    if (xValue < THRESHOLD_FACTOR) {
        isOpti_ = 1;
    }

    /* 优化分支 */
    oneBlockSize = indicesTypeSize_ + (INT32_BYTES + FP32_BYTES + FP32_BYTES) * afterAxisAlignFp32;
    ubRowOptiFactor_ = halfUbSize / oneBlockSize;
    restSize = static_cast<int64_t>(-1);
    while (restSize <= 0) {
        --ubRowOptiFactor_;
        int64_t occupy = Ops::Base::CeilAlign(ubRowOptiFactor_ * indicesTypeSize_, ubBlock) +
                        ubRowOptiFactor_ * Ops::Base::CeilAlign(INT32_BYTES * afterAxis_, ubBlock) + 
                        ubRowOptiFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) + 
                        ubRowOptiFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock);
        restSize = halfUbSize - occupy;
    }
    if (ubRowOptiFactor_ > eachCoreIndexCount_) {
        ubRowOptiFactor_ = eachCoreIndexCount_;
    }
}

void ScatterNdAddSimtTiling::DoOpTilingSimdSplitIndices()
{
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);
    halfUbSize = halfUbSize - MIN_HANDLE_SIZE * FP32_BYTES;//maxScore

    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;
    
    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中:
      * indicesFactor_: indiecesQue + outOfsetBuf + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
      *                 (uniqueIdCntQue_ + 1) + updateSumIdxQue_,
      * indicesFactor_ * eachCoreAfterAxisCount_: updatesQue_ + updateSumQue_
      */
      auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
      int64_t indicesAlignSize = Ops::Base::CeilAlign(oneIndexSize, ubBlock) +
                                Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +
                                Ops::Base::CeilAlign(indicesTypeSize_ + TWO * ALIGN_SIZE, ubBlock) + 
                                Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + 
                                Ops::Base::CeilAlign(INT32_BYTES * TWO, ubBlock) +
                                Ops::Base::CeilAlign(indicesTypeSize_, ubBlock);

      int64_t updateAlignSize = Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) + 
                                Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) + 
                                GetSortTmpSize(indiceDtype_, 1, false);
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
                            Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES * TWO), ubBlock) +
                            Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ubBlock) + 
                            indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_) * afterAxisFactor_, ubBlock) + 
                            indicesFactor_ * Ops::Base::CeilAlign((FP32_BYTES) * afterAxisFactor_, ubBlock) + 
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

void ScatterNdAddSimtTiling::DoOpTilingForDeterminstic()
{
    /* 优先分after */
    int64_t splitThresh = totalCoreNum_ * MIN_HANDLE_SIZE / varTypeSize_;
    if ((afterAxis_ > splitThresh) || (indicesAxis_ < (totalCoreNum_ / TWO))) {
        DoOpTilingSplitAfter();
        return;
    }
    DoOpTilingForDeterminsticSplitIndices();
    return;
}

void ScatterNdAddSimtTiling::DoOpTilingForSimdNonDetermin()
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

ge::graphStatus ScatterNdAddSimtTiling::DoOpTiling()
{
    if (isDeterminstic_ == 1) {
        DoOpTilingForDeterminstic();
    } else if (isSimdNonDeterminstic_ == 1) {
        DoOpTilingForSimdNonDetermin();
    }else {
        if (isSort_ == 1){
           DoOpTilingSimdSplitIndices();
        }else{
            BlockTiling();
            ge::graphStatus res = UbTiling();
            if (res == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
            }
        }   
    }
    SetStride();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdAddSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterNdAddSimtTiling::GetTilingKey() const
{
    uint64_t tilingKey = updateDtype_;

    if (indiceShapeSize < UINT32_MAX && updateShapeSize < UINT32_MAX && outputShapeSize < UINT32_MAX) {
        tilingKey += INPUT_ADDRESS_IN_INT32;
    } else {
        tilingKey += INPUT_ADDRESS_IN_INT64;
    }
    OP_LOGD(opName, "tilingKey = %lld.", tilingKey);
    return tilingKey;
}

ge::graphStatus ScatterNdAddSimtTiling::GetWorkspaceSize()
{
    workspaceSize = RESERVED_WORKSPACE_SIZE;
    /* 非量化不需要开workspace */
    if (isDeterminstic_ == 1 && isSplitAfterAxis_ == 0) {
        int64_t rCoutWsSize = varInAxis_ * INT32_BYTES;
        int64_t rValueWsSize = varInAxis_ * afterAxis_ * FP32_BYTES;
        int64_t sumQuantaIntWsSize = varInAxis_ * afterAxis_ * INT32_BYTES;
        int64_t sumWsSize = totalCoreNum_ * eachCoreIndexCount_ * afterAxis_ * FP32_BYTES;
        int64_t sumIdxWsSize = totalCoreNum_ * eachCoreIndexCount_ * indicesTypeSize_;
        workspaceSize += static_cast<uint64_t>(rCoutWsSize + rValueWsSize + sumQuantaIntWsSize + sumWsSize + sumIdxWsSize);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdAddSimtTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    context_->SetScheduleMode(1);
    if (indiceShapeSize == 0 || updateShapeSize == 0) {
        // 输入为空tensor时，设置blockNum为1，在kernel中直接返回
        context_->SetBlockDim(1);
    }
    if (isSimdNonDeterminstic_ == 1 || isSort_ == 1) {
        context_->SetBlockDim(usedCoreNumBefore_);
    }
    if (isDeterminstic_ == 1) {
        int64_t usedCoreNum = 0;
        if (isOpti_ == 1) {
            usedCoreNum = usedCoreNumBefore_;
        } else {
            usedCoreNum = std::max(usedCoreNumAfter_, usedCoreNumBefore_);
        }
        context_->SetBlockDim(usedCoreNum);
    }
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ScatterNdAddSimtTiling::SetStride()
{
    auto outPutShape = context_->GetOutputShape(OUTPUT_IDX_SHAPE)->GetStorageShape();
    strideList[rankSize_ - ONE] = static_cast<uint64_t>(1);
    for (int16_t dim = static_cast<int16_t>(rankSize_ - TWO); dim >= 0; --dim) {
        strideList[dim] = strideList[dim + 1] * outPutShape.GetDim(dim + 1);
    }
}

void ScatterNdAddSimtTiling::SetTilingData()
{
    tilingData.set_blockNum(blockNum);
    tilingData.set_blockTilingSize(blockTilingSize);
    tilingData.set_tailBlockTilingSize(tailBlockTilingSize);
    tilingData.set_ubTilingSize(ubTilingSize);
    tilingData.set_sliceSize(sliceSize);
    tilingData.set_rankSize(rankSize_);
    tilingData.set_strideList(strideList);
    tilingData.set_outPutShape(outPutShape);

    tilingData.set_varInAxis(varInAxis_);
    tilingData.set_indexRankSize(rankSize_);
    tilingData.set_afterAxis(afterAxis_);
    tilingData.set_usedCoreNumBefore(usedCoreNumBefore_);
    tilingData.set_usedCoreNumAfter(usedCoreNumAfter_);
    tilingData.set_eachCoreAfterAxisCount(eachCoreAfterAxisCount_);
    tilingData.set_tailCoreAfterAxisCount(tailCoreAfterAxisCount_);

    tilingData.set_updateLoopSize(updateLoopSize_);
    tilingData.set_updateTailNum(updateTailNum_);
    tilingData.set_indicesLoopSize(indicesLoopSize_);
    tilingData.set_indiceTailNum(indiceTailNum_);
    tilingData.set_tailUpdateLoopSize(tailUpdateLoopSize_);
    tilingData.set_tailUpdateAxisNum(tailUpdateTailNum_);
    tilingData.set_isSplitAfterAxis(isSplitAfterAxis_);
    tilingData.set_eachCoreIndexCount(eachCoreIndexCount_);
    tilingData.set_tailCoreIndexCount(tailCoreIndexCount_);
    tilingData.set_eachCoreVarCount(eachCoreVarCount_);
    tilingData.set_tailCoreVarCount(tailCoreVarCount_);
    tilingData.set_indicesFactor(indicesFactor_);
    tilingData.set_afterAxisFactor(afterAxisFactor_);
    tilingData.set_ubQuantaIndxFactor(ubQuantaIndxFactor_);
    tilingData.set_ubRowFactor(ubRowFactor_);
    tilingData.set_isDeterminstic(isDeterminstic_);
    tilingData.set_isSimdNonDeterminstic(isSimdNonDeterminstic_);
    tilingData.set_isSort(isSort_);
    tilingData.set_ubRowOptiFactor(ubRowOptiFactor_);
    tilingData.set_isOpti(isOpti_);
}


std::string ScatterNdAddSimtTiling::TilingDataToString()
{
    // 基础分块参数
    std::string result = "blockNum: " + std::to_string(tilingData.get_blockNum());
    result += "blockTilingSize: " + std::to_string(tilingData.get_blockTilingSize());
    result += "tailBlockTilingSize: " + std::to_string(tilingData.get_tailBlockTilingSize());
    result += "ubTilingSize: " + std::to_string(tilingData.get_ubTilingSize());
    result += "sliceSize: " + std::to_string(tilingData.get_sliceSize());
    result += "rankSize: " + std::to_string(tilingData.get_rankSize());

    // 轴相关参数
    result +="varInAxis: " + std::to_string(tilingData.get_varInAxis());
    result +="indexRankSize: " + std::to_string(tilingData.get_indexRankSize());
    result +="afterAxis: " + std::to_string(tilingData.get_afterAxis());
    result +="usedCoreNumBefore: " + std::to_string(tilingData.get_usedCoreNumBefore());
    result +="usedCoreNumAfter: " + std::to_string(tilingData.get_usedCoreNumAfter());
    result +="eachCoreAfterAxisCount: " + std::to_string(tilingData.get_eachCoreAfterAxisCount());
    result +="tailCoreAfterAxisCount: " + std::to_string(tilingData.get_tailCoreAfterAxisCount());

    // 循环与索引相关参数
    result +="updateLoopSize: " + std::to_string(tilingData.get_updateLoopSize());
    result +="updateTailNum: " + std::to_string(tilingData.get_updateTailNum());
    result +="indicesLoopSize: " + std::to_string(tilingData.get_indicesLoopSize());
    result +="indiceTailNum: " + std::to_string(tilingData.get_indiceTailNum());
    result +="tailUpdateLoopSize: " + std::to_string(tilingData.get_tailUpdateLoopSize());
    result +="tailUpdateAxisNum: " + std::to_string(tilingData.get_tailUpdateAxisNum());
    result +="isSplitAfterAxis: " + std::to_string(tilingData.get_isSplitAfterAxis());
    result +="eachCoreIndexCount: " + std::to_string(tilingData.get_eachCoreIndexCount());
    result +="tailCoreIndexCount: " + std::to_string(tilingData.get_tailCoreIndexCount());
    result +="eachCoreVarCount: " + std::to_string(tilingData.get_eachCoreVarCount());
    result +="tailCoreVarCount: " + std::to_string(tilingData.get_tailCoreVarCount());
    result +="indicesFactor: " + std::to_string(tilingData.get_indicesFactor());
    result +="afterAxisFactor: " + std::to_string(tilingData.get_afterAxisFactor());
    result +="ubQuantaIndxFactor: " + std::to_string(tilingData.get_ubQuantaIndxFactor());
    result +="ubRowFactor: " + std::to_string(tilingData.get_ubRowFactor());
    result +="isDeterminstic: " + std::to_string(tilingData.get_isDeterminstic());
    result +="isSimdNonDeterminstic: " + std::to_string(tilingData.get_isSimdNonDeterminstic());
    result +="isSort: " + std::to_string(tilingData.get_isSort());
    result +="ubRowOptiFactor: " + std::to_string(tilingData.get_ubRowOptiFactor());
    result +="isOpti: " + std::to_string(tilingData.get_isOpti());
    return result;
}

void ScatterNdAddSimtTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "Tiling info is: %s", TilingDataToString().c_str());
}

} // namespace optiling