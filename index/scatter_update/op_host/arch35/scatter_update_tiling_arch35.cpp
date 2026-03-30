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
 * \file scatter_update_tiling.cc
 * \brief
 */

#include "scatter_update_tiling_arch35.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/util/math_util.h"

using namespace AscendC;

namespace optiling
{
constexpr int64_t VAR_IDX = 0;
constexpr int64_t INDICES_IDX = 1;
constexpr int64_t UPDATES_IDX = 2;
constexpr int64_t UPDATES_NOT_MASK = 0;
constexpr uint64_t MAX_THREAD_NUM = 2048;
constexpr uint64_t DCACHE_SIZE = 128 * 1024;
constexpr uint64_t DCACHE_SIZE1 = 32 * 1024;
constexpr uint64_t DCACHE_SIZE_DETERMINISTIC = 32 * 1024;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;
constexpr uint64_t UPDATES_IN_SIMD = 1;
constexpr uint64_t VAR_TAIL_DIM_SIZE = 256;
constexpr uint64_t INDICES_ALIGN_32 = 32;
constexpr uint64_t MIN_SIZE_SORT_INDICES_64 = 64;
constexpr uint32_t TILING_KEY_PLACE_HOLD = 3;
constexpr uint32_t DB_BUFFER = 2;
constexpr uint32_t DOUBLE = 2;
constexpr uint64_t EXTRA_BYTE_FOR_COUNT = 64;
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint64_t SPLIT_ROW_INDICES_NUM = 64;
constexpr uint64_t indicesFactorLimit = 1024;
constexpr uint64_t RESERVE_ROW_SIZE = 16 * 1024;
constexpr uint64_t HASH_BUCKET_SIZE = 128 * sizeof(float);
constexpr size_t VAR_SHAPE_LENGTH = 2;
constexpr uint64_t SIMD_MIN_SIZE_SORT_INDICES = 500;
constexpr uint32_t INT64_BLOCK_ALIGN_NUM = 4;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t INT8_BLOCK_ALIGN_NUM = 32;
constexpr float UB_SIZE_SCALE = 0.1;
constexpr uint64_t CORE_NUM_SCALE = 1000;
constexpr uint64_t INDICES_SCALE = 5;
constexpr uint64_t INDICES_MAX_BATCH_COPY_THRESHOLD = 4UL * 1024UL;
constexpr uint64_t MULTI_LINE_THRESHOLD = 2UL * 1024UL;
constexpr uint64_t DETERMINISTIC_MIN_COL = 128;
constexpr uint64_t MIN_INDICES_SIZE = 1024;
constexpr uint64_t CASTMODE1 = 1;   // int32 Cast int16
constexpr uint64_t CASTMODE2 = 2;   // int64 Cast int32
constexpr uint64_t CASTMODE3 = 3;   // int64 Cast int16
constexpr uint64_t CASTMODE4 = 4;   // int32 Cast uint8
constexpr uint64_t CASTMODE5 = 5;   // int64 Cast uint8
constexpr uint64_t TILINGKEYOFFSET = uint64_t(10000000000000000000UL); // 10^19
constexpr uint64_t DETERMINISTIC_MODE_ROW = 2;
constexpr uint64_t TILING_KEY_PARAM_INTERVAL = 10;

static const std::unordered_map<ge::DataType, uint64_t> INDICE_DATA_TYPE{{ge::DataType::DT_INT32, 1},
                                                                         {ge::DataType::DT_INT64, 2}};

static const std::unordered_map<ge::DataType, uint64_t> VAR_DATA_TYPE{{ge::DataType::DT_INT32, 1},
                                                                      {ge::DataType::DT_INT8, 2},
                                                                      {ge::DataType::DT_UINT8, 3},
                                                                      {ge::DataType::DT_FLOAT16, 4},
                                                                      {ge::DataType::DT_FLOAT, 5},
                                                                      {ge::DataType::DT_BF16, 6},
                                                                      {ge::DataType::DT_INT64, 7},
                                                                      {ge::DataType::DT_UINT32, 8},
                                                                      {ge::DataType::DT_UINT64, 9},
                                                                      {ge::DataType::DT_FLOAT8_E4M3FN, 10},
                                                                      {ge::DataType::DT_FLOAT8_E5M2, 11},
                                                                      {ge::DataType::DT_FLOAT8_E8M0, 12}};

ge::graphStatus ScatterUpdateTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to get platform info"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((aivNum <= 0), VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to get coreNum."),
                    return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    // UB Size Need reserve space for Dcache / CCEC Compile Stack.
    ubSize_ = ubSizePlatForm;
    if (isDeterministic_) {
        if ((varShape_[1] * varTypeSize_ > DETERMINISTIC_MIN_COL * totalCoreNum_ / DOUBLE) || (indicesSize_ < totalCoreNum_)) {
            isDeterministicSplitCol_ = 1;
        } else {
            ubSize_ = ubSizePlatForm - DCACHE_SIZE_DETERMINISTIC;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (isSimt_) {
        OP_TILING_CHECK((ubSizePlatForm <= DCACHE_SIZE),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "ub size less than Dcache Size. please check"),
                        return ge::GRAPH_FAILED);
        ubSize_ = ubSizePlatForm - DCACHE_SIZE;
    } else {
        ubSize_ = ubSizePlatForm - DCACHE_SIZE1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterUpdateTiling::GetShapeAttrsInfo()
{
    auto var = context_->GetInputShape(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, var);

    auto varViewStride = context_->GetInputStride(VAR_IDX);
    gert::Shape varShape;

    // 非连续三个条件：InputIsView为true varViewStride不为空 varViewStride.GetDimNum()不为0
    if (context_->InputIsView(VAR_IDX) && varViewStride != nullptr && varViewStride->GetDimNum() != 0) {
        varShape = var->GetShape();
        OP_LOGD(opName, "start processing noncontinuous scenario shape.");
        varStride_ = varViewStride->GetStride(0);
        varSize_ = varShape.GetShapeSize();
        varShape_[0] = varShape.GetDim(0);
        varShape_[1] = varShape_[0] == 0 ? varShape_[1] : varSize_ / varShape_[0];
    } else {
        varShape = var->GetStorageShape();
        OP_TILING_CHECK(varShape.IsScalar(),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "the parameter var cannot be scalar, please check."),
                    return ge::GRAPH_FAILED);
        varSize_ = varShape.GetShapeSize();
        varShape_[0] = varShape.GetDim(0);
        if (varShape_[0] != 0) {
            varShape_[1] = varSize_ / varShape_[0];
            varStride_ = varShape_[1];
        }
    }

    auto indices = context_->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    auto indiceShape = indices->GetStorageShape();
    indicesSize_ = indiceShape.GetShapeSize();

    auto updates = context_->GetInputShape(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updates);
    auto updateShape = updates->GetStorageShape();
    updatesSize_ = updateShape.GetShapeSize();
    uint64_t updateDims = updateShape.GetDimNum();
    if ((updateDims == 1 && updatesSize_ == 1) || updateDims == 0) {
        isUpdateScalar_ = 1;
    } else {
        OP_TILING_CHECK(CheckUpdatesShape(varShape, indiceShape, updateShape) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "update shape check failed."), return ge::GRAPH_FAILED);
    }
    updateShape_[0] = indicesSize_;
    updateShape_[1] = varShape_[1];
    OP_TILING_CHECK(CheckInputDtype() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "input dtype check failed."), return ge::GRAPH_FAILED);

    if (context_->GetDeterministic() && !isUpdateScalar_) {
        isDeterministic_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterUpdateTiling::CheckInputDtype()
{
    auto indicesPtr = context_->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesPtr);
    indicesDtype_ = indicesPtr->GetDataType();
    OP_TILING_CHECK(
        (INDICE_DATA_TYPE.find(indicesDtype_) == INDICE_DATA_TYPE.end()),
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "indices dtype only support int32/int64, but got [%s].", Ops::Base::ToString(indicesDtype_).c_str()),
        return ge::GRAPH_FAILED);
    indicesDtypeSize_ = ge::GetSizeByDataType(indicesDtype_);
    OP_TILING_CHECK(indicesDtypeSize_ <= 0, VECTOR_INNER_ERR_REPORT_TILIING(opName, "get indicesDtype size fail."),
                    return ge::GRAPH_FAILED);
    auto dataPtr = context_->GetInputDesc(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataPtr);
    varDtype_ = dataPtr->GetDataType();
    OP_TILING_CHECK((VAR_DATA_TYPE.find(varDtype_) == VAR_DATA_TYPE.end()),
        VECTOR_INNER_ERR_REPORT_TILIING(opName, 
            "var dtype only support int8/uint8/int32/uint32/int64/uint64/float32/float16/bfloat16/FLOAT8_E8M0/FLOAT8_E5M2/FLOAT8_E4M3FN, but got [%s].", Ops::Base::ToString(varDtype_).c_str()),
        return ge::GRAPH_FAILED);
    varTypeSize_ = ge::GetSizeByDataType(varDtype_);
    OP_TILING_CHECK(varTypeSize_ <= 0, VECTOR_INNER_ERR_REPORT_TILIING(opName, "get dataType size fail."),
                    return ge::GRAPH_FAILED);
    isSimt_ = varShape_[1] * varTypeSize_ < VAR_TAIL_DIM_SIZE;
    auto updatePtr = context_->GetInputDesc(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatePtr);
    auto updatesType = updatePtr->GetDataType();
    OP_TILING_CHECK(
        (VAR_DATA_TYPE.find(updatesType) == VAR_DATA_TYPE.end()),
        VECTOR_INNER_ERR_REPORT_TILIING(opName, 
            "updates dtype only support int8/uint8/int32/uint32/int64/uint64/float32/float16/bfloat16/FLOAT8_E8M0/FLOAT8_E5M2/FLOAT8_E4M3FN, but got [%s].", Ops::Base::ToString(updatesType).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (updatesType != varDtype_),
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "expected updates dtype to be equal to var dtype, please check."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterUpdateTiling::CheckUpdatesShape(const gert::Shape& varShape, const gert::Shape& indicesShape,
                                                    const gert::Shape& updatesShape)
{
    uint64_t varDimNum = static_cast<uint64_t>(varShape.GetDimNum());
    uint64_t indicesDimNum = static_cast<uint64_t>(indicesShape.GetDimNum());
    uint64_t updatesDimNum = static_cast<uint64_t>(updatesShape.GetDimNum());
    OP_TILING_CHECK(
        (updatesDimNum != indicesDimNum + varDimNum - 1),
        VECTOR_INNER_ERR_REPORT_TILIING(
            opName, "updatesDimNum must have the same number of indicesDimNum add varDimNum - 1, please check."),
        return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < indicesDimNum; i++) {
        OP_TILING_CHECK(
            (static_cast<uint32_t>(updatesShape.GetDim(i)) != static_cast<uint32_t>(indicesShape.GetDim(i))),
            VECTOR_INNER_ERR_REPORT_TILIING(
                opName, "updatesShape should be equal to the shape of 'indices' concats the shape of 'var' except for the first dimension."),
            return ge::GRAPH_FAILED);
    }

    for (uint64_t i = 1; i < varDimNum; i++) {
        OP_TILING_CHECK((static_cast<uint32_t>(updatesShape.GetDim(i + indicesDimNum - 1)) !=
                        static_cast<uint32_t>(varShape.GetDim(i))),
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            opName, "updatesShape should be equal to the shape of 'indices' concats the shape of 'var' except for the first dimension."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void ScatterUpdateTiling::SetTilingData()
{
    ScatterUpdateTilingData *tilingData = context_->GetTilingData<ScatterUpdateTilingData>();
    tilingData->varShape[0] = varShape_[0];
    tilingData->varShape[1] = varShape_[1];
    tilingData->indicesSize = indicesSize_;
    tilingData->normBlockIndices = normBlockIndices_;
    tilingData->tailBlockIndices = tailBlockIndices_;
    tilingData->indicesFactor = indicesFactor_;
    tilingData->normBlockLoop = normBlockLoop_;
    tilingData->tailBlockLoop = tailBlockLoop_;
    tilingData->normBlockTail = normBlockTail_;
    tilingData->tailBlockTail = tailBlockTail_;
    tilingData->rowTotal = rowTotalNum_;
    tilingData->colTotal = colTotalNum_;
    tilingData->rowBase = rowNormalNum_;
    tilingData->colBase = colNormalNum_;
    tilingData->rowTail = rowTailNum_;
    tilingData->colTail = colTailNum_;
    tilingData->rowTileNum = rowTileNum_;
    tilingData->colTileNum = colTileNum_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->normBlockColNum = normBlockColNum_;
    tilingData->normBlockRowNum = normBlockRowNum_;
    tilingData->tailBlockColNum = tailBlockColNum_;
    tilingData->tailBlockRowNum = tailBlockRowNum_;
    tilingData->normNeedSplitRow = normNeedSplitRow_;
    tilingData->tailNeedSplitRow = tailNeedSplitRow_;
    tilingData->processRowPerUb = processRowPerUb_;
    tilingData->processColNum = processColNum_;
    tilingData->rowLoopByUb = rowLoopByUb_;
    tilingData->processRowTail = processRowTail_;
    tilingData->indicesUbFactor = indicesUbFactor_;
    tilingData->updateUbSize = updateUbSize_;
    tilingData->processColPerUb = processColPerUb_;
    tilingData->colLoopByUb = colLoopByUb_;
    tilingData->processColTail = processColTail_;
    tilingData->indicesBatchCopySizeAlign = indicesBatchCopySizeAlign_;
    tilingData->varStride = varStride_;
    tilingData->updateColUbFactor = updateColUbFactor_;
    tilingData->indicesLoopSize = indicesLoopSize_;
    tilingData->indicesTailLoopNum = indicesTailLoopNum_;
    tilingData->updatesNormBlockLoop = updatesNormBlockLoop_;
    tilingData->updatesTailBlockLoop = updatesTailBlockLoop_;
    tilingData->updatesNormBlockTailLoopSize = updatesNormBlockTailLoopSize_;
    tilingData->updatesTailBlockTailLoopSize = updatesTailBlockTailLoopSize_;
    tilingData->maskNormBlockLen = maskNormBlockLen_;
    tilingData->maskTailBlockLen = maskTailBlockLen_;
    tilingData->isIndicesSizeInt64 = isIndicesSizeInt64_;
    tilingData->indicesCastMode = indicesCastMode_;
}

static uint64_t GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
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

void ScatterUpdateTiling::DoMaskSimdTiling()
{
    templateKey_ = UPDATES_IN_SIMD;
    uint32_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    uint64_t aglinNum = static_cast<uint64_t>(ubBlock / varTypeSize_);
    tailBlockRowNum_ = rowTotalNum_ - (usedCoreNum_ - 1UL) * normBlockRowNum_;

    uint64_t maskSize = Ops::Base::CeilAlign(varShape_[0], static_cast<uint64_t>(ubBlock));
    uint64_t colTotalAlign = Ops::Base::CeilAlign(colTotalNum_, aglinNum);
    
    uint64_t indicesBatchCopyUbSize = (normBlockRowNum_ * indicesDtypeSize_ > INDICES_MAX_BATCH_COPY_THRESHOLD) ?
                                      INDICES_MAX_BATCH_COPY_THRESHOLD :
                                      normBlockRowNum_ * indicesDtypeSize_;
    indicesBatchCopySizeAlign_ = Ops::Base::CeilAlign(indicesBatchCopyUbSize, static_cast<uint64_t>(ubBlock));
    uint64_t restUbSize = ubSize_ - maskSize - static_cast<uint64_t>(DOUBLE * ubBlock);
    if (colTotalAlign * varTypeSize_ < MULTI_LINE_THRESHOLD) {
        processRowPerUb_ = static_cast<uint64_t>(restUbSize / DB_BUFFER) / (colTotalAlign * varTypeSize_ + indicesDtypeSize_);
        processRowPerUb_ = std::min(processRowPerUb_, normBlockRowNum_);
        processColPerUb_ = colTotalAlign;
    } else {
        processColPerUb_ = static_cast<uint64_t>((restUbSize - indicesBatchCopySizeAlign_) / DB_BUFFER) / varTypeSize_;
        processColPerUb_ = Ops::Base::FloorDiv(processColPerUb_, aglinNum);
        processColPerUb_ = std::min(processColPerUb_, colTotalAlign);
        processRowPerUb_ = 1UL;
        normNeedSplitRow_ = 1UL;
    }
    updateUbSize_ = processColPerUb_ * varTypeSize_ * processRowPerUb_;
    normBlockLoop_ = Ops::Base::CeilDiv(normBlockRowNum_, processRowPerUb_);
    tailBlockLoop_ = Ops::Base::CeilDiv(tailBlockRowNum_, processRowPerUb_);

    colLoopByUb_ = Ops::Base::CeilDiv(colTotalAlign, processColPerUb_);
    processColTail_ = colTotalNum_ - ((colLoopByUb_ - 1UL) * processColPerUb_);
}

void ScatterUpdateTiling::CalcMask()
{
    rowTotalNum_ = updateShape_[0];
    colTotalNum_ = updateShape_[1];
    normBlockRowNum_ = Ops::Base::CeilDiv(rowTotalNum_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(rowTotalNum_, normBlockRowNum_);
    if ((varShape_[0] * sizeof(uint8_t)  < static_cast<uint64_t>(UB_SIZE_SCALE * static_cast<float>(ubSize_))) && (indicesSize_ > usedCoreNum_ * CORE_NUM_SCALE) &&
        (indicesSize_ > INDICES_SCALE * varShape_[0]) && !isSimt_) {
        isMask_ = 1UL;
    }
}

uint64_t ScatterUpdateTiling::CalcSimtIndicesUbFactor()
{
    uint32_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    uint32_t indicesBuff = 1;
    uint32_t posIdexBuff = 1;
    uint64_t mid = 0;
    uint64_t start = 1;
    uint64_t end = indicesSize_;
    while(end - start > 1) {
        mid = (end + start) / static_cast<int64_t>(DOUBLE);
        // 4个LocalTensor： <indicesDtype>indicesLocal;  <indicesDtype>indicesSortedLocal;  <uint32_t>updatesOriginIdx;  <int32_t>uniqueIdCountLocal
        int64_t totalIndexSize = 0;
        int64_t sortTmpSize = 0;
        if (indicesCastMode_ == 0) {
            totalIndexSize = Ops::Base::CeilAlign(mid * static_cast<uint64_t>(indicesDtypeSize_), static_cast<uint64_t>(ubBlock)) * (indicesBuff + posIdexBuff) +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(sizeof(uint32_t)), static_cast<uint64_t>(ubBlock)) +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(sizeof(int32_t)), static_cast<uint64_t>(ubBlock));
            sortTmpSize = GetSortTmpSize(indicesDtype_, mid, false);
        } else {
            totalIndexSize = Ops::Base::CeilAlign(mid * static_cast<uint64_t>(indicesCastDtypeSize_), static_cast<uint64_t>(ubBlock)) * (indicesBuff + posIdexBuff) +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(indicesDtypeSize_), static_cast<uint64_t>(ubBlock)) +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(sizeof(uint32_t)), static_cast<uint64_t>(ubBlock)) +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(sizeof(int32_t)), static_cast<uint64_t>(ubBlock));
            sortTmpSize = GetSortTmpSize(indicesCastDtype_, mid, false);
        }

        int64_t tmpTotalSize = totalIndexSize + Ops::Base::CeilAlign(sortTmpSize, static_cast<int64_t>(ubBlock)) + EXTRA_BYTE_FOR_COUNT * DOUBLE + HASH_BUCKET_SIZE;
        if (tmpTotalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }
    return start;
}

void ScatterUpdateTiling::DoSimtTiling()
{
    GetCastType();
    uint64_t totalSize = varShape_[1] * indicesSize_;
    usedCoreNum_ = Ops::Base::CeilDiv(totalSize, MAX_THREAD_NUM);
    usedCoreNum_ = std::min(usedCoreNum_, totalCoreNum_);
    
    if (isSort_) {        
        // 单次UB可处理indices个数
        indicesFactor_ = CalcSimtIndicesUbFactor();
        uint64_t totalLoop = Ops::Base::CeilDiv(static_cast<uint64_t>(indicesSize_), indicesFactor_);
        // 按照UB总循环次数分核
        normBlockLoop_ = Ops::Base::CeilDiv(totalLoop, totalCoreNum_);
        usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, normBlockLoop_);
        tailBlockLoop_ = totalLoop - normBlockLoop_ * (usedCoreNum_ - 1);
        tailBlockTail_ = indicesSize_ - indicesFactor_ * normBlockLoop_ * (usedCoreNum_ - 1) - indicesFactor_ * (tailBlockLoop_ - 1);
        while(usedCoreNum_ < totalCoreNum_ / DOUBLE && indicesFactor_ > indicesFactorLimit) {
            indicesFactor_ = indicesFactor_ / DOUBLE;
            totalLoop = Ops::Base::CeilDiv(static_cast<uint64_t>(indicesSize_), indicesFactor_);
            normBlockLoop_ = Ops::Base::CeilDiv(totalLoop, totalCoreNum_);
            usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, normBlockLoop_);
            tailBlockLoop_ = totalLoop - normBlockLoop_ * (usedCoreNum_ - 1);
            tailBlockTail_ = indicesSize_ - indicesFactor_ * normBlockLoop_ * (usedCoreNum_ - 1) - indicesFactor_ * (tailBlockLoop_ - 1);
        }
        normBlockIndices_ = indicesFactor_ * normBlockLoop_;
    }
}

uint64_t ScatterUpdateTiling::CalcIndicesUbFactor()
{
    GetCastType();
    uint64_t mid = 0;
    uint64_t start = 1;
    uint64_t end = normBlockIndices_ + 1;
    uint64_t sortTmpSize = 0;
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    while(end - start > 1) {
        mid = (end + start) / static_cast<uint64_t>(DOUBLE);
        uint64_t totalIndexSize = 0;
        // 所需空间：indicesLocal + indicesSortedLocal + updatesOriginIdxLocal + uniqueIdCountLocal
        if (indicesCastMode_ == 0) {
            totalIndexSize = Ops::Base::CeilAlign(mid * indicesDtypeSize_, ubBlock) * DOUBLE +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(sizeof(uint32_t)), ubBlock) + ubBlock * DOUBLE +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(sizeof(int32_t)), ubBlock) + ubBlock * DOUBLE;
            sortTmpSize = GetSortTmpSize(indicesDtype_, mid, false);
        } else {
            totalIndexSize = Ops::Base::CeilAlign(mid * indicesCastDtypeSize_, ubBlock) * DOUBLE +
                             Ops::Base::CeilAlign(mid * indicesDtypeSize_, ubBlock) +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(sizeof(uint32_t)), ubBlock) + ubBlock * DOUBLE +
                             Ops::Base::CeilAlign(mid * static_cast<uint64_t>(sizeof(int32_t)), ubBlock) + ubBlock * DOUBLE;
            sortTmpSize = GetSortTmpSize(indicesCastDtype_, mid, false);
        }

        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, ubBlock);
        uint64_t updateSize = Ops::Base::CeilAlign(mid * varShape_[1] * varTypeSize_, ubBlock);
        uint64_t tmpTotalSize = totalIndexSize + sortTmpSize + updateSize + HASH_BUCKET_SIZE;
        if (tmpTotalSize <= ubSize_) {
            start = mid;
        } else {
            end = mid;
        }
    }

    return start;
}

void ScatterUpdateTiling::DoDeterministicTiling()
{
    if (isDeterministicSplitCol_) {
        // 按列分核，核间数据不重叠，计算结果确定
        isSimt_ = 0;  // 分列不涉及simt
        deterministicMode_ = 1;
        usedCoreNum_ = std::min(Ops::Base::CeilDiv(varShape_[1] * varTypeSize_, DETERMINISTIC_MIN_COL), totalCoreNum_);
        normBlockColNum_ = std::max(DETERMINISTIC_MIN_COL / varTypeSize_, Ops::Base::CeilDiv(varShape_[1], usedCoreNum_));
        usedCoreNum_ = Ops::Base::CeilDiv(varShape_[1], normBlockColNum_);
        tailBlockColNum_ = varShape_[1] - normBlockColNum_ * (usedCoreNum_ - 1);
        if (normBlockColNum_ * varTypeSize_ > (ubSize_ - MIN_INDICES_SIZE)) { // 尾轴大于UBSize
            indicesUbFactor_ = 
                Ops::Base::CeilAlign(std::min(MIN_INDICES_SIZE, indicesSize_ * indicesDtypeSize_), UB_AGLIN_VALUE) / indicesDtypeSize_;
            updateColUbFactor_ = Ops::Base::FloorAlign((ubSize_ - indicesUbFactor_ * indicesDtypeSize_), UB_AGLIN_VALUE) / varTypeSize_;
            updateColUbFactor_ = 
                std::min(updateColUbFactor_, Ops::Base::CeilAlign(normBlockColNum_ * varTypeSize_, UB_AGLIN_VALUE) / varTypeSize_);
        } else {
            updateColUbFactor_ = Ops::Base::CeilAlign(normBlockColNum_ * varTypeSize_, UB_AGLIN_VALUE) / varTypeSize_;
            indicesUbFactor_= Ops::Base::FloorAlign(ubSize_ - updateColUbFactor_ * varTypeSize_, UB_AGLIN_VALUE) / indicesDtypeSize_;
            indicesUbFactor_ = 
                std::min(indicesUbFactor_, Ops::Base::CeilAlign(indicesSize_ * indicesDtypeSize_, UB_AGLIN_VALUE) / indicesDtypeSize_);
        }
        
        indicesLoopSize_ = Ops::Base::CeilDiv(indicesSize_, indicesUbFactor_);
        indicesTailLoopNum_ = indicesSize_ - (indicesLoopSize_ - 1) * indicesUbFactor_;
        updatesNormBlockLoop_ = Ops::Base::CeilDiv(normBlockColNum_, updateColUbFactor_);
        updatesTailBlockLoop_ = Ops::Base::CeilDiv(tailBlockColNum_, updateColUbFactor_);
        updatesNormBlockTailLoopSize_ = normBlockColNum_ - (updatesNormBlockLoop_ - 1) * updateColUbFactor_;
        updatesTailBlockTailLoopSize_ = tailBlockColNum_ - (updatesTailBlockLoop_ - 1) * updateColUbFactor_;
    } else {
        // 按行分核
        deterministicMode_ = DETERMINISTIC_MODE_ROW;
        normBlockIndices_ = Ops::Base::CeilDiv(indicesSize_, totalCoreNum_);
        usedCoreNum_ = Ops::Base::CeilDiv(indicesSize_, normBlockIndices_);
        tailBlockIndices_ = indicesSize_ - (usedCoreNum_ - 1) * normBlockIndices_;

        indicesUbFactor_ = CalcIndicesUbFactor();
        indicesUbFactor_ = std::min(indicesUbFactor_, normBlockIndices_);
        normBlockLoop_ = Ops::Base::CeilDiv(normBlockIndices_, indicesUbFactor_);
        tailBlockLoop_ = Ops::Base::CeilDiv(tailBlockIndices_, indicesUbFactor_);
        normBlockTail_ = normBlockIndices_ - (normBlockLoop_ - 1) * indicesUbFactor_;
        tailBlockTail_ = tailBlockIndices_ - (tailBlockLoop_ - 1) * indicesUbFactor_;

        maskNormBlockLen_ = Ops::Base::FloorDiv(varShape_[0], usedCoreNum_);
        maskTailBlockLen_ = varShape_[0] - (usedCoreNum_ - 1) * maskNormBlockLen_;
        isIndicesSizeInt64_ = indicesSize_ >= INT32_MAX ? true : false;
    }
}

ge::graphStatus ScatterUpdateTiling::DoOpTiling()
{
    uint64_t totalSize = varShape_[1] * indicesSize_;
    if (varShape_[0] * totalSize == 0) {
        usedCoreNum_ = 1;
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }

    if (isDeterministic_) {
        DoDeterministicTiling();
        if (!isSimt_) {
            templateKey_ = UPDATES_IN_SIMD;
        }
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }

    CalcMask();
    if (isMask_ == 1UL) {
        DoMaskSimdTiling();
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }

    isSort_ = indicesSize_ > varShape_[0] ? 1 : 0;
    isSort_ = indicesSize_ >= MIN_SIZE_SORT_INDICES_64 ? isSort_ : 0;

    if (!isSimt_) {
        templateKey_ = UPDATES_IN_SIMD;
        AutoTiling();
        rowNormalNum_ = Ops::Base::FloorDiv(rowTotalNum_, rowTileNum_);
        colNormalNum_ = Ops::Base::FloorDiv(colTotalNum_, colTileNum_);
        rowTailNum_ = rowTotalNum_ - rowNormalNum_ * rowTileNum_ + rowNormalNum_;
        colTailNum_ = colTotalNum_ - colNormalNum_ * colTileNum_ + colNormalNum_;
        CalcKernelParam();
    } else {
        DoSimtTiling();
    }

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterUpdateTiling::GetCastType()
{
    indicesCastDtype_ = indicesDtype_;

    if (indicesDtype_ == ge::DT_INT32) {
 	    if (varShape_[0] < UINT8_MAX) {
 	        indicesCastMode_ = CASTMODE4;          // int32 Cast uint8
 	        indicesCastDtype_ = ge::DT_UINT8;
 	    } else if (varShape_[0] < INT16_MAX) {
 	        indicesCastMode_ = CASTMODE1;          // int32 Cast int16
 	        indicesCastDtype_ = ge::DT_INT16;
 	    }
 	} else {
 	    if (varShape_[0] < UINT8_MAX) {
 	        indicesCastMode_ = CASTMODE5;          // int64 Cast uint8
 	        indicesCastDtype_ = ge::DT_UINT8;
 	    } else if (varShape_[0] < INT16_MAX) {
 	        indicesCastMode_ = CASTMODE3;          // int64 Cast int16
 	        indicesCastDtype_ = ge::DT_INT16;
 	    } else if (varShape_[0] < INT32_MAX) {
 	        indicesCastMode_ = CASTMODE2;          // int64 Cast int32
 	        indicesCastDtype_ = ge::DT_INT32;
 	    }
 	}

    if (indicesCastMode_ != 0) {
        indicesCastDtypeSize_ = ge::GetSizeByDataType(indicesCastDtype_);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterUpdateTiling::CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd, uint64_t updatesSize, uint64_t reserveSize)
{
    uint64_t idsSortBufCnt = 2;
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    uint64_t avaliableSize = ubSize_ - reserveSize;
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / DOUBLE;
        uint64_t sortNeedTmpSize = 0;
        if (indicesCastMode_ == 0) {
            sortNeedTmpSize = GetSortTmpSize(indicesDtype_, baseXoMid, false);
            tmpTotalSize = baseXoMid * updatesSize * DB_BUFFER +                                                         // xQue
                           Ops::Base::CeilAlign(baseXoMid * indicesDtype_, UB_AGLIN_VALUE) * DB_BUFFER +          // idQue
                           Ops::Base::CeilAlign(baseXoMid * indicesDtype_, UB_AGLIN_VALUE) +                          // sortedkeyBuf
                           Ops::Base::CeilAlign((baseXoMid + 1) * sizeof(uint32_t), UB_AGLIN_VALUE) * idsSortBufCnt +      // sortedIdxBuf
                           UB_AGLIN_VALUE + UB_AGLIN_VALUE +                                           // sort padding
                           Ops::Base::CeilAlign(sortNeedTmpSize, UB_AGLIN_VALUE) +
                           HASH_BUCKET_SIZE; // sort shared buf size
        } else {
            sortNeedTmpSize = GetSortTmpSize(indicesCastDtype_, baseXoMid, false);
            tmpTotalSize = baseXoMid * updatesSize * DB_BUFFER +                                                         // xQue
                           Ops::Base::CeilAlign(baseXoMid * indicesCastDtypeSize_, UB_AGLIN_VALUE) +
                           Ops::Base::CeilAlign(baseXoMid * indicesDtype_, UB_AGLIN_VALUE) * DB_BUFFER +          // idQue
                           Ops::Base::CeilAlign(baseXoMid * indicesCastDtypeSize_, UB_AGLIN_VALUE) +                          // sortedkeyBuf
                           Ops::Base::CeilAlign((baseXoMid + 1) * sizeof(uint32_t), UB_AGLIN_VALUE) * idsSortBufCnt +      // sortedIdxBuf
                           UB_AGLIN_VALUE + UB_AGLIN_VALUE +                                           // sort padding
                           Ops::Base::CeilAlign(sortNeedTmpSize, UB_AGLIN_VALUE) +
                           HASH_BUCKET_SIZE; // sort shared buf size
        }
        if (tmpTotalSize <= avaliableSize) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

void ScatterUpdateTiling::ClacColUbParam(uint64_t blockColNum)
{
    colLoopByUb_ = (blockColNum + processColPerUb_ - 1) / processColPerUb_;
    processColTail_ = blockColNum - processColPerUb_ * (colLoopByUb_ - 1);
}

void ScatterUpdateTiling::ClacRowUbParam(uint64_t processRowPerub, uint64_t blockColNum)
{
    processRowPerUb_ = processRowPerub;
    rowLoopByUb_ = (blockColNum + processRowPerUb_ - 1) / processRowPerUb_;
    processRowTail_ = blockColNum - processRowPerUb_ * (rowLoopByUb_ - 1);
}

void ScatterUpdateTiling::ProcessSimdSort()
{
    int32_t normNeedSplitCol = normBlockColNum_ * varTypeSize_ > RESERVE_ROW_SIZE ? 1 : 0;
    int32_t tailNeedSplitCol = tailBlockColNum_ * varTypeSize_ > RESERVE_ROW_SIZE ? 1 : 0;
    uint64_t normColProcessAlign = Ops::Base::CeilAlign(normBlockColNum_ * varTypeSize_, UB_AGLIN_VALUE);
    uint64_t tailColProcessAlign = Ops::Base::CeilAlign(tailBlockColNum_ * varTypeSize_, UB_AGLIN_VALUE);
    GetCastType();
    if (normNeedSplitCol) {
        normNeedSplitRow_ = 1;
        indicesUbFactor_ = CalBestBaseSize(1, normBlockRowNum_, 0, RESERVE_ROW_SIZE * DB_BUFFER);
        processColPerUb_ = RESERVE_ROW_SIZE / varTypeSize_;
        ClacColUbParam(normBlockColNum_);
        updateUbSize_ = RESERVE_ROW_SIZE;
        ClacRowUbParam(indicesUbFactor_, normBlockRowNum_);
    } else {
        processColNum_ = normBlockColNum_;
        indicesUbFactor_ = CalBestBaseSize(1, normBlockRowNum_, normColProcessAlign, 0);
        ClacRowUbParam(indicesUbFactor_, normBlockRowNum_);
        updateUbSize_ = processRowPerUb_ * normColProcessAlign;
    }
    if (tailNeedSplitCol) {
        tailNeedSplitRow_ = 1;
        indicesUbFactor_ = CalBestBaseSize(1, tailBlockRowNum_, 0, RESERVE_ROW_SIZE * DB_BUFFER);
        processColPerUb_ = RESERVE_ROW_SIZE / varTypeSize_;
        ClacColUbParam(tailBlockColNum_);
        updateUbSize_ = RESERVE_ROW_SIZE;
        ClacRowUbParam(indicesUbFactor_, tailBlockRowNum_);
    } else {
        processColNum_ = tailBlockColNum_;
        indicesUbFactor_ = CalBestBaseSize(1, tailBlockRowNum_, tailColProcessAlign, 0);
        ClacRowUbParam(indicesUbFactor_, tailBlockRowNum_);
        updateUbSize_ = processRowPerUb_ * tailColProcessAlign;
    }
}

void ScatterUpdateTiling::ProcessSimdNonSort(uint64_t existNodeSize)
{
    existNodeSize = existNodeSize / DOUBLE;
    int32_t normNeedSplitCol = normBlockColNum_ * varTypeSize_ + UB_AGLIN_VALUE > existNodeSize ? 1 : 0;
    int32_t tailNeedSplitCol = tailBlockColNum_ * varTypeSize_ + UB_AGLIN_VALUE > existNodeSize ? 1 : 0;
    uint64_t normColProcessAlign = Ops::Base::CeilAlign(normBlockColNum_ * varTypeSize_, UB_AGLIN_VALUE);
    uint64_t tailColProcessAlign = Ops::Base::CeilAlign(tailBlockColNum_ * varTypeSize_, UB_AGLIN_VALUE);
    if (normNeedSplitCol) {
        normNeedSplitRow_ = 1;
        processColPerUb_ = ((existNodeSize - SPLIT_ROW_INDICES_NUM * indicesDtypeSize_) / UB_AGLIN_VALUE * UB_AGLIN_VALUE) / varTypeSize_;
        ClacColUbParam(normBlockColNum_);
        indicesUbFactor_ = SPLIT_ROW_INDICES_NUM;
        updateUbSize_ = processColPerUb_ * varTypeSize_;
        ClacRowUbParam(SPLIT_ROW_INDICES_NUM, normBlockRowNum_);
    } else {
        processColNum_ = normBlockColNum_;
        ClacRowUbParam((existNodeSize - UB_AGLIN_VALUE) / (normColProcessAlign + indicesDtypeSize_), normBlockRowNum_);
        indicesUbFactor_ = (processRowPerUb_ * indicesDtypeSize_ + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE * UB_AGLIN_VALUE / indicesDtypeSize_;
        updateUbSize_ = processRowPerUb_ * normColProcessAlign;
    }
    if (tailNeedSplitCol) {
        tailNeedSplitRow_ = 1;
        processColPerUb_ = ((existNodeSize - SPLIT_ROW_INDICES_NUM * indicesDtypeSize_) / UB_AGLIN_VALUE * UB_AGLIN_VALUE) / varTypeSize_;
        ClacColUbParam(tailBlockColNum_);
        indicesUbFactor_ = SPLIT_ROW_INDICES_NUM;
        updateUbSize_ = processColPerUb_ * varTypeSize_;
        ClacRowUbParam(SPLIT_ROW_INDICES_NUM, tailBlockRowNum_);
    } else {
        processColNum_ = tailBlockColNum_;
        ClacRowUbParam((existNodeSize - UB_AGLIN_VALUE) / (tailColProcessAlign + indicesDtypeSize_), tailBlockRowNum_);
        indicesUbFactor_ = (processRowPerUb_ * indicesDtypeSize_ + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE * UB_AGLIN_VALUE / indicesDtypeSize_;
        updateUbSize_ = processRowPerUb_ * tailColProcessAlign;
    }
}

void ScatterUpdateTiling::CalcKernelParam()
{
    OP_LOGD(opName, "ScatterUpdateTiling CalcKernelParam Enter.");
    normBlockColNum_ = colNormalNum_;
    normBlockRowNum_ = rowNormalNum_;
    tailBlockColNum_ = colTailNum_;
    tailBlockRowNum_ = rowTailNum_;
    uint64_t existNodeSize = ubSize_;
    isSort_ = normBlockRowNum_ >= SIMD_MIN_SIZE_SORT_INDICES ? isSort_ : 0;
    if (isSort_) {
        ProcessSimdSort();
    } else {
        ProcessSimdNonSort(existNodeSize);
    }
}

void ScatterUpdateTiling::AutoTiling()
{
    OP_LOGD(opName, "ScatterUpdateTiling AutoTiling Enter. ");
    rowTotalNum_ = updateShape_[0];
    colTotalNum_ = updateShape_[1];
    uint64_t base = BASE_BLOCK_COPY_ALIGN / varTypeSize_;
    uint64_t colNumAlign = (colTotalNum_ + base - 1) / base;
    /*
    **给定一个Shape[M,N]和block core num，找到尽可能均匀且尽量用满核的切分方式
    */
    usedCoreNum_ = std::min(totalCoreNum_, static_cast<uint64_t>((colTotalNum_ * rowTotalNum_ + BASE_BLOCK_SIZE - 1) / BASE_BLOCK_SIZE));
    /* 给定Shape[M, N] 和 block core num
    ** M切分成m块，N切分成n块，找到m*n <= usedCoreNum, 且m*n尽可能接近usedCoreNum的所有m和n的可能
    */ 
    std::set<uint64_t> cutSet = FindUniqueCut();
    std::vector<std::vector<uint64_t>> allTiling;
    // 核先按照行方向切分，枚举m的取值
    for (uint64_t m : cutSet) {
        // 行方向分核超过行方向数据量，则说明有空闲核
        if (m > colNumAlign) {
            continue;
        }
        uint64_t n = usedCoreNum_ / m;
        n = n < 1 ? 1 : n;
        // 列方向分核超过列方向数据量，则说明有空闲核
        if (n > rowTotalNum_) {
            continue;
        }

        uint64_t colNormalBlock = Ops::Base::CeilDiv(colNumAlign, m);
        uint64_t rowNormalBlock = Ops::Base::CeilDiv(rowTotalNum_, n);
        // 正常切分块和尾块的差值计算
        uint64_t delta = rowNormalBlock * colNormalBlock;
        if (m * n == usedCoreNum_) {
            if (colNumAlign % m == 0 && rowTotalNum_ % n == 0) {
                colTileNum_ = m;
                rowTileNum_ = n;
                return;
            } else if (colNumAlign % m == 0) {
                delta = delta - colNormalBlock * (rowTotalNum_ % rowNormalBlock);
            } else if (rowTotalNum_ % n == 0) {
                delta = delta - (colNumAlign % colNormalBlock) * rowNormalBlock;
            } else {
                delta = delta - (colNumAlign % colNormalBlock) * (rowTotalNum_ % rowNormalBlock);
            }
        }
        allTiling.push_back({m, n, m * n, delta});
    }
    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        constexpr int NIndex = 1;
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[DeltaIndex], a[NIndex]) < std::make_pair(b[DeltaIndex], b[NIndex]);
    });
    colTileNum_ = allTiling[0][0];
    rowTileNum_ = allTiling[0][1];
    usedCoreNum_ = colTileNum_ * rowTileNum_;
}

std::set<uint64_t> ScatterUpdateTiling::FindUniqueCut()
{
    std::set<uint64_t> result;
    uint64_t upbound = std::ceil(std::sqrt(usedCoreNum_) + 1);
    
    for (uint64_t m = 1; m < upbound; m++) {
        uint64_t y = usedCoreNum_ / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

ge::graphStatus ScatterUpdateTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + TILING_KEY_PARAM_INTERVAL * RecursiveSum(templateIds...);
}

template <typename... Args> constexpr uint64_t GET_TILINGKEY(Args... templateIds)
{
    return TILINGKEYOFFSET + RecursiveSum(templateIds...);
}

uint64_t ScatterUpdateTiling::GetTilingKey() const
{
    uint64_t totalSize = varShape_[1] * indicesSize_;
    if (varShape_[0] * totalSize == 0) {
        return 0;
    }
    uint32_t sizeAddrType = ((varShape_[1] * indicesSize_  > INT32_MAX) || (varShape_[0] * varStride_ > INT32_MAX)) ? 1 : 0;
    if (!isSimt_ || isDeterministic_) {
        sizeAddrType = 0;
    }

    uint64_t tilingKey = 0;
    tilingKey = GET_TILINGKEY(isSort_, templateKey_, sizeAddrType, isUpdateScalar_,TILING_KEY_PLACE_HOLD, TILING_KEY_PLACE_HOLD,
                    TILING_KEY_PLACE_HOLD, TILING_KEY_PLACE_HOLD, TILING_KEY_PLACE_HOLD, isMask_, deterministicMode_);
    OP_LOGD("ScatterUpdateTiling", "tilingKey is %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus ScatterUpdateTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    if (isDeterministic_ && !isDeterministicSplitCol_) {
        workspaceSize_ += varShape_[0] * sizeof(int64_t);  // mask类型与indices一致
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterUpdateTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(usedCoreNum_);
    if (isDeterministic_) {
        context_->SetScheduleMode(1);
    }
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_TILING_CHECK((res != ge::GRAPH_SUCCESS),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "SetLocalMemorySize ubSize = %lu failed.", ubSize_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void ScatterUpdateTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "varShape: " << varShape_[0] << "," << varShape_[1] << std::endl;
    info << "indicesSize: " << indicesSize_ << std::endl;
    info << "normBlockIndices: " << normBlockIndices_ << std::endl;
    info << "tailBlockIndices_: " << tailBlockIndices_ << std::endl;
    info << "indicesFactor: " << indicesFactor_ <<std::endl;
    info << "normBlockLoop: " << normBlockLoop_ << std::endl;
    info << "tailBlockLoop: " << tailBlockLoop_ << std::endl;
    info << "normBlockTail: " << normBlockTail_ << std::endl;
    info << "tailBlockTail: " << tailBlockTail_ << std::endl;
    info << "rowTotalNum: " << rowTotalNum_ << std::endl;
    info << "colTotalNum: " << colTotalNum_ << std::endl;
    info << "rowNormalNum: " << rowNormalNum_ << std::endl;
    info << "colNormalNum: " << colNormalNum_ << std::endl;
    info << "rowTailNum: " << rowTailNum_ << std::endl;
    info << "colTailNum: " << colTailNum_ << std::endl;
    info << "rowTileNum: " << rowTileNum_ << std::endl;
    info << "colTileNum: " << colTileNum_ << std::endl;
    info << "usedCoreNum: " << usedCoreNum_ << std::endl;
    info << "normBlockColNum: " << normBlockColNum_ << std::endl;
    info << "normBlockRowNum: " << normBlockRowNum_ <<std::endl;
    info << "tailBlockColNum: " << tailBlockColNum_ << std::endl;
    info << "tailBlockRowNum: " << tailBlockRowNum_ << std::endl;
    info << "normNeedSplitRow: " << normNeedSplitRow_ << std::endl;
    info << "tailNeedSplitRow: " << tailNeedSplitRow_ << std::endl;
    info << "processRowPerUb: " << processRowPerUb_ << std::endl;
    info << "processColNum: " << processColNum_ << std::endl;
    info << "rowLoopByUb: " << rowLoopByUb_ << std::endl;
    info << "processRowTail: " << processRowTail_ << std::endl;
    info << "indicesUbFactor: " << indicesUbFactor_ << std::endl;
    info << "updateUbSize: " << updateUbSize_ << std::endl;
    info << "processColPerUb: " << processColPerUb_ << std::endl;
    info << "colLoopByUb: " << colLoopByUb_ << std::endl;
    info << "processColTail: " << processColTail_ << std::endl;
    info << "indicesBatchCopySizeAlign: " << indicesBatchCopySizeAlign_ << std::endl;
    info << "varStride: " << varStride_ << std::endl;
    info << "updateColUbFactor: " << updateColUbFactor_ << std::endl;
    info << "indicesLoopSize: " << indicesLoopSize_ << std::endl;
    info << "indicesTailLoopNum: " << indicesTailLoopNum_ << std::endl;
    info << "updatesNormBlockLoop: " << updatesNormBlockLoop_ << std::endl;
    info << "updatesTailBlockLoop: " << updatesTailBlockLoop_ << std::endl;
    info << "updatesNormBlockTailLoopSize: " << updatesNormBlockTailLoopSize_ << std::endl;
    info << "updatesTailBlockTailLoopSize: " << updatesTailBlockTailLoopSize_ << std::endl;
    info << "maskNormBlockLen: " << maskNormBlockLen_ << std::endl;
    info << "maskTailBlockLen: " << maskTailBlockLen_ << std::endl;
    info << "isIndicesSizeInt64: " << isIndicesSizeInt64_ << std::endl;
    info << "indicesCastMode: " << indicesCastMode_ << std::endl;

    OP_LOGI(opName, "Tiling inf is: %s", info.str().c_str());
}
}  // namespace optiling