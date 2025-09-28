/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_top_k_top_p_with_sorted_tiling.cpp
 * \brief
 */

#include <iostream>
#include <map>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "apply_top_k_top_p_with_sorted_tiling.h"

namespace {
    constexpr uint32_t SYS_RESERVED_UB = uint32_t(16 * 1024);
    constexpr uint32_t SELECT_RESERVED_UB = uint32_t(8 * 1024);
    constexpr uint32_t DIM_ONE = 1;
    constexpr uint32_t DIM_TWO = 2;
    constexpr int32_t SORTED_VALUE_INPUT_INDEX = 0;
    constexpr int32_t SORTED_INDICES_INPUT_INDEX = 1;
    constexpr int32_t P_INPUT_INDEX = 2;
    constexpr int32_t K_INPUT_INDEX = 3;
    constexpr uint32_t DIM_INDEX0 = 0;
    static std::map<ge::DataType, uint32_t> DTYPE_MAP = {{ge::DT_BF16, 2}, {ge::DT_FLOAT16, 1}, {ge::DT_FLOAT, 0}};
    static std::map<ge::DataType, uint32_t> DATATYPE_LEN_MAP = {
        {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}, {ge::DT_FLOAT, 4}};
    const static uint32_t SYS_WORKSPACESIZE = uint32_t(16 * 1024 * 1024);

    constexpr uint32_t DATA_PER_BLOCK_B32 = 8;
    constexpr uint32_t BYTES_B32 = 4;
    constexpr uint32_t BLOCK_BYTES = 32;
    constexpr uint32_t K_VALUE_MAX = 1024;
    constexpr uint32_t ONLY_TOP_P_KEY = 2;
    constexpr uint32_t ONLY_TOP_K_KEY = 1;
} // namespace

namespace optiling {
class ApplyTopKTopPWithSortedTiling {
public:
    explicit ApplyTopKTopPWithSortedTiling(gert::TilingContext* context) : tilingcontext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
private:
    ApplyTopKTopPWithSortedTilingData tilingData;
    gert::TilingContext* tilingcontext = nullptr;
    ge::graphStatus CheckShape();
    void SetTilingKey();
    void GetUsedCore();
    void CalDataPerCore();
    void FillTilingData();
    void PrintTilingData();
    template <typename T1>
    inline auto CeilAlign(T1 a, T1 b) const -> T1
    {
        return b == 0 ? a : (a + b - 1) / b * b;
    }
    template <typename T1>
    inline auto FloorAlign(T1 a, T1 b) const -> T1
    {
        return b == 0 ? a : a / b * b;
    }

    const char *opName_ = nullptr;
    uint32_t coreNum_ = 0;
    uint32_t calUbSize_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t tilingKey_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t batchPerCore_ = 1;
    uint32_t tailBatch_ = 0;
    uint32_t dataNumInit_ = 0;
    uint32_t dataNumInitAligned_ = 0;
    uint32_t ubFactorElement_ = 0;
    uint32_t ubFactorElementAligned_ = 0;
    uint32_t tailUbFactorElement_ = 0;
    uint32_t tailUbFactorElementAligned_ = 0;
    uint32_t onlyTopK = 0;
    uint32_t onlyTopP = 0;
};

ge::graphStatus ApplyTopKTopPWithSortedTiling::CheckShape() {
    auto sortedValueShapePtr = tilingcontext->GetInputShape(SORTED_VALUE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingcontext, sortedValueShapePtr);
    auto sortedValueShape = sortedValueShapePtr->GetStorageShape();
    if (sortedValueShape.GetDimNum() != DIM_TWO) {
        OP_LOGE(opName_, "the dimNum of sorted_value should be 2, but got %ld.", sortedValueShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto sortedIndicesShapePtr = tilingcontext->GetInputShape(SORTED_INDICES_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingcontext, sortedIndicesShapePtr);
    auto sortedIndicesShape = sortedIndicesShapePtr->GetStorageShape();
    if (sortedIndicesShape.GetDimNum() != DIM_TWO) {
        OP_LOGE(opName_, "the dimNum of sorted_indices should be 2, but got %ld.", sortedIndicesShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    batchSize_ = sortedValueShape.GetDim(DIM_INDEX0);
    vocabSize_ = sortedValueShape.GetDim(DIM_ONE);
    if (sortedIndicesShape.GetDim(DIM_INDEX0) != batchSize_ || sortedIndicesShape.GetDim(DIM_ONE) != vocabSize_) {
        OP_LOGE(opName_, "the shape of sorted_indices should be equal to sorted_value.");
        return ge::GRAPH_FAILED;
    }

    auto pShapePtr = tilingcontext->GetOptionalInputShape(P_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingcontext, pShapePtr);
    auto pShape = pShapePtr->GetStorageShape();
    auto pDimNum = pShape.GetDimNum();
    if (pDimNum != DIM_ONE && pDimNum != 0) {
        OP_LOGE(opName_, "the dimNum of p should be 1 or 0, but got %ld.", pDimNum);
        return ge::GRAPH_FAILED;
    }
    if (pDimNum != 0 && batchSize_ != pShape.GetDim(DIM_INDEX0)) {
        OP_LOGE(opName_, "p.shape[0] should be equal to logits.shape[0].");
        return ge::GRAPH_FAILED;
    }

    auto kShapePtr = tilingcontext->GetOptionalInputShape(K_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingcontext, kShapePtr);
    auto kShape = kShapePtr->GetStorageShape();
    auto kDimNum = kShape.GetDimNum();
    if (kDimNum != DIM_ONE && kDimNum != 0) {
        OP_LOGE(opName_, "the dimNum of k should be 1 or 0, but got %ld.", kShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (kDimNum != 0 && batchSize_ != kShape.GetDim(DIM_INDEX0)) {
        OP_LOGE(opName_, "k.shape[0] should be equal to logits.shape[0].");
        return ge::GRAPH_FAILED;
    }
    if (kDimNum == 0 && pDimNum == 0) {
        OP_LOGE(opName_, "the dimNum of q and k should be 0 at the same time.");
        return ge::GRAPH_FAILED;
    }
    onlyTopK = (kDimNum != 0 && pDimNum == 0) ? ONLY_TOP_K_KEY : 0;
    onlyTopP = (pDimNum != 0 && kDimNum == 0) ? ONLY_TOP_P_KEY : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyTopKTopPWithSortedTiling::Init() {
    opName_ = tilingcontext->GetNodeName();
    OP_LOGD(opName_, "TilingForApplyTopKTopPWithSorted init.");
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingcontext->GetPlatformInfo());
    coreNum_ = platformInfo.GetCoreNumAiv();
    uint64_t platformUbSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformUbSize);
    OP_LOGD(opName_, "platformUbSize: %lu.", platformUbSize);
    uint32_t avaliableUb = static_cast<uint32_t>(platformUbSize) - SYS_RESERVED_UB - SELECT_RESERVED_UB;
    calUbSize_ = FloorAlign(avaliableUb, BLOCK_BYTES);
    if (CheckShape() == ge::GRAPH_FAILED) {
        OP_LOGE(opName_, "check shape failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void ApplyTopKTopPWithSortedTiling::SetTilingKey() {
    tilingKey_ += onlyTopK;
    tilingKey_ += onlyTopP;
    tilingcontext->SetTilingKey(tilingKey_);
}

void ApplyTopKTopPWithSortedTiling::GetUsedCore()
{
    if (batchSize_ <= coreNum_) {
        batchPerCore_ = uint32_t(1);
        usedCoreNum_ = batchSize_;
        tailBatch_ = uint32_t(0);
        return;
    }
    batchPerCore_ = coreNum_ == uint32_t(0) ? batchSize_ : batchSize_ / coreNum_;
    tailBatch_ = batchSize_ % coreNum_;
    usedCoreNum_ = coreNum_;
}

void ApplyTopKTopPWithSortedTiling::CalDataPerCore()
{
    uint32_t inputDataTypeByte = DATATYPE_LEN_MAP[tilingcontext->GetInputDesc(SORTED_VALUE_INPUT_INDEX)->GetDataType()];
    uint32_t dataPerBlock = BLOCK_BYTES / inputDataTypeByte;
    dataNumInit_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    dataNumInitAligned_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    ubFactorElement_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    ubFactorElementAligned_ = CeilAlign(ubFactorElement_, dataPerBlock);
    tailUbFactorElement_ = vocabSize_ % ubFactorElement_;
    tailUbFactorElement_ = tailUbFactorElement_ == uint32_t(0) ? ubFactorElement_ : tailUbFactorElement_;
    tailUbFactorElementAligned_ = CeilAlign(tailUbFactorElement_, dataPerBlock);

    uint32_t sortedValueBytes = ubFactorElementAligned_ * inputDataTypeByte + K_VALUE_MAX  * inputDataTypeByte;
    uint32_t sortedIndicesBytes = ubFactorElementAligned_ * BYTES_B32 + K_VALUE_MAX  * BYTES_B32;
    uint32_t pBytes = dataPerBlock * inputDataTypeByte;
    uint32_t kBytes = DATA_PER_BLOCK_B32 * BYTES_B32;
    uint32_t outTensorBytes = ubFactorElementAligned_ * inputDataTypeByte;

    calUbSize_ = calUbSize_ - sortedValueBytes - sortedIndicesBytes - pBytes - kBytes - outTensorBytes;
}

void ApplyTopKTopPWithSortedTiling::FillTilingData()
{
    tilingData.set_batchSize(batchSize_);
    tilingData.set_vocabSize(vocabSize_);
    tilingData.set_batchPerCore(batchPerCore_);
    tilingData.set_tailBatch(tailBatch_);
    tilingData.set_blockNum(usedCoreNum_);
    tilingData.set_dataNumInit(dataNumInit_);
    tilingData.set_dataNumInitAligned(dataNumInitAligned_);
    tilingData.set_ubFactorElement(ubFactorElement_);
    tilingData.set_ubFactorElementAligned(ubFactorElementAligned_);
    tilingData.set_tailUbFactorElement(tailUbFactorElement_);
    tilingData.set_tailUbFactorElementAligned(tailUbFactorElementAligned_);
    tilingData.set_calUbSize(calUbSize_);
}

void ApplyTopKTopPWithSortedTiling::PrintTilingData()
{
    OP_LOGD(opName_, "batchSize: %u.", tilingData.get_batchSize());
    OP_LOGD(opName_, "vocabSize: %u.", tilingData.get_vocabSize());
    OP_LOGD(opName_, "batchPerCore: %u.", tilingData.get_batchPerCore());
    OP_LOGD(opName_, "tailBatch: %u.", tilingData.get_tailBatch());
    OP_LOGD(opName_, "usedCoreNum: %u.", tilingData.get_blockNum());
    OP_LOGD(opName_, "dataNumInit_: %u.", tilingData.get_dataNumInit());
    OP_LOGD(opName_, "dataNumInitAligned_: %u.", tilingData.get_dataNumInitAligned());
    OP_LOGD(opName_, "ubFactorElement: %u.", tilingData.get_ubFactorElement());
    OP_LOGD(opName_, "ubFactorElementAligned: %u.", tilingData.get_ubFactorElementAligned());
    OP_LOGD(opName_, "tailUbFactorElement: %u.", tilingData.get_tailUbFactorElement());
    OP_LOGD(opName_, "tailUbFactorElementAligned: %u.", tilingData.get_tailUbFactorElementAligned());
    OP_LOGD(opName_, "calUbSize: %u.", tilingData.get_calUbSize());
}

ge::graphStatus ApplyTopKTopPWithSortedTiling::RunKernelTiling()
{
    OP_LOGD(opName_, "TilingForApplyTopKTopPWithSorted start.");

    SetTilingKey();
    GetUsedCore();
    CalDataPerCore();
    FillTilingData();
    PrintTilingData();

    OP_LOGD(opName_, "tilingKey: %u.", tilingKey_);
    uint32_t syncWorkspaceSize = SYS_WORKSPACESIZE;
    size_t* currentWorkspace = tilingcontext->GetWorkspaceSizes(1);
    currentWorkspace[0] = syncWorkspaceSize;

    tilingData.SaveToBuffer(tilingcontext->GetRawTilingData()->GetData(),
                            tilingcontext->GetRawTilingData()->GetCapacity());
    tilingcontext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingcontext->SetBlockDim(usedCoreNum_);

    OP_LOGD(opName_, "TilingForApplyTopKTopPWithSorted end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForApplyTopKTopPWithSorted(gert::TilingContext* context)
{
    ApplyTopKTopPWithSortedTiling tilingObject(context);
    auto ret = tilingObject.Init();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling Init failed.");
        return ge::GRAPH_FAILED;
    }
    ret = tilingObject.RunKernelTiling();
    OP_LOGD(context->GetNodeName(), "TilingForApplyTopKTopPWithSorted end.");
    return ret;
}

static ge::graphStatus TilingPrepareForApplyTopKTopPWithSorted(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForApplyTopKTopPWithSorted start");
    auto compileInfo = context->GetCompiledInfo<TilingForApplyTopKTopPWithSortedCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(compileInfo->ubSizePlatForm <= 0,
                OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
    OP_LOGD(context->GetNodeName(), "TilingPrepareForApplyTopKTopPWithSorted end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyTopKTopPWithSorted)
    .Tiling(TilingForApplyTopKTopPWithSorted)
    .TilingParse<TilingForApplyTopKTopPWithSortedCompileInfo>(TilingPrepareForApplyTopKTopPWithSorted);
} // namespace optiling