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
 * \file inplace_index_add_with_sorted_tiling_arch35.cpp
 * \brief A5 (ascend950) tiling logic — bufCnt=5, tilingKey 10000
 */

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "inplace_index_add_with_sorted_tiling_arch35.h"
#include "platform/platform_infos_def.h"

using namespace std;

namespace {
const int32_t SORTED_TILING_KEY = 10000;

const int32_t SIZE_OF_FP16 = 2;
const int32_t SIZE_OF_BF16 = 2;
const int32_t SIZE_OF_FP32 = 4;
const int32_t SIZE_OF_INT32 = 4;
const int32_t MAX_DIM_NUM = 8;

const int32_t INPUT_0 = 0;
const int32_t INPUT_1 = 1;
const int32_t INPUT_2 = 2;
const int32_t INPUT_3 = 3;
const int32_t INPUT_4 = 4;
const int32_t BUF_CNT = 5; // double buffer : (var + update, b32) + (out, b16)
const int32_t BLOCK_SIZE = 32;
const int64_t UB_INDEX_NUM = 1536;
const int64_t INDEX_BUFFER_SIZE = UB_INDEX_NUM * 2 * SIZE_OF_INT32;
const int64_t RESERVED_BUFFER_SIZE = 1024;
} // namespace

namespace optiling {
class InplaceIndexAddWithSortedTiling
{
public:
    explicit InplaceIndexAddWithSortedTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    void TilingDataSet();
    void TilingDataPrint() const;
    void processFirstDimTilingData();
    bool CheckParam();
    InplaceIndexAddWithSortedTilingData* tilingData_{nullptr};
    gert::TilingContext* tilingContext = nullptr;
    uint32_t tilingKey = 0;
    int64_t dimAttr = -1;
    int64_t ubSize = 1;
    int64_t inputCount = 1;
    int64_t updatesCount = 1;
    int64_t indicesCount = 1;
    int64_t updatesOneTime = 1;
    int64_t inputSize = 1;
    int32_t coreNum = 1;

    int32_t usedCoreNum = 1;
    int32_t enableAlpha = 0;
    int64_t eachIndexCount = 1;
    int64_t lastIndexCount = 1;
    int64_t maxSize = 0;
    int64_t eachNum = 1;
    int64_t eachLoop = 1;
    int64_t eachTail = 0;
    int64_t batchNum = 1;
    int64_t eachBatchNum = 1;
    int64_t lastBatchNum = 1;
    int64_t varDimNum = 1;
    int64_t eachUBIndexRound = 1;
    int64_t eachUBIndexCount = 1;
    int64_t eachUBIndexTail = 0;
    int64_t lastUBIndexRound = 1;
    int64_t lastUBIndexCount = 1;
    int64_t lastUBIndexTail = 0;
    uint64_t workspaceSize = 1024 * 1024 * 16;
};

bool InplaceIndexAddWithSortedTiling::CheckParam()
{
    if (tilingContext->GetInputShape(INPUT_0) == nullptr || tilingContext->GetInputShape(INPUT_1) == nullptr ||
        tilingContext->GetInputShape(INPUT_2) == nullptr || tilingContext->GetInputShape(INPUT_3) == nullptr ||
        tilingContext->GetInputDesc(INPUT_0) == nullptr || tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE(tilingContext, "tilingContext inputShape or outputShape is nullptr.");
        return false;
    }
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    auto valueDtype = tilingContext->GetInputDesc(INPUT_1)->GetDataType();
    inputSize = ge::GetSizeByDataType(inputDtype);

    if (inputDtype != valueDtype) {
        OP_LOGE(tilingContext, "value dtype must be same as var.");
        return false;
    }

    if (ge::DT_INT32 != tilingContext->GetInputDesc(INPUT_2)->GetDataType() ||
        ge::DT_INT32 != tilingContext->GetInputDesc(INPUT_3)->GetDataType()) {
        OP_LOGE(tilingContext, "sorted_index and pos only support int32.");
        return false;
    }
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto alphaShape = tilingContext->GetOptionalInputShape(INPUT_4);
    enableAlpha = (alphaShape == nullptr) ? 0 : 1;

    auto inputDimNum = inputShape.GetDimNum();
    if (inputDimNum > MAX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "dim",
            std::to_string(inputDimNum).c_str(), "<= 8");
        return false;
    }
    if (inputDimNum != updatesShape.GetDimNum()) {
        OP_LOGE(tilingContext, "the dimNum of input must equal the dimNum of updates.");
        return false;
    }
    const int64_t* ptrDim = tilingContext->GetAttrs()->GetAttrPointer<int64_t>(0);
    dimAttr = *ptrDim;
    dimAttr = dimAttr < 0 ? inputDimNum + dimAttr : dimAttr;
    if (dimAttr != 0) {
        OP_LOGE(tilingContext, "Dim only support 0 on the version, but get %ld.", dimAttr);
        return false;
    }
    for (int64_t idx = 0; idx < static_cast<int64_t>(inputDimNum); ++idx) {
        if (dimAttr != idx) {
            if (inputShape.GetDim(idx) != updatesShape.GetDim(idx)) {
                OP_LOGE(tilingContext, "The size of self must match the size of source at dimension %ld.", idx);
                return false;
            }
        }
    }
    return true;
}

ge::graphStatus InplaceIndexAddWithSortedTiling::Init()
{
    if (tilingContext == nullptr) {
        OP_LOGE(tilingContext, "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = static_cast<const InplaceIndexAddWithSortedCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    coreNum = static_cast<int32_t>(compileInfo->totalCoreNum);
    if (coreNum == 0) {
        OP_LOGE(tilingContext, "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    ubSize = static_cast<int64_t>(compileInfo->ubSizePlatForm) - INDEX_BUFFER_SIZE - RESERVED_BUFFER_SIZE;
    workspaceSize = compileInfo->workspaceSize;

    if (!CheckParam()) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplaceIndexAddWithSortedTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    auto inputDimNum = inputShape.GetDimNum();
    for (int64_t i = 0; i < static_cast<int64_t>(inputDimNum); ++i) {
        auto dimInput = inputShape.GetDim(i);
        auto dimUpdates = updatesShape.GetDim(i);
        if (i < dimAttr) {
            batchNum *= dimUpdates;
        }
        if (i == dimAttr) {
            varDimNum = dimInput;
        }
        if (i >= dimAttr + 1) {
            updatesOneTime *= dimUpdates;
        }
        inputCount *= dimInput;
        updatesCount *= dimUpdates;
    }
    indicesCount = indicesShape.GetDim(INPUT_0);
    if (inputCount == 0 || updatesCount == 0 || indicesCount == 0) {
        OP_LOGE(tilingContext, "shape size cannot equal 0.");
        return ge::GRAPH_FAILED;
    }
    tilingKey = SORTED_TILING_KEY;
    processFirstDimTilingData();
    TilingDataSet();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void InplaceIndexAddWithSortedTiling::processFirstDimTilingData()
{
    maxSize = (ubSize / BUF_CNT) / BLOCK_SIZE * BLOCK_SIZE;
    maxSize /= SIZE_OF_FP32; // 转为element
    maxSize = (maxSize / 16) * 16; // 元素对齐

    usedCoreNum = indicesCount < coreNum ? indicesCount : coreNum;
    eachIndexCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (indicesCount + eachIndexCount - 1) / eachIndexCount;
    lastIndexCount = indicesCount - eachIndexCount * (usedCoreNum - 1);
    eachNum = updatesOneTime;
    eachTail = eachNum;
    if (eachNum > maxSize) {
        eachLoop = (eachNum + maxSize - 1) / maxSize;
        eachNum = maxSize;
        eachTail = updatesOneTime - (eachLoop - 1) * eachNum;
    }
    if (eachIndexCount > UB_INDEX_NUM) {
        eachUBIndexRound = (eachIndexCount + UB_INDEX_NUM - 1) / UB_INDEX_NUM;
        eachUBIndexCount = UB_INDEX_NUM;
        eachUBIndexTail = eachIndexCount - (eachUBIndexRound - 1) * UB_INDEX_NUM;
    } else {
        eachUBIndexRound = 1;
        eachUBIndexCount = eachIndexCount;
        eachUBIndexTail = eachIndexCount;
    }
    if (lastIndexCount > UB_INDEX_NUM) {
        lastUBIndexRound = (lastIndexCount + UB_INDEX_NUM - 1) / UB_INDEX_NUM;
        lastUBIndexCount = UB_INDEX_NUM;
        lastUBIndexTail = lastIndexCount - (lastUBIndexRound - 1) * UB_INDEX_NUM;
    } else {
        lastUBIndexRound = 1;
        lastUBIndexCount = lastIndexCount;
        lastUBIndexTail = lastIndexCount;
    }
}

void InplaceIndexAddWithSortedTiling::TilingDataSet()
{
    tilingData_ = tilingContext->GetTilingData<InplaceIndexAddWithSortedTilingData>();
    tilingData_->usedCoreNum = usedCoreNum;
    tilingData_->enableAlpha = enableAlpha;
    tilingData_->eachIndexCount = eachIndexCount;
    tilingData_->lastIndexCount = lastIndexCount;
    tilingData_->batchCount = batchNum;
    tilingData_->eachBatchNum = eachBatchNum;
    tilingData_->lastBatchNum = lastBatchNum;
    tilingData_->inputCount = inputCount;
    tilingData_->indicesCount = indicesCount;
    tilingData_->updatesCount = updatesCount;
    tilingData_->updatesOneTime = updatesOneTime;
    tilingData_->maxSize = maxSize;
    tilingData_->eachNum = eachNum;
    tilingData_->eachLoop = eachLoop;
    tilingData_->eachTail = eachTail;
    tilingData_->varDimNum = varDimNum;
    tilingData_->eachUBIndexRound = eachUBIndexRound;
    tilingData_->eachUBIndexCount = eachUBIndexCount;
    tilingData_->eachUBIndexTail = eachUBIndexTail;
    tilingData_->lastUBIndexRound = lastUBIndexRound;
    tilingData_->lastUBIndexCount = lastUBIndexCount;
    tilingData_->lastUBIndexTail = lastUBIndexTail;

    TilingDataPrint();

    tilingContext->SetBlockDim(usedCoreNum);
    tilingContext->SetTilingKey(static_cast<uint64_t>(tilingKey));
    
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize;
}

void InplaceIndexAddWithSortedTiling::TilingDataPrint() const
{
    OP_LOGI(tilingContext, "usedCoreNum: %u.", tilingData_->usedCoreNum);
    OP_LOGI(tilingContext, "enableAlpha: %u.", tilingData_->enableAlpha);
    OP_LOGI(tilingContext, "eachIndexCount: %ld.", tilingData_->eachIndexCount);
    OP_LOGI(tilingContext, "lastIndexCount: %ld.", tilingData_->lastIndexCount);
    OP_LOGI(tilingContext, "batchNum: %ld.", tilingData_->batchCount);
    OP_LOGI(tilingContext, "eachBatchNum: %ld.", tilingData_->eachBatchNum);
    OP_LOGI(tilingContext, "lastBatchNum: %ld.", tilingData_->lastBatchNum);
    OP_LOGI(tilingContext, "inputCount: %ld.", tilingData_->inputCount);
    OP_LOGI(tilingContext, "indicesCount: %ld.", tilingData_->indicesCount);
    OP_LOGI(tilingContext, "updatesCount: %ld.", tilingData_->updatesCount);
    OP_LOGI(tilingContext, "updatesOneTime: %ld.", tilingData_->updatesOneTime);
    OP_LOGI(tilingContext, "maxSize: %ld.", tilingData_->maxSize);
    OP_LOGI(tilingContext, "eachNum: %ld.", tilingData_->eachNum);
    OP_LOGI(tilingContext, "eachLoop: %ld.", tilingData_->eachLoop);
    OP_LOGI(tilingContext, "eachTail: %ld.", tilingData_->eachTail);
    OP_LOGI(tilingContext, "varDimNum: %ld.", tilingData_->varDimNum);
    OP_LOGI(tilingContext, "eachUBIndexRound: %ld.", tilingData_->eachUBIndexRound);
    OP_LOGI(tilingContext, "eachUBIndexCount: %ld.", tilingData_->eachUBIndexCount);
    OP_LOGI(tilingContext, "eachUBIndexTail: %ld.", tilingData_->eachUBIndexTail);
    OP_LOGI(tilingContext, "lastUBIndexRound: %ld.", tilingData_->lastUBIndexRound);
    OP_LOGI(tilingContext, "lastUBIndexCount: %ld.", tilingData_->lastUBIndexCount);
    OP_LOGI(tilingContext, "lastUBIndexTail: %ld.", tilingData_->lastUBIndexTail);
    OP_LOGI(tilingContext, "tilingKey: %u.", tilingKey);
}

ge::graphStatus TilingInplaceIndexAddWithSorted(gert::TilingContext* context)
{
    InplaceIndexAddWithSortedTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForInplaceIndexAddWithSorted(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForInplaceIndexAddWithSorted start.");
    auto compileInfo = context->GetCompiledInfo<InplaceIndexAddWithSortedCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForInplaceIndexAddWithSorted end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplaceIndexAddWithSorted)
    .Tiling(TilingInplaceIndexAddWithSorted)
    .TilingParse<InplaceIndexAddWithSortedCompileInfo>(TilingPrepareForInplaceIndexAddWithSorted);
} // namespace optiling
