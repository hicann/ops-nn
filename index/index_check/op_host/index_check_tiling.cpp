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
 * \file index_check_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "index_check_tiling.h"
#include <algorithm>

constexpr int64_t SCALE_SPACE = 20480;
constexpr int64_t MIN_UB_SIZE = 1024;
constexpr uint64_t BLOCK_BYTES = 32;
constexpr size_t MAX_DIM_NUM = 8;

namespace optiling {
class IndexCheckTiling {
public:
    explicit IndexCheckTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    void CalcCoreAndBatch();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    gert::TilingContext* tilingContext_ = nullptr;
    IndexCheckTilingData tilingData_;
    uint64_t workspaceSize_ = 0;
    ge::DataType indicesDtype_ = ge::DT_UNDEFINED;
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t tensorId_ = 0;
    uint64_t maxBatchSize_ = 0;
    std::vector<uint64_t> tensorLens_;
};

ge::graphStatus IndexCheckTiling::Init()
{
    OP_LOGD(tilingContext_, "Tiling start.");
    if (tilingContext_ == nullptr) {
        OP_LOGE(tilingContext_, "tilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = tilingContext_->GetCompileInfo<IndexCheckCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);
    totalCoreNum_ = compileInfo->totalCoreNum;
    workspaceSize_ = compileInfo->workspaceSize;
    OP_CHECK_IF(compileInfo->ubSizePlatform < static_cast<uint64_t>(SCALE_SPACE + MIN_UB_SIZE),
                OP_LOGE(tilingContext_, "platform ub size %lu is less than the minimum %ld",
                        compileInfo->ubSizePlatform, SCALE_SPACE + MIN_UB_SIZE),
                return ge::GRAPH_FAILED);
    ubSize_ = compileInfo->ubSizePlatform - static_cast<uint64_t>(SCALE_SPACE);
    OP_LOGD(tilingContext_, "totalCoreNum: %lu, ubSize: %lu, workspaceSize: %lu", totalCoreNum_, ubSize_,
            workspaceSize_);

    auto computeNodeInfoPtr = tilingContext_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, computeNodeInfoPtr);

    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, idxInstanceInfoPtr);
    tensorId_ = idxInstanceInfoPtr->GetInstanceNum();
    OP_CHECK_IF(tensorId_ == 0, OP_LOGE(tilingContext_, "indices can not be a empty tensor list"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tensorId_ > MAX_TENSOR_NUM,
                OP_LOGE(tilingContext_, "indices tensor num %lu exceeds max %lu", tensorId_, MAX_TENSOR_NUM),
                return ge::GRAPH_FAILED);

    auto idxTensorDtypePtr = tilingContext_->GetDynamicInputDesc(1, 0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, idxTensorDtypePtr);
    indicesDtype_ = idxTensorDtypePtr->GetDataType();
    if (indicesDtype_ != ge::DT_INT32 && indicesDtype_ != ge::DT_INT64) {
        OP_LOGE(tilingContext_, "indices only support int32 or int64");
        return ge::GRAPH_FAILED;
    }

    auto boundsShapePtr = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, boundsShapePtr);
    auto boundsShape = boundsShapePtr->GetStorageShape();
    OP_CHECK_IF(boundsShape.GetDimNum() != 1 || boundsShape.GetDim(0) != static_cast<int64_t>(tensorId_),
                OP_LOGE(tilingContext_, "bounds must be a 1-D tensor with %lu elements", tensorId_),
                return ge::GRAPH_FAILED);

    tensorLens_.resize(MAX_TENSOR_NUM, 0);
    for (uint64_t i = 0; i < tensorId_; i++) {
        auto idxTensorShapePtr = tilingContext_->GetDynamicInputShape(1, i);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, idxTensorShapePtr);
        auto idxTensorShape = idxTensorShapePtr->GetStorageShape();
        OP_CHECK_IF(idxTensorShape.GetDimNum() > MAX_DIM_NUM,
                    OP_LOGE(tilingContext_, "indices tensor[%lu] dim num %zu exceeds max %lu", i,
                            idxTensorShape.GetDimNum(), MAX_DIM_NUM),
                    return ge::GRAPH_FAILED);
        uint64_t tensorLen = 1;
        for (uint32_t d = 0; d < idxTensorShape.GetDimNum(); d++) {
            tensorLen *= idxTensorShape.GetDim(d);
        }
        tensorLens_[i] = tensorLen;
        OP_LOGD(tilingContext_, "tensor[%lu] len: %lu", i, tensorLen);
    }

    CalcCoreAndBatch();

    OP_LOGD(tilingContext_, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void IndexCheckTiling::CalcCoreAndBatch()
{
    uint64_t maxTensorLen = 0;
    for (uint64_t i = 0; i < tensorId_; i++) {
        if (tensorLens_[i] > maxTensorLen) {
            maxTensorLen = tensorLens_[i];
        }
    }
    usedCoreNum_ = std::min(totalCoreNum_, maxTensorLen);
    if (usedCoreNum_ == 0) {
        usedCoreNum_ = 1;
    }

    uint64_t bytesPerElement;
    if (indicesDtype_ == ge::DT_INT64) {
        // int64 path is scalar-only (vector queues not instantiated): the
        // indices queue is the sole per-element buffer.
        bytesPerElement = sizeof(int64_t);
    } else {
        bytesPerElement = sizeof(int32_t) + sizeof(int32_t) + sizeof(float) + sizeof(float) + sizeof(int32_t) +
                          sizeof(int32_t);
    }
    uint64_t alignUnit = BLOCK_BYTES / sizeof(int32_t);
    uint64_t maxBatch = ubSize_ / bytesPerElement;
    maxBatchSize_ = (maxBatch / alignUnit) * alignUnit;
    if (maxBatchSize_ == 0) {
        maxBatchSize_ = alignUnit;
    }
}

ge::graphStatus IndexCheckTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext_, "set tiling data start.");
    tilingData_.params.set_usedCoreNum(usedCoreNum_);
    tilingData_.params.set_tensorId(tensorId_);
    tilingData_.params.set_maxBatchSize(maxBatchSize_);
    tilingData_.params.set_tensorLens(tensorLens_.data());

    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    OP_CHECK_IF(tilingContext_->SetBlockDim(usedCoreNum_) != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_, "SetBlockDim failed."), return ge::GRAPH_FAILED);
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, workspaces);
    workspaces[0] = workspaceSize_;
    TilingDataPrint();
    OP_LOGD(tilingContext_, "set tiling data end.");
    return ge::GRAPH_SUCCESS;
}

void IndexCheckTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "indicesDtype:           %d", static_cast<int32_t>(indicesDtype_));
    OP_LOGD(tilingContext_, "usedCoreNum:            %lu", usedCoreNum_);
    OP_LOGD(tilingContext_, "tensorId:               %lu", tensorId_);
    OP_LOGD(tilingContext_, "maxBatchSize:           %lu", maxBatchSize_);
    for (uint64_t i = 0; i < tensorId_; i++) {
        OP_LOGD(tilingContext_, "tensorLen[%lu]:         %lu", i, tensorLens_[i]);
    }
}

ge::graphStatus TilingIndexCheck(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "IndexCheck tiling begin");
    IndexCheckTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "tiling init failed");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForIndexCheck(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepareForIndexCheck start");
    auto compileInfo = context->GetCompiledInfo<IndexCheckCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    compileInfo->ubSizePlatform = ubSizePlatform;
    OP_CHECK_IF((compileInfo->ubSizePlatform == 0), OP_LOGE(context, "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform: %lu", compileInfo->ubSizePlatform);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size: %lu", totalUbSize);
    OP_LOGD(context, "TilingPrepareForIndexCheck end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexCheck).Tiling(TilingIndexCheck).TilingParse<IndexCheckCompileInfo>(TilingPrepareForIndexCheck);
} // namespace optiling
