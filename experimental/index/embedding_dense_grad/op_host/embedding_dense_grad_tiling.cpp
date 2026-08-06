/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_tiling.cpp
 * \brief
 */
#include "embedding_dense_grad_tiling.h"
#include "../op_kernel/embedding_dense_grad_tiling_key.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t RESERVED_UB_SIZE = 1 * 1024;
constexpr uint64_t BUFFER_NUM = 1;

class EmbeddingDenseGradTiling {
public:
    EmbeddingDenseGradTiling(gert::TilingContext* ctx) { this->context = ctx; }

    inline ge::graphStatus Init()
    {
        OP_LOGD(context, "Tiling initing");
        InitFromInput();
        auto ascendCPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendCPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        coreNum = ascendCPlatform.GetCoreNumAiv();
        coreNum = std::min(coreNum, batchSize);
        coreNum = scaleGrad ? std::min(coreNum, (uint64_t)numWeights) : coreNum;
        size_t usrSize = 2 * 1024 + CeilAlign(numWeights, BLOCK_SIZE / 4) * sizeof(float);
        if (gradTypeLength == 2) {
            usrSize += CeilAlign((uint64_t)numWeights * dimSize, BLOCK_SIZE / sizeof(float)) * sizeof(float);
        }
        size_t sysWorkspaceSize = ascendCPlatform.GetLibApiWorkSpaceSize();
        size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
        currentWorkSpace[0] = sysWorkspaceSize + usrSize;
        BaseTiling();

        context->SetNeedAtomic(true);
        OP_LOGD(context, "Tiling inited");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetKernelTiling()
    {
        context->SetBlockDim(coreNum);
        context->SetTilingKey(tilingKey);
        tilingData.set_dimSize(dimSize);
        tilingData.set_numWeights(numWeights);
        tilingData.set_paddingIdx(paddingIdx);
        tilingData.set_scaleGradByFreq(scaleGrad ? 1 : 0);
        tilingData.set_formerCoreNum(formerCoreNum);
        tilingData.set_formerBatchSize(formerBatchSize);
        tilingData.set_tailBatchSize(tailBatchSize);
        tilingData.set_scaleFormerCoreNum(scaleFormerCoreNum);
        tilingData.set_scaleFormerBatchSize(scaleFormerBatchSize);
        tilingData.set_scaleTailBatchSize(scaleTailBatchSize);
        tilingData.set_ubProcessNum(ubProcessNum);
        tilingData.set_scaleUbProcessNum(scaleUbProcessNum);

        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        TilingDataPrint();
        return ge::GRAPH_SUCCESS;
    }

    void TilingDataPrint() const
    {
        OP_LOGD(context, "tilingKey:             %u", tilingKey);
        OP_LOGD(context, "dimSize:               %lu", dimSize);
        OP_LOGD(context, "numWeights:            %ld", numWeights);
        OP_LOGD(context, "paddingIdx:            %ld", paddingIdx);
        OP_LOGD(context, "scaleGrad:             %d", scaleGrad);
        OP_LOGD(context, "batchSize:             %lu", batchSize);
        OP_LOGD(context, "ubSize:                %lu", ubSize);
        OP_LOGD(context, "coreNum:               %lu", coreNum);
        OP_LOGD(context, "formerBatchSize:       %lu", formerBatchSize);
        OP_LOGD(context, "tailBatchSize:         %lu", tailBatchSize);
        OP_LOGD(context, "scaleFormerCoreNum:    %lu", scaleFormerCoreNum);
        OP_LOGD(context, "scaleFormerBatchSize:  %lu", scaleFormerBatchSize);
        OP_LOGD(context, "scaleTailBatchSize:    %lu", scaleTailBatchSize);
        OP_LOGD(context, "formerCoreNum:         %lu", formerCoreNum);
        OP_LOGD(context, "ubProcessNum:          %ld", ubProcessNum);
        OP_LOGD(context, "scaleUbProcessNum:     %lu", scaleUbProcessNum);
    }

private:
    gert::TilingContext* context;
    uint64_t dimSize;
    int64_t numWeights;
    int64_t paddingIdx;
    uint64_t batchSize = 1;
    uint64_t ubSize;
    int64_t ubProcessNum;
    uint64_t scaleUbProcessNum;
    uint64_t coreNum;
    uint32_t tilingKey;
    uint32_t gradTypeLength;
    uint32_t indicesTypeLength;

    bool scaleGrad = false;

    uint64_t formerBatchSize;
    uint64_t tailBatchSize;
    uint64_t formerCoreNum;
    uint64_t scaleFormerCoreNum;
    uint64_t scaleFormerBatchSize;
    uint64_t scaleTailBatchSize;

    optiling::EmbeddingDenseGradTilingData tilingData;

    inline void InitFromInput()
    {
        auto selfShape = context->GetInputShape(0)->GetStorageShape();
        dimSize = selfShape.GetDim(selfShape.GetDimNum() - 1);
        for (size_t i = 0; i < selfShape.GetDimNum() - 1; i++) {
            batchSize *= selfShape.GetDim(i);
        }
        auto attrs = context->GetAttrs();
        numWeights = *(attrs->GetAttrPointer<int64_t>)(0);
        paddingIdx = *(attrs->GetAttrPointer<int64_t>)(1);
        scaleGrad = *(attrs->GetAttrPointer<bool>)(2);
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), gradTypeLength);
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(1)->GetDataType(), indicesTypeLength);
    }

    inline void BaseTiling()
    {
        OP_LOGD(context, "BaseTiling start");
        formerCoreNum = batchSize % coreNum;
        tailBatchSize = batchSize / coreNum;
        formerBatchSize = tailBatchSize + 1;
        scaleFormerCoreNum = numWeights % coreNum;
        scaleTailBatchSize = numWeights / coreNum;
        scaleFormerBatchSize = scaleTailBatchSize + 1;
        SetTilingKey();
        OP_LOGD(context, "BaseTiling finish");
    }

    inline void SetTilingKey()
    {
        int avaliableUbSize = ubSize - RESERVED_UB_SIZE;

        if (gradTypeLength == 2) {
            int64_t alignFp32DimSize = CeilAlign(dimSize, BLOCK_SIZE / sizeof(float));
            int64_t fp32Tmp = avaliableUbSize / (BUFFER_NUM * sizeof(float)) / 32 * 32;
            scaleUbProcessNum = avaliableUbSize / (BUFFER_NUM * sizeof(float) + gradTypeLength) / 32 * 32;
            ubProcessNum = fp32Tmp;
            tilingKey = (fp32Tmp > 0 && alignFp32DimSize < avaliableUbSize) ? EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW :
                                                                              EMBEDDING_DENSE_GRAD_SCH_MODE_SEGMENTED;
            return;
        }

        scaleUbProcessNum = avaliableUbSize / BUFFER_NUM / gradTypeLength / 32 * 32;
        int64_t alignTDimSize = CeilAlign(dimSize, BLOCK_SIZE / gradTypeLength);
        int64_t tmp = avaliableUbSize / (BUFFER_NUM * gradTypeLength) / 32 * 32;

        bool dAligned32B = (gradTypeLength == 4) && !scaleGrad && (dimSize % (BLOCK_SIZE / gradTypeLength) == 0);
        if (dAligned32B && tmp > 0 && alignTDimSize <= tmp) {
            tilingKey = EMBEDDING_DENSE_GRAD_SCH_MODE_PACKED;
            ubProcessNum = tmp;
            return;
        }

        if (tmp > 0 && alignTDimSize < avaliableUbSize) {
            tilingKey = EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW;
            ubProcessNum = tmp;
            return;
        }
        tilingKey = EMBEDDING_DENSE_GRAD_SCH_MODE_SEGMENTED;
        ubProcessNum = avaliableUbSize / (BUFFER_NUM * gradTypeLength) / 32 * 32;
    }

    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    }
};
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    EmbeddingDenseGradTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetKernelTiling();
}

static ge::graphStatus TilingPrepare4EmbeddingDenseGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling prepare for EmbeddingDenseGrad start");
    auto compileInfo = context->GetCompiledInfo<EmbeddingDenseGradCompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE(context, "compileInfo is null");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context, "platformInfo is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    if (compileInfo->ubSizePlatForm == 0) {
        OP_LOGE(context, "Failed to get ub size");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Tiling prepare for EmbeddingDenseGrad end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingDenseGrad)
    .Tiling(TilingFunc)
    .TilingParse<EmbeddingDenseGradCompileInfo>(TilingPrepare4EmbeddingDenseGrad);
} // namespace optiling
