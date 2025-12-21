/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file linear_index_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "linear_index_tiling.h"

using namespace std;

namespace {
const int BUFFER_NUM = 1;
const int TWO_DIM = 2;
const int THREE_DIM = 3;

const int SIZE_OF_INT32 = 4;
const int SIZE_OF_INT64 = 8;
const int SIZE_OF_FLOAT = 4;

const int DT_INT32_TYPE = 1;
const int DT_INT64_TYPE = 2;
const int COMBINE_DIM0 = 10;
const int COMBINE_DIM1 = 20;

const int INPUT_0 = 0;
const int INPUT_1 = 1;
const int OUTPUT_0 = 0;
const int DATA_ALIGN = 32;
} // namespace

namespace optiling {
class LinearIndexTiling
{
public:
    explicit LinearIndexTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    LinearIndexTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t tilingKey = 0;
    uint64_t usedCoreNum = 0;
    uint64_t eachCount = 1;
    uint64_t lastCount = 1;
    uint64_t indicesCount = 1;
    uint64_t indicesAlign = 0;
    uint64_t indicesSize = 8;
    uint64_t eachNum = 0;
    uint64_t eachLoop = 0;
    uint64_t eachTail = 0;
    uint64_t lastNum = 0;
    uint64_t lastLoop = 0;
    uint64_t lastTail = 0;
    uint64_t maxSize = 0;
    uint64_t target = 0;
    uint64_t selfStride = 0;
    uint64_t indicesStride = 0;
    uint64_t workspaceSize = 1024 * 1024 * 16;
    uint64_t max_ub = 20480;
};

ge::graphStatus LinearIndexTiling::Init()
{
    if (tilingContext == nullptr) {
        OP_LOGE("LinearIndex", "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const LinearIndexCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    static uint32_t coreNum = static_cast<uint32_t>(compileInfo->totalCoreNum);
    if (coreNum == 0) {
        OP_LOGE(tilingContext, "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    workspaceSize = compileInfo->workspaceSize;
    auto ubSizePlatForm = compileInfo->ubSizePlatForm;
    max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
    OP_LOGD(tilingContext, "ubSizePlatForm: %lu.", ubSizePlatForm);

    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr || tilingContext->GetInputShape(INPUT_0) == nullptr ||
        tilingContext->GetInputShape(INPUT_1) == nullptr || tilingContext->GetInputDesc(INPUT_0) == nullptr ||
        tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE(tilingContext, "tilingContext inputshape or outputshape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto indicesShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto varShapeTensor = tilingContext->GetInputTensor(INPUT_1);
    auto varShapeSize = varShapeTensor->GetShapeSize();

    // 输入是二维时，indices会被slice成一维
    // 输入是三维时，indices会被slice成二维
    int dim = *(attrs->GetAttrPointer<int>(0));
    bool combine = *(attrs->GetAttrPointer<bool>(1));
    if (dim < 0) {
        dim += varShapeSize;
    }

    auto varShapeData = varShapeTensor->GetData<int>();
    if (varShapeData != nullptr && varShapeSize > dim) {
        target = varShapeData[dim];
    }

    // 二维场景，合轴和不合轴没有区别
    if (combine) {
        if (varShapeData != nullptr && varShapeSize >= TWO_DIM) {
            selfStride = varShapeData[1];
        }
        if (varShapeSize == THREE_DIM && dim == 0) {
            tilingKey += COMBINE_DIM0; // 三维合轴场景，dim = 0
            indicesStride = indicesShape.GetDim(1);
            indicesSize += SIZE_OF_FLOAT + SIZE_OF_INT32;
        } else if (varShapeSize == THREE_DIM && dim == 1) {
            tilingKey += COMBINE_DIM1; // 三维合轴场景，dim = 1
            indicesStride = indicesShape.GetDim(1);
            indicesSize += SIZE_OF_FLOAT;
        }
    }

    auto indicesDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    if (ge::DT_INT32 == indicesDtype) {
        tilingKey += DT_INT32_TYPE;
    } else if (ge::DT_INT64 == indicesDtype) {
        tilingKey += DT_INT64_TYPE;
        indicesSize += SIZE_OF_INT64;
    } else {
        OP_LOGE(tilingContext, "indices only support int32, int64.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < indicesShape.GetDimNum(); ++i) {
        auto dimIndice = indicesShape.GetDim(i);
        indicesCount *= dimIndice;
    }

    if (indicesCount == 0) {
        OP_LOGE(tilingContext, "shape cannot equal 0.");
        return ge::GRAPH_FAILED;
    }

    uint32_t times = (indicesCount + DATA_ALIGN - 1) / DATA_ALIGN;

    usedCoreNum = times < coreNum ? times : coreNum;

    max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
    maxSize = max_ub / indicesSize;

    eachCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (indicesCount + eachCount - 1) / eachCount;
    lastCount = indicesCount - eachCount * (usedCoreNum - 1);

    eachNum = eachCount;
    eachLoop = 1;
    eachTail = eachCount;
    if (eachCount > maxSize) {
        eachNum = maxSize;
        eachLoop = (eachCount + maxSize - 1) / maxSize;
        eachTail = eachCount - (eachLoop - 1) * eachNum;
    }

    lastNum = lastCount;
    lastLoop = 1;
    lastTail = lastCount;
    if (lastCount > maxSize) {
        lastNum = maxSize;
        lastLoop = (lastCount + maxSize - 1) / maxSize;
        lastTail = lastCount - (lastLoop - 1) * lastNum;
    }

    indicesAlign = DATA_ALIGN / SIZE_OF_INT32;
    indicesAlign = (eachNum + indicesAlign - 1) / indicesAlign * indicesAlign;

    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LinearIndexTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");

    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_eachCount(eachCount);
    tilingData.set_lastCount(lastCount);
    tilingData.set_indicesCount(indicesCount);
    tilingData.set_indicesAlign(indicesAlign);
    tilingData.set_maxSize(maxSize);
    tilingData.set_eachNum(eachNum);
    tilingData.set_eachLoop(eachLoop);
    tilingData.set_eachTail(eachTail);
    tilingData.set_lastNum(lastNum);
    tilingData.set_lastLoop(lastLoop);
    tilingData.set_lastTail(lastTail);
    tilingData.set_target(target);
    tilingData.set_selfStride(selfStride);
    tilingData.set_indicesStride(indicesStride);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(usedCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;
    TilingDataPrint();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void LinearIndexTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "usedCoreNum: %lu.", usedCoreNum);
    OP_LOGD(tilingContext, "indicesCount: %lu.", indicesCount);
    OP_LOGD(tilingContext, "indicesAlign: %lu.", indicesAlign);
    OP_LOGD(tilingContext, "eachCount: %lu.", eachCount);
    OP_LOGD(tilingContext, "lastCount: %lu.", lastCount);
    OP_LOGD(tilingContext, "maxSize: %lu.", maxSize);
    OP_LOGD(tilingContext, "eachNum: %lu.", eachNum);
    OP_LOGD(tilingContext, "eachLoop: %lu.", eachLoop);
    OP_LOGD(tilingContext, "eachTail: %lu.", eachTail);
    OP_LOGD(tilingContext, "lastNum: %lu.", lastNum);
    OP_LOGD(tilingContext, "lastLoop: %lu.", lastLoop);
    OP_LOGD(tilingContext, "lastTail: %lu.", lastTail);
    OP_LOGD(tilingContext, "target: %lu.", target);
    OP_LOGD(tilingContext, "selfStride: %lu.", selfStride);
    OP_LOGD(tilingContext, "indicesStride: %lu.", indicesStride);
    OP_LOGD(tilingContext, "tilingKey: %u.", tilingKey);
    OP_LOGD(tilingContext, "max_ub: %lu.", max_ub);
}

ge::graphStatus TilingLinearIndex(gert::TilingContext* context)
{
    LinearIndexTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForLinearIndex(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForLinearIndex start.");
    auto compileInfo = context->GetCompiledInfo<LinearIndexCompileInfo>();
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
    OP_LOGD(context, "TilingPrepareForLinearIndex end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LinearIndex)
    .Tiling(TilingLinearIndex)
    .TilingParse<LinearIndexCompileInfo>(TilingPrepareForLinearIndex);
} // namespace optiling
