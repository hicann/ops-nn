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
 * \file layer_norm_quant_tiling.cpp
 * \brief
 */

#include "layer_norm_quant_tiling.h"
#include "layer_norm_quant_tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using namespace Ops::NN::OpTiling;
using namespace Ops::NN::Optiling;
constexpr uint32_t LAYER_NORM_TILING_KEY_BASE = 2000000000;
constexpr uint32_t LAYER_NORM_TILING_KEY_DTYPE = 100000000; // 0: fp16; 1: bf16
constexpr uint32_t LAYER_NORM_TILING_KEY_FAST = 10000000;   // 0: fast; 1:slice

constexpr int64_t FP16_DATA_USED = 5;
constexpr int64_t FP16_OTHER_USED = 6;
constexpr uint32_t SCALAR_USED = 50;
constexpr uint32_t NUM_TEMP_BUF = 32;
constexpr uint32_t MEAN_AND_VAR_SIZE = 64;

void LayerNormQuantTiling::PrintTilingData()
{
    OP_LOGD(context, "Start LayerNormQuantTilingData priting");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "numCore is %u", tilingData.get_numCore());
    OP_LOGD(context, "numFirstDim is %u", tilingData.get_numFirstDim());
    OP_LOGD(context, "lFirstdimPerCore is %u", tilingData.get_lFirstdimPerCore());
    OP_LOGD(context, "nlFirstdimPerCore is %u", tilingData.get_nlFirstdimPerCore());
    OP_LOGD(context, "epsStr is %f", tilingData.get_epsStr());
    OP_LOGD(context, "aveStr is %f", tilingData.get_aveStr());
    OP_LOGD(context, "numLastDim is %u", tilingData.get_numLastDim());
    OP_LOGD(context, "firstDimPerTimes is %u", tilingData.get_firstDimPerTimes());
    OP_LOGD(context, "sliceNum is %u", tilingData.get_sliceNum());
    OP_LOGD(context, "sliceSize is %u", tilingData.get_sliceSize());
    OP_LOGD(context, "tailSliceSize is %u", tilingData.get_tailSliceSize());
    OP_LOGD(context, "tilingKey is %lu", context->GetTilingKey());
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "End LayerNormQuantTilingData priting");
}

size_t roundDown(size_t size, size_t divisor)
{
    if (divisor == 0) {
        return size;
    }
    return size / divisor * divisor;
}

ge::graphStatus LayerNormQuantTiling::GetTilingSliceInfo()
{
    uint32_t singleRowSizePerElem = fp32BufNum * sizeof(uint32_t) + fp16BufNum * sizeof(uint16_t); // 3*4 + 2*2  -->  3*4 + 2*2
    uint32_t multiRowSizePerElem = fp16BufNumForMulRow * sizeof(uint16_t) + i8BufNumForMulRow * sizeof(uint8_t); // 2*2 +1

    OP_CHECK_IF(numCol > (UINT_MAX / (singleRowSizePerElem + multiRowSizePerElem)),
                    OP_LOGE(context->GetNodeName(), "RowBufferSize invalid!"),
                    return ge::GRAPH_FAILED);
    uint32_t singleRowBufferSize = singleRowSizePerElem * numCol;
    uint32_t multiRowBufferSize = multiRowSizePerElem * numCol;

    if ((maxUbSize - MEAN_AND_VAR_SIZE) < (singleRowBufferSize + multiRowBufferSize)) {
        uint32_t oneRepeatElemCount = 256U / 2;
        uint32_t elemSize = roundDown((maxUbSize - MEAN_AND_VAR_SIZE) / (singleRowSizePerElem + multiRowSizePerElem),
                                      oneRepeatElemCount);
        tilingData.set_sliceNum(CeilDiv(numCol, elemSize));
        tilingData.set_sliceSize(elemSize);
        tilingData.set_tailSliceSize(numCol - (tilingData.get_sliceNum() - 1) * elemSize);
    } else {
        tilingData.set_sliceNum(1);
        tilingData.set_sliceSize(numCol);
        tilingData.set_tailSliceSize(numCol);
    }

    return ge::GRAPH_SUCCESS;
}

void LayerNormQuantTiling::GetTilingBasicInfo()
{
    float tempAve = float(1.0 / numCol);
    tilingData.set_aveStr(tempAve);
    tilingData.set_numLastDim(numCol);
    uint32_t numCore = layerNormPtrCon.numCore;
    tilingData.set_numCore(numCore);
    uint32_t numRow = layerNormPtrCon.numRow;
    tilingData.set_numFirstDim(numRow);
    uint32_t nlFirstdimPerCoreNum = layerNormPtrCon.nlFirstdimPerCoreNum;
    tilingData.set_nlFirstdimPerCore(nlFirstdimPerCoreNum);
    tilingData.set_lFirstdimPerCore(numRow - nlFirstdimPerCoreNum * (numCore - 1));
}

ge::graphStatus LayerNormQuantTiling::startTiling()
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto eps = attrs->GetAttrPointer<float>(EPSILON_ATTR_INDEX);
    this->tilingData.set_epsStr(*eps);

    ge::graphStatus ret =
        PostLayerNormPtrFunc<LayerNormQuantTilingData>(&this->tilingData, this->layerNormPtrCon, this->context);
    if (ret == ge::GRAPH_FAILED) {  // OP_CHECK_IF_STATUS_RETURN(ret);
        return ret;
    }

    this->maxUbSize = layerNormPtrCon.maxUbSize;  // maxUb
    this->numCol = layerNormPtrCon.numCol;
    GetTilingBasicInfo();
    GetTilingSliceInfo();

    if (tilingData.get_sliceNum() == 1) {
        OP_CHECK_IF(layerNormPtrCon.numCol > (UINT_MAX / (FP16_DATA_USED * layerNormPtrCon.nlFirstdimPerCoreNum)),
                        OP_LOGE(context->GetNodeName(), "totalMemNeed is invalid!"),
                        return ge::GRAPH_FAILED);
        uint32_t totalMemNeed =
            static_cast<uint32_t>(FP16_DATA_USED) * layerNormPtrCon.nlFirstdimPerCoreNum * layerNormPtrCon.numCol;
        OP_CHECK_IF(layerNormPtrCon.numCol > (UINT_MAX / FP16_OTHER_USED),
                        OP_LOGE(context->GetNodeName(), "sumData is invalid!"),
                        return ge::GRAPH_FAILED);
        uint32_t sumData = layerNormPtrCon.maxEleFp16 - NUM_TEMP_BUF -
                           static_cast<uint32_t>(FP16_OTHER_USED) * layerNormPtrCon.numCol - SCALAR_USED;

        ret = CheckSplit(&tilingData, totalMemNeed, sumData, layerNormPtrCon, context);
        if (ret == ge::GRAPH_FAILED) {
            return ret;
        }
    } else {
        tilingData.set_firstDimPerTimes(1);
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    auto XDesc = context->GetInputDesc(0);
    auto XType = ge::DT_FLOAT;
    if (XDesc != nullptr) {
        XType = XDesc->GetDataType();
    }

    uint64_t tilingKey = LAYER_NORM_TILING_KEY_BASE;
    tilingKey += XType == ge::DataType::DT_BF16 ? LAYER_NORM_TILING_KEY_DTYPE : 0;
    tilingKey += tilingData.get_sliceNum() == 1 ? LAYER_NORM_TILING_KEY_FAST : 0;
    context->SetTilingKey(tilingKey);  // 2000000000 + 100000000 + 10000000
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CanUseRegbase(gert::TilingContext* context, bool& useRegbase)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        auto npuArch = ascendcPlatform.GetCurNpuArch();
        useRegbase = (IsRegbaseSocVersion(context) ||
                      npuArch == NpuArch::DAV_5102);
    } else {
        auto compileInfo = reinterpret_cast<const Tiling4LayerNormQuantCompileInfo*>(context->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        useRegbase = compileInfo->isRegbase;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4LayerNormQuant(gert::TilingContext* context)
{
    bool useRegbase = false;
    OP_CHECK_IF(
        CanUseRegbase(context, useRegbase) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Check SocInfo Failed"), return ge::GRAPH_FAILED);
    
    if (useRegbase) {
        LayerNormQuantRegTiling Regobject(context);
        auto ret = Regobject.DoTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    } else {
        LayerNormQuantTiling object(context);
        auto ret = object.startTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    *workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4LayerNormQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4LayerNormQuant enter.");
    auto compileInfo = context->GetCompiledInfo<Tiling4LayerNormQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                    OP_LOGE(context->GetNodeName(), "Get core num failed"),
                    return ge::GRAPH_FAILED);

    auto npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfo->isRegbase =
        (IsRegbaseSocVersion(context) ||
         npuArch == NpuArch::DAV_5102) ? true : false;

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                    OP_LOGE(context->GetNodeName(), "Get ub size failed"),
                    return ge::GRAPH_FAILED);

    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context, "TilingPrepare4LayerNormQuant exit.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LayerNormQuant)
    .Tiling(Tiling4LayerNormQuant)
    .TilingParse<Tiling4LayerNormQuantCompileInfo>(TilingPrepare4LayerNormQuant);

}   // namespace optiling
