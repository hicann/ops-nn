/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file swi_glu_quant_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "swi_glu_quant_tiling_utils.h"


namespace optiling {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t L2_CACHE_LINE_SIZE = 512; // pack unit in cache 512B

constexpr uint32_t SINGLE_UB_SIZE = 25;

static std::map<const ge::DataType, const uint32_t> x_dTypeLen = { { ge::DT_FLOAT16, 2 },
    { ge::DT_BF16, 2 },
    { ge::DT_FLOAT, 4 } };

inline static ge::graphStatus SetTilingDataForSwiGluQuant(gert::TilingContext *context,
    SwiGluQuantTilingData &tilingData)
{
    OP_LOGD(context, "SetTilingDataForSwiGluQuant start.");
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGD(context, "SetTilingDataForSwiGluQuant end.");
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus GetCompileInfo(gert::TilingContext *context, SwiGluQuantCompileInfo &compileInfo)
{
    OP_LOGD(context, "GetCompileInfo start.");
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv(); // 总核数
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);

    if (totalCoreNum == 0 || ubSize <= 0) {
        OP_LOGD(context, "GetCompileInfo Failed, coreNum:%d, ubSize:%d.", totalCoreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    compileInfo.totalCore = totalCoreNum;
    compileInfo.ubSize = ubSize;
    OP_LOGD(context, "GetCompileInfo end.");
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus GetTillingData(gert::TilingContext *context, SwiGluQuantCompileInfo &compileInfo,
    SwiGluQuantTilingParam &tilingParam, SwiGluQuantTilingData &tilingData)
{
    OP_LOGD(context, "GetTillingData start.");
    if (CheckOpParams(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto xDtype = context->GetInputDesc(INPUT_X_INDEX)->GetDataType();
    uint32_t tilingKey = CalculateTilingKey(xDtype, compileInfo);

    tilingData.set_tilingKey(tilingKey);
    tilingData.set_groupLen(compileInfo.groupLength);
    compileInfo.inputDataByte = x_dTypeLen[xDtype];
    compileInfo.dataNumSingleUb = compileInfo.ubSize / SINGLE_UB_SIZE;

    // UB空间可处理的最大数据量，32-Byte对齐
    compileInfo.block_num = BLOCK_SIZE / compileInfo.inputDataByte;
    compileInfo.cacheLineLen = L2_CACHE_LINE_SIZE / compileInfo.inputDataByte;

    // 1.计算rowLen和colLen
    auto inputShape = context->GetInputTensor(INPUT_X_INDEX)->GetStorageShape();
    if (!SetTotalShape(inputShape, context, tilingData)) {
        return ge::GRAPH_FAILED;
    }
    CalTilingData(context, compileInfo, tilingParam, tilingData);
    OP_LOGD(context, "GetTillingData end.");
    SetTilingData(compileInfo, tilingParam, tilingData);
    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus Tiling4SwiGluQuant(gert::TilingContext *context)
{
    OP_LOGD(context, "Tiling4SwiGluQuant start.");
    SwiGluQuantCompileInfo compileInfo;
    SwiGluQuantTilingParam tilingParam;
    SwiGluQuantTilingData tilingData;
    if (GetCompileInfo(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 读取属性和dtype来计算tilingKey
    GetTillingData(context, compileInfo, tilingParam, tilingData);
    SetTilingDataForSwiGluQuant(context, tilingData);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t *workspaces = context->GetWorkspaceSizes(1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaces[0] = sysWorkspaceSize + tilingData.get_realCoreNum() * BLOCK_SIZE;
    OP_LOGD(context, "Tiling4SwiGluQuant end.");
    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingPrepare4SwiGluQuant(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4SwiGluQuant start and end.");
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(SwiGluQuant).Tiling(Tiling4SwiGluQuant).TilingParse<CoreCompileInfo>(TilingPrepare4SwiGluQuant);
} // namespace optiling
