/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file silu_grad_tiling.cpp
 */
#include <algorithm>
#include <cstdint>
#include "../op_kernel/silu_grad_tiling_data.h"
#include "../op_kernel/silu_grad_tiling_key.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

using Ops::Base::GetUbBlockSize;
// 单缓冲 UB 常数
#define UB_NUM_FLOAT_SINGLE 5U
#define UB_NUM_FLOAT16_SINGLE 13U
// 双缓冲 UB 常数
#define UB_NUM_FLOAT_DOUBLE 8U
#define UB_NUM_FLOAT16_DOUBLE 16U
constexpr uint32_t WS_SYS_SIZE = 0;

struct SiluGradCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        OP_LOGE(context, "GetPlatformInfo: context or platformInfo is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    if (coreNum == 0 || ubSize == 0) {
        OP_LOGE(context, "GetPlatformInfo: coreNum(%ld) or ubSize(%lu) is zero.", coreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE(context, "GetWorkspaceSize: context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSiluGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum,
                                         uint64_t& inputBytes, uint64_t& singleTileDataNum, uint64_t& doubleTileDataNum,
                                         uint64_t& inputLengthAlgin, uint64_t& ubBlockSize)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr) {
        OP_LOGE(context, "GetShapeAttrsInfo: context or inputShape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 2;
    ge::DataType dt = context->GetInputDesc(0)->GetDataType();
    if (dt == ge::DT_FLOAT) {
        typeLength = 4;
    }
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        OP_LOGE(context, "GetShapeAttrsInfo: inputNum is 0.");
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    if (inputBytes == 0) {
        OP_LOGE(context, "GetShapeAttrsInfo: inputBytes is 0.");
        return ge::GRAPH_FAILED;
    }
    ubBlockSize = static_cast<uint64_t>(GetUbBlockSize(context));
    // 单缓冲 tileDataNum
    uint64_t singleUbDataNumber = UB_NUM_FLOAT16_SINGLE;
    if (dt == ge::DT_FLOAT) {
        singleUbDataNumber = UB_NUM_FLOAT_SINGLE;
    }
    if (ubBlockSize == 0 || singleUbDataNumber == 0) {
        OP_LOGE(context, "GetShapeAttrsInfo: ubBlockSize(%lu) or singleUbDataNumber(%lu) is zero.", ubBlockSize,
                singleUbDataNumber);
        return ge::GRAPH_FAILED;
    }
    uint64_t singleTileBlockNum = (ubSize / ubBlockSize) / singleUbDataNumber;
    singleTileDataNum = (singleTileBlockNum * ubBlockSize) / inputBytes;
    // 双缓冲 tileDataNum
    uint64_t doubleUbDataNumber = UB_NUM_FLOAT16_DOUBLE;
    if (dt == ge::DT_FLOAT) {
        doubleUbDataNumber = UB_NUM_FLOAT_DOUBLE;
    }
    if (doubleUbDataNumber == 0) {
        OP_LOGE(context, "GetShapeAttrsInfo: doubleUbDataNumber is 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t doubleTileBlockNum = (ubSize / ubBlockSize) / doubleUbDataNumber;
    doubleTileDataNum = (doubleTileBlockNum * ubBlockSize) / inputBytes;
    inputLengthAlgin = (((inputLength + ubBlockSize - 1) / ubBlockSize) * ubBlockSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(gert::TilingContext* context, uint64_t inputLengthAlgin, int64_t coreNum,
                                              uint64_t inputBytes, uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum,
                                              uint64_t& tailBlockNum, uint64_t ubBlockSize)
{
    if (0 == ubBlockSize || 0 == coreNum || 0 == inputBytes) {
        OP_LOGE(context, "CalculateCoreBlockNums: ubBlockSize(%lu) coreNum(%ld) inputBytes(%lu) invalid.", ubBlockSize,
                coreNum, inputBytes);
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlgin / ubBlockSize / coreNum;
    tailBlockNum = (inputLengthAlgin / ubBlockSize) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * ubBlockSize / inputBytes;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * ubBlockSize / inputBytes;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncSiluGrad(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "TilingFuncSiluGrad: GetPlatformInfo failed.");
        return ge::GRAPH_FAILED;
    }
    // 2、获取shape、属性信息
    uint64_t inputNum, inputBytes, singleTileDataNum, doubleTileDataNum, inputLengthAlgin, ubBlockSize;
    ret = GetShapeAttrsInfo(context, ubSize, inputNum, inputBytes, singleTileDataNum, doubleTileDataNum,
                            inputLengthAlgin, ubBlockSize);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "TilingFuncSiluGrad: GetShapeAttrsInfo failed.");
        return ge::GRAPH_FAILED;
    }
    // 3、获取WorkspaceSize信息
    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "TilingFuncSiluGrad: GetWorkspaceSize failed.");
        return ge::GRAPH_FAILED;
    }

    // 4、设置tiling信息
    SiluGradTilingData* tiling = context->GetTilingData<SiluGradTilingData>();
    if (tiling == nullptr) {
        OP_LOGE(context, "TilingFuncSiluGrad: GetTilingData failed.");
        return ge::GRAPH_FAILED;
    }
    if (memset_s(tiling, sizeof(SiluGradTilingData), 0, sizeof(SiluGradTilingData)) != EOK) {
        OP_LOGE(context, "TilingFuncSiluGrad: memset_s failed.");
        return ge::GRAPH_FAILED;
    }

    // 计算coreNum：目标每核约1024个元素，避免过度并行
    constexpr uint64_t TARGET_ELEM_PER_CORE = 1024;
    uint64_t calcCoreNum = (inputNum + TARGET_ELEM_PER_CORE - 1) / TARGET_ELEM_PER_CORE;
    uint64_t maxBlockCoreNum = inputLengthAlgin / ubBlockSize;
    uint64_t platformCoreNum = static_cast<uint64_t>(coreNum);

    uint64_t finalCoreNum = platformCoreNum < calcCoreNum ? platformCoreNum : calcCoreNum;
    finalCoreNum = finalCoreNum < maxBlockCoreNum ? finalCoreNum : maxBlockCoreNum;
    coreNum = static_cast<int64_t>(finalCoreNum);
    if (coreNum <= 0) {
        coreNum = 1;
    }
    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, tailBlockNum;
    ret = CalculateCoreBlockNums(context, inputLengthAlgin, coreNum, inputBytes, smallCoreDataNum, bigCoreDataNum,
                                 tailBlockNum, ubBlockSize);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "TilingFuncSiluGrad: CalculateCoreBlockNums failed.");
        return ge::GRAPH_FAILED;
    }
    // 决定 schMode：单核 tile 数 <= 2 时走单缓冲（增大 tile 消除 tail），否则走双缓冲流水线
    uint64_t schMode;
    uint64_t tileDataNum;
    if (smallCoreDataNum <= 2 * singleTileDataNum) {
        schMode = SILU_GRAD_TPL_SCH_MODE_SINGLE;
        tileDataNum = singleTileDataNum;
    } else {
        schMode = SILU_GRAD_TPL_SCH_MODE_DOUBLE;
        tileDataNum = doubleTileDataNum;
    }

    // 设置tiling数据
    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint64_t>(tileDataNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);

    context->SetBlockDim(coreNum);
    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(schMode);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SiluGrad).Tiling(TilingFuncSiluGrad).TilingParse<SiluGradCompileInfo>(TilingParseForSiluGrad);
} // namespace optiling
