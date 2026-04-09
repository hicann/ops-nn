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
 * \file hardtanh_grad_tiling.cpp
 * \brief
 */

#include <algorithm> 
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/hardtanh_grad_tiling_data.h"
#include "../op_kernel/hardtanh_grad_tiling_key.h"

namespace optiling {
#define UB_DATA_NUM_UINT8 16U // 对应DT_FLOAT, DT_INT32, DT_UINT32类型的ub分块数量
#define UB_DATA_NUM_OTHER 8U // 对应DT_FLOAT16, DT_BFLOAT16, DT_INT16, DT_UINT16其他数据类型的ub分块数量
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t GM_ALIGN = 512;
constexpr uint32_t REPEAT_ALIGN = 256;
constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024; // 910b, ub固定预留8k，用于select api的mode1和mode2
constexpr uint32_t DB_THRESHOLD_VALUE = 6144 * 20;
constexpr uint32_t MAX_TILEDATA = 6144;
constexpr uint32_t TILE_BASE_SIZE = 1024;
constexpr uint32_t TILE_THRESHOLD_FACTOR = 8;
constexpr uint32_t TILE_MAX_REGULAR_INDEX = 4;
constexpr uint32_t BITS_PER_BYTE = 8;
struct HardtanhGradCompileInfo {};

static ge::graphStatus TilingParseForHardtanhGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeWithLargeShape(uint64_t& tilingKey, int64_t& coreNum, 
                           uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& finalBigTileNum, uint64_t& finalSmallTileNum, 
                           uint64_t& tileDataNum, uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& tailBlockNum,
                           int64_t real_coreNum, uint64_t inputLengthAlgin512, uint64_t elemsPerGmBlock)
{
    tileDataNum = MAX_TILEDATA; 
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    
    // 核间切分
    int64_t needCoreNum = (inputLengthAlgin512 + tileDataNum * BUFFER_NUM - 1) / (tileDataNum * BUFFER_NUM);
    coreNum = ((real_coreNum) < needCoreNum) ? real_coreNum : needCoreNum;
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    uint64_t needTileDataNum = (needCoreDataNum + BUFFER_NUM - 1) / BUFFER_NUM;
    if (elemsPerGmBlock == 0) {
        return ge::GRAPH_FAILED;
    }    
    needTileDataNum = (needTileDataNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;
    tileDataNum = (tileDataNum < needTileDataNum) ? tileDataNum : needTileDataNum;
    if (elemsPerGmBlock == 0) {
        return ge::GRAPH_FAILED;
    }        
    uint64_t everyCoreInputBlockNum = inputLengthAlgin512 / elemsPerGmBlock;
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }     
    everyCoreInputBlockNum = everyCoreInputBlockNum / coreNum;
    tailBlockNum = (inputLengthAlgin512 / elemsPerGmBlock) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * elemsPerGmBlock;
    if (tileDataNum == 0) {
        return ge::GRAPH_FAILED;
    }     
    finalSmallTileNum = (smallCoreDataNum + tileDataNum - 1) / tileDataNum;
    smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum;
    bigCoreDataNum = smallCoreDataNum + elemsPerGmBlock;
    finalBigTileNum = (bigCoreDataNum + tileDataNum - 1) / tileDataNum;
    bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum;   
    return ge::GRAPH_SUCCESS;
}

void SetTilingParam(uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum, uint64_t finalSmallTileNum,
                    uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, uint64_t maskTileDataNum, float min, float max,
                    HardtanhGradTilingData* tiling)
{
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->maskTileDataNum = maskTileDataNum;
    tiling->min = min;
    tiling->max = max; 
}


static ge::graphStatus ComputeWithSmallShape(uint64_t& tilingKey, int64_t& coreNum, 
                           uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& finalBigTileNum, uint64_t& finalSmallTileNum, 
                           uint64_t& tileDataNum, uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& tailBlockNum,
                           int64_t real_coreNum, uint64_t inputLengthAlgin512, uint64_t elemsPerGmBlock)
{
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    uint64_t threshold = TILE_BASE_SIZE * TILE_THRESHOLD_FACTOR;
    if (threshold == 0) {
        return ge::GRAPH_FAILED;
    }        
    uint64_t index = (inputLengthAlgin512 - 1) / threshold;
    tileDataNum = (index < TILE_MAX_REGULAR_INDEX) ? 
        (TILE_BASE_SIZE * (index + 1)) : MAX_TILEDATA;
    // 计算coreNum
    if (tileDataNum == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t needCoreNum = (inputLengthAlgin512 + tileDataNum  - 1) / tileDataNum;
    if (needCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }        
    tileDataNum = (inputLengthAlgin512 + needCoreNum  - 1) / needCoreNum;
    if (elemsPerGmBlock == 0) {
        return ge::GRAPH_FAILED;
    }     
    tileDataNum = (tileDataNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;
    if (tileDataNum == 0) {
        return ge::GRAPH_FAILED;
    }        
    coreNum = (inputLengthAlgin512 + tileDataNum  - 1) / tileDataNum;
    smallTailDataNum = inputLengthAlgin512 - (coreNum - 1) * tileDataNum;
    smallCoreDataNum = 0;
    bigCoreDataNum = 0;
    bigTailDataNum = 0;
    finalSmallTileNum = 0;
    finalBigTileNum = 0;
    tailBlockNum = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HardtanhGradTilingFunc(gert::TilingContext* context)
{
    HardtanhGradTilingData* tiling = context->GetTilingData<HardtanhGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardtanhGradTilingData), 0, sizeof(HardtanhGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    int64_t real_coreNum, coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, real_coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    float min = *context->GetAttrs()->GetFloat(0);
    float max = *context->GetAttrs()->GetFloat(1);
    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(1)->GetDataType(), typeLength);
    uint64_t elemsPerGmBlock = (GM_ALIGN / typeLength);
    uint64_t inputLengthAlgin512 = (inputNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum, tileDataNum, tilingKey;
    if (inputNum > DB_THRESHOLD_VALUE) 
    {// 大shape
        ComputeWithLargeShape(tilingKey, coreNum, smallCoreDataNum, bigCoreDataNum, finalBigTileNum, finalSmallTileNum, tileDataNum, smallTailDataNum, bigTailDataNum,
                              tailBlockNum, real_coreNum, inputLengthAlgin512, elemsPerGmBlock);
    }
    else
    { // 小shape
        ComputeWithSmallShape(tilingKey, coreNum, smallCoreDataNum, bigCoreDataNum, finalBigTileNum, finalSmallTileNum, tileDataNum, smallTailDataNum, bigTailDataNum,
                              tailBlockNum, real_coreNum, inputLengthAlgin512, elemsPerGmBlock);
    }
    uint64_t maskTileDataNum = (tileDataNum + BITS_PER_BYTE - 1)/ BITS_PER_BYTE;
    maskTileDataNum = (maskTileDataNum + REPEAT_ALIGN - 1) / REPEAT_ALIGN * REPEAT_ALIGN;
    SetTilingParam(smallCoreDataNum, bigCoreDataNum, finalBigTileNum, finalSmallTileNum, 
               tileDataNum, smallTailDataNum, bigTailDataNum, tailBlockNum, maskTileDataNum, min, max, tiling);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);   
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(HardtanhGrad).Tiling(HardtanhGradTilingFunc).TilingParse<HardtanhGradCompileInfo>(TilingParseForHardtanhGrad);
} // namespace optiling
