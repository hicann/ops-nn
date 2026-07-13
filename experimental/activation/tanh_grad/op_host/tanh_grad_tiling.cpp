/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file tanh_grad_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/tanh_grad_tiling_data.h"
#include "../op_kernel/tanh_grad_tiling_key.h"
#include <cmath>
#include <algorithm>

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t BUFFER_NUM = 2U;

struct TanhGradCompileInfo {};

static ge::graphStatus TilingParseForTanhGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetPlatformInfo() == nullptr, OP_LOGE(context, "GetPlatformInfo is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TanhGradTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2、获取shape、属性信息
    OP_CHECK_IF(context->GetInputShape(0) == nullptr, OP_LOGE(context, "GetInputShape is nullptr"),
                return ge::GRAPH_FAILED);
    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t inputBytes = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), inputBytes);
    if (inputNum == 0 || inputBytes == 0) {
        OP_LOGE(context, "inputNum or inputBytes is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputLength = inputNum * inputBytes;

    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 4、tileDataNum
    uint32_t blockDataNum = BLOCK_SIZE / inputBytes;
    bool needFloatPromote = (inputBytes == 2U);
    uint32_t ubBytesPerElem = BUFFER_NUM * 3U * inputBytes + (needFloatPromote ? (2U * sizeof(float)) : 0U);
    uint32_t ubMaxElements = static_cast<uint32_t>(ubSize) / ubBytesPerElem; // 用满 UB
    uint32_t tileDataNum = (ubMaxElements / blockDataNum) * blockDataNum;    // 向下对齐到块
    if (tileDataNum == 0) {
        tileDataNum = blockDataNum; // 保证 >= 1 块
    }

    uint32_t tileBlockNum = (tileDataNum * inputBytes) / BLOCK_SIZE;
    if (tileBlockNum == 0)
        tileBlockNum = 1;

    uint64_t inputLengthAlgin = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    uint32_t totalAlignedBlocks = inputLengthAlgin / BLOCK_SIZE;
    if (static_cast<uint32_t>(coreNum) > totalAlignedBlocks) {
        coreNum = totalAlignedBlocks;
    }

    // 5、计算coreNum（平台核数为上限，按数据量收敛；不再使用 cost-model 返回值）
    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlgin / BLOCK_SIZE) ?
                      coreNum :
                      static_cast<int64_t>(inputLengthAlgin / BLOCK_SIZE);
    }

    // 6、计算每个core处理的数据块数（大小核负载均衡）
    uint64_t everyCoreInputBlockNum = inputLengthAlgin / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin / BLOCK_SIZE) % coreNum;

    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 保证 tileNum >= 1
    if (finalSmallTileNum == 0)
        finalSmallTileNum = 1;
    if (finalBigTileNum == 0)
        finalBigTileNum = 1;

    // 7、设置tiling信息
    TanhGradTilingData* tiling = context->GetTilingData<TanhGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TanhGradTilingData), 0, sizeof(TanhGradTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    context->SetBlockDim(coreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TanhGrad).Tiling(TanhGradTilingFunc).TilingParse<TanhGradCompileInfo>(TilingParseForTanhGrad);
} // namespace optiling
