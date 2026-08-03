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
 * \file nll_loss_grad_tiling.cpp
 * \brief NllLossGrad 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/nll_loss_grad_tiling_data.h"
#include "../op_kernel/nll_loss_grad_tiling_key.h"
#include <cstring>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t RESERVE_UB = 8 * 1024; // 预留 UB(标量/临时/对齐)

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static int64_t ParseReduction(gert::TilingContext* context)
{
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return 2; // mean
    }
    const char* reduction = attrs->GetAttrPointer<char>(0);
    if (reduction == nullptr) {
        return 2;
    }
    if (strcmp(reduction, "none") == 0) {
        return 0;
    }
    if (strcmp(reduction, "sum") == 0) {
        return 1;
    }
    return 2; // mean
}

static ge::graphStatus NllLossGradTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("NllLossGrad", "context is nullptr"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    NllLossGradTilingData* tiling = context->GetTilingData<NllLossGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 解析 x shape -> (N, C)
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    int64_t nDim = 1;
    int64_t cDim = 1;
    if (xShape.GetDimNum() == 1) {
        nDim = 1;
        cDim = xShape.GetDim(0);
    } else {
        nDim = xShape.GetDim(0);
        cDim = xShape.GetDim(1);
    }
    OP_CHECK_IF(nDim <= 0, OP_LOGE(context, "nDim <= 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cDim <= 0, OP_LOGE(context, "cDim <= 0"), return ge::GRAPH_FAILED);

    // 浮点 dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType floatDtype = inputDesc->GetDataType();
    bool isLowPrec = (floatDtype == ge::DT_FLOAT16 || floatDtype == ge::DT_BF16);
    int64_t tSize = isLowPrec ? 2 : 4;

    // target dtype (input index 2)
    auto targetDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    ge::DataType targetDtype = targetDesc->GetDataType();
    int64_t targetSize = 4;
    if (targetDtype == ge::DT_INT64) {
        targetSize = 8;
    } else if (targetDtype == ge::DT_UINT8) {
        targetSize = 1;
    }

    int64_t reduction = ParseReduction(context);

    // reduction 属性 offset 1 -> ignore_index
    int64_t ignoreIndex = -100;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* ip = attrs->GetAttrPointer<int64_t>(1);
        if (ip != nullptr) {
            ignoreIndex = *ip;
        }
    }

    // 行切分
    int64_t validCore = coreNum;
    if (validCore > nDim) {
        validCore = nDim;
    }
    if (validCore < 1) {
        validCore = 1;
    }
    int64_t baseLine = nDim / validCore;
    int64_t tailLine = nDim % validCore;

    // 可用 UB。显式核算所有 buffer（含 TQue 双缓冲 + 低精度 fp32 中间 buffer），
    // 因此不再用 2/3 heuristic，直接以物理 UB 减固定预留为硬上限。
    int64_t usableUb = static_cast<int64_t>(ubSize) - RESERVE_UB;
    int64_t cAlign = (cDim + 7) / 8 * 8;

    int64_t bigWeight = 0;
    int64_t lineTile = 1;
    int64_t outUbSize = 0;
    int64_t colTile = 0;
    int64_t moveOutTime = 1;

    // 固定占用：常驻 weight(fp32)；低精度再加一份 weightT(T)
    int64_t weightFixed = cAlign * 4;
    if (isLowPrec) {
        weightFixed += cAlign * tSize;
    }
    // 每行 UB 占用（NormalWeight）：
    //   fp32   : outQue(fp32)*2 + targetQue*2 + yGradQue(fp32)*2
    //   低精度 : outQue(T)*2 + outFloat(fp32) + targetQue*2 + yGradQue(T)*2 + yGradFloat(fp32)
    int64_t perLineBytes;
    if (isLowPrec) {
        perLineBytes = cDim * tSize * 2 + cDim * 4 + targetSize * 2 + tSize * 2 + 4;
    } else {
        perLineBytes = cDim * 4 * 2 + targetSize * 2 + 4 * 2;
    }
    int64_t reserveMisc = 1024; // scalarBuf + 各 buffer 32B 对齐余量
    int64_t availForOut = usableUb - weightFixed - reserveMisc;
    if (availForOut >= perLineBytes) {
        // NormalWeight
        bigWeight = 0;
        lineTile = availForOut / perLineBytes;
        int64_t maxLinePerCore = baseLine + (tailLine > 0 ? 1 : 0);
        if (lineTile > maxLinePerCore) {
            lineTile = maxLinePerCore;
        }
        if (lineTile < 1) {
            lineTile = 1;
        }
        outUbSize = lineTile * cDim;
        outUbSize = (outUbSize + 7) / 8 * 8;
    } else {
        // BigWeight: 按列切分。outFloat(fp32) 占 usableUb 的一半以内，低精度再留 outT(T)。
        bigWeight = 1;
        int64_t denom = isLowPrec ? (4 + tSize) : 4;
        colTile = (usableUb - reserveMisc) / denom;
        colTile = colTile / 8 * 8;
        if (colTile < 8) {
            colTile = 8;
        }
        if (colTile > cDim) {
            colTile = (cDim + 7) / 8 * 8;
        }
        moveOutTime = (cDim + colTile - 1) / colTile;
    }

    tiling->nDim = nDim;
    tiling->cDim = cDim;
    tiling->coreNum = validCore;
    tiling->reduction = reduction;
    tiling->ignoreIndex = ignoreIndex;
    tiling->bigWeight = bigWeight;
    tiling->maxLine = (tailLine > 0) ? (baseLine + 1) : baseLine;
    tiling->lowerLine = baseLine;
    tiling->redundantLine = tailLine;
    tiling->lineTile = lineTile;
    tiling->cAlign = cAlign;
    tiling->outUbSize = outUbSize;
    tiling->colTile = colTile;
    tiling->moveOutTime = moveOutTime;

    context->SetBlockDim(validCore);

    // tilingKey: 浮点dtype × target dtype 组合，顺序与 op proto 一致
    int floatIdx = 0; // 0 float32, 1 bf16, 2 float16
    if (floatDtype == ge::DT_BF16) {
        floatIdx = 1;
    } else if (floatDtype == ge::DT_FLOAT16) {
        floatIdx = 2;
    }
    int targetIdx = 0; // 0 int32, 1 int64, 2 uint8
    if (targetDtype == ge::DT_INT64) {
        targetIdx = 1;
    } else if (targetDtype == ge::DT_UINT8) {
        targetIdx = 2;
    }
    // schMode 排布:
    //   float32/int32=0 bf16/int32=1 float32/int64=2 bf16/int64=3
    //   float32/uint8=4 bf16/uint8=5 float16/int32=6 float16/int64=7 float16/uint8=8
    uint32_t schMode;
    if (floatIdx < 2) {
        schMode = static_cast<uint32_t>(targetIdx * 2 + floatIdx); // 0..5
    } else {
        schMode = static_cast<uint32_t>(6 + targetIdx); // 6..8
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_0);
    switch (schMode) {
        case 0:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_0);
            break;
        case 1:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_1);
            break;
        case 2:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_2);
            break;
        case 3:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_3);
            break;
        case 4:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_4);
            break;
        case 5:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_5);
            break;
        case 6:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_6);
            break;
        case 7:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_7);
            break;
        case 8:
            tilingKey = GET_TPL_TILING_KEY(NLLLOSSGRAD_TPL_SCH_MODE_8);
            break;
        default:
            break;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForNllLossGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct NllLossGradCompileInfo {};

IMPL_OP_OPTILING(NllLossGrad)
    .Tiling(NllLossGradTilingFunc)
    .TilingParse<NllLossGradCompileInfo>(TilingParseForNllLossGrad);

} // namespace optiling
