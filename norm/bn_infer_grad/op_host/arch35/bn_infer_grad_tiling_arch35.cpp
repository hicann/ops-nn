/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file bn_infer_grad_tiling.cpp
 * \brief BNInferGrad Tiling 实现（通用，支持 arch32/arch35）
 *
 * Tiling 通过 platform API 动态获取平台参数，
 * 因此同一份代码可适配 arch32 (Ascend910B) 和 arch35 (Ascend950) 等架构。
 *
 * 迭代三：支持 CONTIGUOUS(TilingKey=0) + NC1HWC0(TilingKey=1)，
 * 多核切分，NCHW/NHWC/NC1HWC0 三种格式，
 * 全 dtype 支持（fp32/fp16/bf16），边界处理。
 * inv_std 在 Kernel 侧从 scale 和 batch_variance 计算。
 */

#include <cmath>
#include <cstring>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/arch35/bn_infer_grad_tiling_data.h"
#include "../op_kernel/arch35/bn_infer_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::GetAivCoreNum;
using Ops::Base::GetUbSize;

constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t FLOAT_SIZE = 4U;
constexpr uint32_t ALIGN_ELEM_FP32 = BLOCK_SIZE / FLOAT_SIZE;  // 8 个 float = 32 字节
constexpr uint32_t SCH_CONTIGUOUS = 0;
constexpr uint32_t SCH_NC1HWC0 = 1;

// UB 每元素总占用（字节）：
// fp32 路径: inQueue(2*4) + outQueue(2*4) + invStdExpand(4) = 20
// fp16/bf16 路径: inQueue(2*2) + outQueue(2*2) + invStdExpand(4) + cast(4) + result(4) = 20
constexpr uint32_t BYTES_PER_ELEM = 20U;

static inline int64_t AlignUp(int64_t value, int64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

static inline int64_t AlignDown(int64_t value, int64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value / alignment) * alignment;
}

// Tiling 计算上下文：在各阶段 helper 之间传递平台/形状/输出参数
struct BNInferGradCtx {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t totalElements = 0;
    int64_t rank = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_NCHW;
    float epsilon = 0.0001f;

    int64_t channelSize = 0;
    int64_t spatialSize = 0;
    int64_t N = 0;
    int64_t C1 = 0;
    int64_t C0 = 0;
    int64_t formatMode = 0;
    int64_t alignedC = 0;

    uint32_t schMode = SCH_CONTIGUOUS;
    int64_t usedCoreNum = 1;
    int64_t elemsPerCore = 0;
    int64_t tailCoreElems = 0;
    int64_t tileLen = 0;
    int64_t numTiles = 0;
    int64_t lastTileLen = 0;
    int64_t totalTasks = 0;
    int64_t tasksPerCore = 0;
    int64_t tailCoreTasks = 0;
    int64_t tileHW = 0;
    int64_t numTilesHW = 0;
    int64_t lastTileHW = 0;
    int64_t alignedC0 = 0;
};

static ge::graphStatus HandleEmptyTensorTiling(gert::TilingContext* context, ge::DataType dataType)
{
    OP_LOGW(context, "BNInferGrad: totalElements is 0 (empty tensor), skipping");
    BNInferGradTilingData* tiling = context->GetTilingData<BNInferGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BNInferGradTilingData), 0, sizeof(BNInferGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    context->SetBlockDim(1);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    uint32_t schMode = SCH_CONTIGUOUS;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, schMode);
    return ge::GRAPH_SUCCESS;
}

// 解析 epsilon 与 format_mode 属性；对 5D 输入自动识别为 NC1HWC0
static void ParseEpsilonAndFormat(gert::TilingContext* context, int64_t rank,
                                  float& epsilon, ge::Format& format)
{
    epsilon = 0.0001f;
    format = ge::FORMAT_NCHW;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* epsilonPtr = attrs->GetFloat(0);
        if (epsilonPtr != nullptr) {
            epsilon = *epsilonPtr;
        }
    }
    if (rank == 5) {
        format = ge::FORMAT_NC1HWC0;
        return;
    }
    if (attrs == nullptr) {
        return;
    }
    const int64_t* fmtModePtr = attrs->GetInt(1);
    if (fmtModePtr == nullptr) {
        return;
    }
    if (*fmtModePtr == 1) {
        format = ge::FORMAT_NHWC;
    } else if (*fmtModePtr == 2) {
        format = ge::FORMAT_NC1HWC0;
    }
}

static ge::graphStatus ExtractShapeInfo(gert::TilingContext* context,
                                        const gert::Shape& gradsShape, BNInferGradCtx& ctx)
{
    int64_t rank = ctx.rank;
    if (ctx.format == ge::FORMAT_NCHW || ctx.format == ge::FORMAT_ND) {
        OP_CHECK_IF(rank < 4, OP_LOGE(context, "NCHW/ND rank < 4"), return ge::GRAPH_FAILED);
        ctx.N = gradsShape.GetDim(0);
        ctx.channelSize = gradsShape.GetDim(1);
        ctx.spatialSize = 1;
        for (int64_t i = 2; i < rank; i++) {
            ctx.spatialSize *= gradsShape.GetDim(i);
        }
        ctx.formatMode = 0;
        return ge::GRAPH_SUCCESS;
    }
    if (ctx.format == ge::FORMAT_NHWC) {
        OP_CHECK_IF(rank < 4, OP_LOGE(context, "NHWC rank < 4"), return ge::GRAPH_FAILED);
        ctx.N = gradsShape.GetDim(0);
        ctx.channelSize = gradsShape.GetDim(rank - 1);
        ctx.spatialSize = 1;
        for (int64_t i = 1; i < rank - 1; i++) {
            ctx.spatialSize *= gradsShape.GetDim(i);
        }
        ctx.formatMode = 1;
        return ge::GRAPH_SUCCESS;
    }
    if (ctx.format == ge::FORMAT_NC1HWC0) {
        OP_CHECK_IF(rank < 5, OP_LOGE(context, "NC1HWC0 rank < 5"), return ge::GRAPH_FAILED);
        ctx.N = gradsShape.GetDim(0);
        ctx.C1 = gradsShape.GetDim(1);
        ctx.spatialSize = gradsShape.GetDim(2) * gradsShape.GetDim(3);
        ctx.C0 = gradsShape.GetDim(4);
        ctx.channelSize = ctx.C1 * ctx.C0;
        ctx.formatMode = 2;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(context, "Unsupported format");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus ComputeContiguousTiling(gert::TilingContext* context, BNInferGradCtx& ctx)
{
    // UB 切分：
    // 总 UB 占用 = tileLen * BYTES_PER_ELEM + alignedC * 3 * FLOAT_SIZE
    // 3 个 alignedC 大小的 buffer: invStdBuf + scaleBuf + varianceBuf
    int64_t overhead = ctx.alignedC * 3 * static_cast<int64_t>(FLOAT_SIZE);
    int64_t tileLen = (ctx.ubSize - overhead) / static_cast<int64_t>(BYTES_PER_ELEM);
    tileLen = AlignDown(tileLen, static_cast<int64_t>(ALIGN_ELEM_FP32));
    OP_CHECK_IF(tileLen <= 0, OP_LOGE(context, "tileLen <= 0, UB too small"), return ge::GRAPH_FAILED);
    // NHWC: 对齐到 C 以保证通道组完整；C 太大无法对齐时回退到基本对齐
    if (ctx.formatMode == 1 && ctx.channelSize > 0 && tileLen >= ctx.channelSize) {
        tileLen = AlignDown(tileLen, ctx.channelSize);
        if (tileLen <= 0) {
            tileLen = AlignDown((ctx.ubSize - overhead) / static_cast<int64_t>(BYTES_PER_ELEM),
                                static_cast<int64_t>(ALIGN_ELEM_FP32));
        }
    }
    OP_CHECK_IF(tileLen <= 0, OP_LOGE(context, "tileLen <= 0 after format align"), return ge::GRAPH_FAILED);
    int64_t maxCores = CeilDiv(ctx.totalElements, tileLen); // 限制核数避免单核数据太少
    int64_t usedCoreNum = (maxCores < ctx.coreNum) ? maxCores : ctx.coreNum;
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    int64_t elemsPerCore = CeilDiv(ctx.totalElements, usedCoreNum);
    int64_t coreAlignment = static_cast<int64_t>(ALIGN_ELEM_FP32);
    if (ctx.formatMode == 1 && ctx.channelSize > coreAlignment) {
        coreAlignment = ctx.channelSize;
    }
    elemsPerCore = AlignUp(elemsPerCore, coreAlignment);
    usedCoreNum = CeilDiv(ctx.totalElements, elemsPerCore);
    int64_t tailCoreElems = ctx.totalElements - (usedCoreNum - 1) * elemsPerCore;
    if (tailCoreElems <= 0) {
        tailCoreElems = elemsPerCore;
    }
    int64_t numTiles = CeilDiv(elemsPerCore, tileLen);
    int64_t lastTileLen = elemsPerCore - (numTiles - 1) * tileLen;
    if (lastTileLen <= 0) {
        lastTileLen = tileLen;
    }
    ctx.usedCoreNum = usedCoreNum;
    ctx.elemsPerCore = elemsPerCore;
    ctx.tailCoreElems = tailCoreElems;
    ctx.tileLen = tileLen;
    ctx.numTiles = numTiles;
    ctx.lastTileLen = lastTileLen;
    OP_LOGI(context, "BNInferGrad CONTIGUOUS: tileLen=%ld numTiles=%ld lastTileLen=%ld "
            "usedCoreNum=%ld elemsPerCore=%ld tailCoreElems=%ld",
            tileLen, numTiles, lastTileLen, usedCoreNum, elemsPerCore, tailCoreElems);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeNc1hwc0TileSize(gert::TilingContext* context, BNInferGradCtx& ctx)
{
    // overhead: invStdBuf(alignedC * 4) + scaleBuf(alignedC * 4) + varianceBuf(alignedC * 4)
    int64_t overheadNc1hwc0 = ctx.alignedC * 3 * static_cast<int64_t>(FLOAT_SIZE);
    int64_t hwc0TileLen = (ctx.ubSize - overheadNc1hwc0) / static_cast<int64_t>(BYTES_PER_ELEM);
    int64_t tileHW = hwc0TileLen / ctx.C0;
    if (tileHW > ctx.spatialSize) {
        tileHW = ctx.spatialSize;
    }
    OP_CHECK_IF(tileHW <= 0, OP_LOGE(context, "tileHW <= 0, UB too small"), return ge::GRAPH_FAILED);

    int64_t hwc0Tile = AlignDown(tileHW * ctx.C0, static_cast<int64_t>(ALIGN_ELEM_FP32));
    tileHW = hwc0Tile / ctx.C0;
    OP_CHECK_IF(tileHW <= 0, OP_LOGE(context, "tileHW <= 0 after align"), return ge::GRAPH_FAILED);

    ctx.tileHW = tileHW;
    ctx.tileLen = tileHW * ctx.C0;
    return ge::GRAPH_SUCCESS;
}

static void SplitNc1hwc0Tasks(BNInferGradCtx& ctx)
{
    int64_t totalTasks = ctx.N * ctx.C1;
    int64_t usedCoreNum = (totalTasks < ctx.coreNum) ? totalTasks : ctx.coreNum;
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    int64_t tasksPerCore = CeilDiv(totalTasks, usedCoreNum);
    usedCoreNum = CeilDiv(totalTasks, tasksPerCore);
    int64_t tailCoreTasks = totalTasks - (usedCoreNum - 1) * tasksPerCore;
    if (tailCoreTasks <= 0) {
        tailCoreTasks = tasksPerCore;
    }
    int64_t numTilesHW = CeilDiv(ctx.spatialSize, ctx.tileHW);
    int64_t lastTileHW = ctx.spatialSize - (numTilesHW - 1) * ctx.tileHW;
    if (lastTileHW <= 0) {
        lastTileHW = ctx.tileHW;
    }
    ctx.usedCoreNum = usedCoreNum;
    ctx.elemsPerCore = tasksPerCore * ctx.spatialSize * ctx.C0;
    ctx.tailCoreElems = tailCoreTasks * ctx.spatialSize * ctx.C0;
    ctx.numTiles = numTilesHW;
    ctx.lastTileLen = lastTileHW * ctx.C0;
    ctx.totalTasks = totalTasks;
    ctx.tasksPerCore = tasksPerCore;
    ctx.tailCoreTasks = tailCoreTasks;
    ctx.numTilesHW = numTilesHW;
    ctx.lastTileHW = lastTileHW;
}

static ge::graphStatus ComputeNc1hwc0Tiling(gert::TilingContext* context, BNInferGradCtx& ctx)
{
    // 显式 0 校验，防止后续除法触发除零（静态分析也据此识别 C0 != 0）
    if (ctx.C0 <= 0) {
        OP_LOGE(context, "C0 <= 0");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ctx.spatialSize <= 0, OP_LOGE(context, "spatialSize <= 0"), return ge::GRAPH_FAILED);

    ctx.alignedC0 = AlignUp(ctx.C0, static_cast<int64_t>(ALIGN_ELEM_FP32));

    ge::graphStatus status = ComputeNc1hwc0TileSize(context, ctx);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    SplitNc1hwc0Tasks(ctx);

    OP_LOGI(context, "BNInferGrad NC1HWC0: tileHW=%ld numTilesHW=%ld lastTileHW=%ld "
            "totalTasks=%ld usedCoreNum=%ld tasksPerCore=%ld tailCoreTasks=%ld",
            ctx.tileHW, ctx.numTilesHW, ctx.lastTileHW, ctx.totalTasks, ctx.usedCoreNum,
            ctx.tasksPerCore, ctx.tailCoreTasks);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillTilingData(gert::TilingContext* context,
                                      BNInferGradTilingData* tiling, const BNInferGradCtx& ctx)
{
    tiling->totalElements = ctx.totalElements;
    tiling->channelSize = ctx.channelSize;
    tiling->spatialSize = ctx.spatialSize;
    tiling->formatMode = ctx.formatMode;
    tiling->N = ctx.N;
    tiling->C1 = ctx.C1;
    tiling->C0 = ctx.C0;
    tiling->usedCoreNum = ctx.usedCoreNum;
    tiling->elemsPerCore = ctx.elemsPerCore;
    tiling->tailCoreElems = ctx.tailCoreElems;
    tiling->totalTasks = ctx.totalTasks;
    tiling->tasksPerCore = ctx.tasksPerCore;
    tiling->tailCoreTasks = ctx.tailCoreTasks;
    tiling->tileLen = ctx.tileLen;
    tiling->numTiles = ctx.numTiles;
    tiling->lastTileLen = ctx.lastTileLen;
    tiling->tileHW = ctx.tileHW;
    tiling->numTilesHW = ctx.numTilesHW;
    tiling->lastTileHW = ctx.lastTileHW;
    tiling->alignedC = ctx.alignedC;
    tiling->alignedC0 = ctx.alignedC0;

    // 将 epsilon float 位模式存入 int64_t（通过 memcpy_s 避免 strict aliasing 问题）
    int64_t epsBits = 0;
    float epsVal = ctx.epsilon;
    OP_CHECK_IF(memcpy_s(&epsBits, sizeof(epsBits), &epsVal, sizeof(epsVal)) != EOK,
        OP_LOGE(context, "copy epsilon bits error"), return ge::GRAPH_FAILED);
    tiling->epsilonBits = epsBits;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitContextAndShape(gert::TilingContext* context, BNInferGradCtx& ctx,
                                           gert::Shape& gradsShape)
{
    ctx.coreNum = static_cast<int64_t>(GetAivCoreNum(context));
    OP_CHECK_IF(ctx.coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ctx.ubSize = static_cast<int64_t>(GetUbSize(context));
    OP_CHECK_IF(ctx.ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    auto gradsDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsDesc);
    auto gradsShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsShapePtr);
    gradsShape = gradsShapePtr->GetStorageShape();

    ctx.dataType = gradsDesc->GetDataType();
    ctx.rank = static_cast<int64_t>(gradsShape.GetDimNum());
    ctx.totalElements = gradsShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BNInferGradTilingFunc(gert::TilingContext* context)
{
    BNInferGradCtx ctx{};
    gert::Shape gradsShape;
    if (InitContextAndShape(context, ctx, gradsShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ctx.totalElements == 0) {
        return HandleEmptyTensorTiling(context, ctx.dataType);
    }
    OP_CHECK_IF(ctx.totalElements < 0, OP_LOGE(context, "totalElements < 0"), return ge::GRAPH_FAILED);

    // 框架的 AutoContiguous 会将 NHWC 等格式归一化为 ND，
    // 因此使用显式属性区分 NCHW 与 NHWC；5D 输入直接识别为 NC1HWC0
    ParseEpsilonAndFormat(context, ctx.rank, ctx.epsilon, ctx.format);
    OP_LOGI(context, "BNInferGrad: format=%d rank=%ld dtype=%d",
            (int)ctx.format, ctx.rank, (int)ctx.dataType);
    OP_CHECK_IF(ExtractShapeInfo(context, gradsShape, ctx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ExtractShapeInfo failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ctx.channelSize <= 0, OP_LOGE(context, "channelSize <= 0"), return ge::GRAPH_FAILED);
    OP_LOGI(context, "BNInferGrad: format=%d N=%ld C=%ld spatial=%ld total=%ld dtype=%d coreNum=%ld",
            (int)ctx.formatMode, ctx.N, ctx.channelSize, ctx.spatialSize, ctx.totalElements,
            (int)ctx.dataType, ctx.coreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    BNInferGradTilingData* tiling = context->GetTilingData<BNInferGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BNInferGradTilingData), 0, sizeof(BNInferGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    ctx.alignedC = AlignUp(ctx.channelSize, static_cast<int64_t>(ALIGN_ELEM_FP32));
    ctx.schMode = (ctx.format == ge::FORMAT_NC1HWC0) ? SCH_NC1HWC0 : SCH_CONTIGUOUS;
    ge::graphStatus status = (ctx.schMode == SCH_CONTIGUOUS)
        ? ComputeContiguousTiling(context, ctx)
        : ComputeNc1hwc0Tiling(context, ctx);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    if (FillTilingData(context, tiling, ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(ctx.usedCoreNum);
    uint32_t dTypeX = static_cast<uint32_t>(ctx.dataType);
    uint32_t schModeOut = ctx.schMode;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, schModeOut);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBNInferGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BNInferGradCompileInfo {};

IMPL_OP_OPTILING(BNInferGrad)
    .Tiling(BNInferGradTilingFunc)
    .TilingParse<BNInferGradCompileInfo>(TilingParseForBNInferGrad);

} // namespace optiling
