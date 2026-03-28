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
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file bn_infer_grad_tiling.cpp
 * \brief BnInferGrad Tiling 实现（通用，支持 arch32/arch35）
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
#include "../../op_kernel/common/bn_infer_grad_tiling_data.h"
#include "../../op_kernel/common/bn_infer_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t FLOAT_SIZE = 4U;
constexpr uint32_t ALIGN_ELEM_FP32 = BLOCK_SIZE / FLOAT_SIZE;  // 8 个 float = 32 字节
constexpr uint32_t SCH_CONTIGUOUS = 0;
constexpr uint32_t SCH_NC1HWC0 = 1;

// UB 每元素总占用（字节）：
// fp32 路径: inQueue(2*4) + outQueue(2*4) + invStdExpand(4) = 20
// fp16/bf16 路径: inQueue(2*2) + outQueue(2*2) + invStdExpand(4) + cast(4) + result(4) = 20
constexpr uint32_t BYTES_PER_ELEM = 20U;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

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

static ge::graphStatus BnInferGradTilingFunc(gert::TilingContext* context)
{
    // ================================================================
    // Step 0: 获取平台参数
    // ================================================================
    uint64_t ubSizeU64;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSizeU64, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    int64_t ubSize = static_cast<int64_t>(ubSizeU64);

    // ================================================================
    // Step 1: 获取输入信息
    // ================================================================
    auto gradsDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsDesc);
    auto gradsShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsShapePtr);
    auto gradsShape = gradsShapePtr->GetStorageShape();

    ge::DataType dataType = gradsDesc->GetDataType();

    int64_t rank = static_cast<int64_t>(gradsShape.GetDimNum());
    int64_t totalElements = gradsShape.GetShapeSize();

    // 边界处理：空 tensor（totalElements == 0）
    if (totalElements == 0) {
        OP_LOGW(context, "BnInferGrad: totalElements is 0 (empty tensor), skipping");
        BnInferGradTilingData* tiling = context->GetTilingData<BnInferGradTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(memset_s(tiling, sizeof(BnInferGradTilingData), 0, sizeof(BnInferGradTilingData)) != EOK,
            OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
        context->SetBlockDim(1);
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
        currentWorkspace[0] = 0;
        // 设置 TilingKey 参数（即使空 tensor 也需要有效的 TilingKey）
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        uint32_t schMode = SCH_CONTIGUOUS;
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, schMode);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(totalElements < 0, OP_LOGE(context, "totalElements < 0"), return ge::GRAPH_FAILED);

    // 获取 epsilon 属性（属性索引 0）
    const auto* attrs = context->GetAttrs();
    float epsilon = 0.0001f;
    if (attrs != nullptr) {
        const float* epsilonPtr = attrs->GetFloat(0);
        if (epsilonPtr != nullptr) {
            epsilon = *epsilonPtr;
        }
    }

    // 获取 format_mode 属性（属性索引 1）确定逻辑数据布局。
    // 框架的 AutoContiguous 会将 NHWC 等格式归一化为 ND，
    // 无法通过 GetFormat() 区分 NCHW 与 NHWC，因此使用显式属性。
    // 对于 NC1HWC0（5D 输入），可以通过 rank 自动检测。
    // 取值：0=NCHW(默认) / 1=NHWC / 2=NC1HWC0
    ge::Format format = ge::FORMAT_NCHW;
    if (rank == 5) {
        format = ge::FORMAT_NC1HWC0;
    } else if (attrs != nullptr) {
        const int64_t* fmtModePtr = attrs->GetInt(1);
        if (fmtModePtr != nullptr && *fmtModePtr == 1) {
            format = ge::FORMAT_NHWC;
        } else if (fmtModePtr != nullptr && *fmtModePtr == 2) {
            format = ge::FORMAT_NC1HWC0;
        }
    }

    OP_LOGI(context, "BnInferGrad: format=%d rank=%ld dtype=%d",
            (int)format, rank, (int)dataType);

    // ================================================================
    // Step 2: 提取通道和空间信息
    // ================================================================
    int64_t channelSize = 0;
    int64_t spatialSize = 0;
    int64_t N = 0;
    int64_t C1 = 0;
    int64_t C0 = 0;
    int64_t formatMode = 0;

    if (format == ge::FORMAT_NCHW || format == ge::FORMAT_ND) {
        // NCHW / ND 布局（batch-first, channel-second）
        if (rank >= 4) {
            N = gradsShape.GetDim(0);
            channelSize = gradsShape.GetDim(1);
            spatialSize = 1;
            for (int64_t i = 2; i < rank; i++) {
                spatialSize *= gradsShape.GetDim(i);
            }
        } else {
            OP_LOGE(context, "NCHW/ND rank < 4");
            return ge::GRAPH_FAILED;
        }
        formatMode = 0;
    } else if (format == ge::FORMAT_NHWC) {
        OP_CHECK_IF(rank < 4, OP_LOGE(context, "NHWC rank < 4"), return ge::GRAPH_FAILED);
        N = gradsShape.GetDim(0);
        channelSize = gradsShape.GetDim(rank - 1);
        spatialSize = 1;
        for (int64_t i = 1; i < rank - 1; i++) {
            spatialSize *= gradsShape.GetDim(i);
        }
        formatMode = 1;
    } else if (format == ge::FORMAT_NC1HWC0) {
        OP_CHECK_IF(rank < 5, OP_LOGE(context, "NC1HWC0 rank < 5"), return ge::GRAPH_FAILED);
        N = gradsShape.GetDim(0);
        C1 = gradsShape.GetDim(1);
        spatialSize = gradsShape.GetDim(2) * gradsShape.GetDim(3);
        C0 = gradsShape.GetDim(4);
        channelSize = C1 * C0;
        formatMode = 2;
    } else {
        OP_LOGE(context, "Unsupported format");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(channelSize <= 0, OP_LOGE(context, "channelSize <= 0"), return ge::GRAPH_FAILED);

    OP_LOGI(context, "BnInferGrad: format=%d N=%ld C=%ld spatial=%ld total=%ld dtype=%d coreNum=%ld",
            (int)formatMode, N, channelSize, spatialSize, totalElements, (int)dataType, coreNum);

    // ================================================================
    // Step 3: 设置 workspace（无需预计算，Kernel 侧计算 inv_std）
    // ================================================================
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    // ================================================================
    // Step 4: 计算 Tiling 参数
    // ================================================================
    BnInferGradTilingData* tiling = context->GetTilingData<BnInferGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BnInferGradTilingData), 0, sizeof(BnInferGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 通道对齐到 8 个 float（32 字节）
    int64_t alignedC = AlignUp(channelSize, static_cast<int64_t>(ALIGN_ELEM_FP32));

    // 确定 TilingKey 分支
    uint32_t schMode = SCH_CONTIGUOUS;
    if (format == ge::FORMAT_NC1HWC0) {
        schMode = SCH_NC1HWC0;
    }

    int64_t usedCoreNum = 1;
    int64_t elemsPerCore = totalElements;
    int64_t tailCoreElems = totalElements;
    int64_t tileLen = 0;
    int64_t numTiles = 0;
    int64_t lastTileLen = 0;

    // NC1HWC0 专用参数
    int64_t totalTasks = 0;
    int64_t tasksPerCore = 0;
    int64_t tailCoreTasks = 0;
    int64_t tileHW = 0;
    int64_t numTilesHW = 0;
    int64_t lastTileHW = 0;
    int64_t alignedC0 = 0;

    if (schMode == SCH_CONTIGUOUS) {
        // ============================================================
        // CONTIGUOUS 分支：NCHW/NHWC 多核切分
        // ============================================================

        // UB 切分：tileLen 计算
        // 总 UB 占用 = tileLen * BYTES_PER_ELEM + alignedC * 3 * FLOAT_SIZE
        // 3 个 alignedC 大小的 buffer: invStdBuf + scaleBuf + varianceBuf
        int64_t overhead = alignedC * 3 * static_cast<int64_t>(FLOAT_SIZE);
        tileLen = (ubSize - overhead) / static_cast<int64_t>(BYTES_PER_ELEM);

        // 对齐到 8 个元素（32 字节）
        tileLen = AlignDown(tileLen, static_cast<int64_t>(ALIGN_ELEM_FP32));
        OP_CHECK_IF(tileLen <= 0, OP_LOGE(context, "tileLen <= 0, UB too small"), return ge::GRAPH_FAILED);

        // 根据格式进一步对齐 tileLen
        if (formatMode == 1 && channelSize > 0 && tileLen >= channelSize) {
            // NHWC: 对齐到 C 以保证通道组完整
            tileLen = AlignDown(tileLen, channelSize);
            if (tileLen <= 0) {
                // C 太大无法对齐，回退到基本对齐
                tileLen = AlignDown((ubSize - overhead) / static_cast<int64_t>(BYTES_PER_ELEM),
                                    static_cast<int64_t>(ALIGN_ELEM_FP32));
            }
        }
        OP_CHECK_IF(tileLen <= 0, OP_LOGE(context, "tileLen <= 0 after format align"), return ge::GRAPH_FAILED);

        // 多核切分：按 totalElements 平均分配
        // 限制核数不超过总元素数 / tileLen（避免核分到的数据太少）
        int64_t maxCores = CeilDiv(totalElements, tileLen);
        if (maxCores < coreNum) {
            usedCoreNum = maxCores;
        } else {
            usedCoreNum = coreNum;
        }
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }

        elemsPerCore = CeilDiv(totalElements, usedCoreNum);
        // 对齐 elemsPerCore：NHWC 需要对齐到 channelSize 以保证通道组完整
        // 其他格式对齐到 8 元素（32 字节）
        int64_t coreAlignment = static_cast<int64_t>(ALIGN_ELEM_FP32);
        if (formatMode == 1 && channelSize > coreAlignment) {
            // NHWC: 对齐到 channelSize（保证核边界在通道组边界）
            coreAlignment = channelSize;
        }
        elemsPerCore = AlignUp(elemsPerCore, coreAlignment);

        // 重新计算实际核数
        usedCoreNum = CeilDiv(totalElements, elemsPerCore);
        tailCoreElems = totalElements - (usedCoreNum - 1) * elemsPerCore;
        if (tailCoreElems <= 0) {
            tailCoreElems = elemsPerCore;
        }

        // 计算每核的 tile 数（基于 elemsPerCore）
        numTiles = CeilDiv(elemsPerCore, tileLen);
        lastTileLen = elemsPerCore - (numTiles - 1) * tileLen;
        if (lastTileLen <= 0) {
            lastTileLen = tileLen;
        }

        OP_LOGI(context, "BnInferGrad CONTIGUOUS: tileLen=%ld numTiles=%ld lastTileLen=%ld "
                "usedCoreNum=%ld elemsPerCore=%ld tailCoreElems=%ld",
                tileLen, numTiles, lastTileLen, usedCoreNum, elemsPerCore, tailCoreElems);

    } else {
        // ============================================================
        // NC1HWC0 分支：按 N*C1 任务切分到多核
        // ============================================================
        OP_CHECK_IF(C0 <= 0, OP_LOGE(context, "C0 <= 0"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(spatialSize <= 0, OP_LOGE(context, "spatialSize <= 0"), return ge::GRAPH_FAILED);

        alignedC0 = AlignUp(C0, static_cast<int64_t>(ALIGN_ELEM_FP32));

        // UB 切分：每次处理 tileHW 个空间位置的 C0 个通道
        // 每个元素占用：20 字节（fp32 和 fp16/bf16 路径均为 20）
        // overhead: invStdBuf(alignedC * 4) + scaleBuf(alignedC * 4) + varianceBuf(alignedC * 4)
        // 注意：invStdBuf 存储完整通道的 inv_std，大小为 alignedC（非 alignedC0）
        int64_t overheadNc1hwc0 = alignedC * 3 * static_cast<int64_t>(FLOAT_SIZE);
        int64_t hwc0TileLen = (ubSize - overheadNc1hwc0) / static_cast<int64_t>(BYTES_PER_ELEM);
        // hwc0TileLen = tileHW * C0, 所以 tileHW = hwc0TileLen / C0
        tileHW = hwc0TileLen / C0;
        // 对齐 tileHW 使得 tileHW * C0 对齐到 8 个元素
        // 由于 C0 通常是 16 或 32（已经是 8 的倍数），tileHW 无需额外对齐
        if (tileHW > spatialSize) {
            tileHW = spatialSize;
        }
        OP_CHECK_IF(tileHW <= 0, OP_LOGE(context, "tileHW <= 0, UB too small"), return ge::GRAPH_FAILED);

        // 确保 tileHW * C0 对齐到 8
        int64_t hwc0Tile = tileHW * C0;
        hwc0Tile = AlignDown(hwc0Tile, static_cast<int64_t>(ALIGN_ELEM_FP32));
        tileHW = hwc0Tile / C0;
        OP_CHECK_IF(tileHW <= 0, OP_LOGE(context, "tileHW <= 0 after align"), return ge::GRAPH_FAILED);

        tileLen = tileHW * C0;

        // 多核切分：总任务数 = N * C1
        totalTasks = N * C1;
        if (totalTasks < coreNum) {
            usedCoreNum = totalTasks;
        } else {
            usedCoreNum = coreNum;
        }
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }

        tasksPerCore = CeilDiv(totalTasks, usedCoreNum);
        usedCoreNum = CeilDiv(totalTasks, tasksPerCore);
        tailCoreTasks = totalTasks - (usedCoreNum - 1) * tasksPerCore;
        if (tailCoreTasks <= 0) {
            tailCoreTasks = tasksPerCore;
        }

        // 每个任务内的 tile 切分
        numTilesHW = CeilDiv(spatialSize, tileHW);
        lastTileHW = spatialSize - (numTilesHW - 1) * tileHW;
        if (lastTileHW <= 0) {
            lastTileHW = tileHW;
        }

        // 同时设置 CONTIGUOUS 风格的参数（保持兼容）
        elemsPerCore = tasksPerCore * spatialSize * C0;
        tailCoreElems = tailCoreTasks * spatialSize * C0;
        numTiles = numTilesHW;
        lastTileLen = lastTileHW * C0;

        OP_LOGI(context, "BnInferGrad NC1HWC0: tileHW=%ld numTilesHW=%ld lastTileHW=%ld "
                "totalTasks=%ld usedCoreNum=%ld tasksPerCore=%ld tailCoreTasks=%ld",
                tileHW, numTilesHW, lastTileHW, totalTasks, usedCoreNum, tasksPerCore, tailCoreTasks);
    }

    // ================================================================
    // Step 5: 填充 TilingData
    // ================================================================
    tiling->totalElements = totalElements;
    tiling->channelSize = channelSize;
    tiling->spatialSize = spatialSize;
    tiling->formatMode = formatMode;
    tiling->N = N;
    tiling->C1 = C1;
    tiling->C0 = C0;
    tiling->usedCoreNum = usedCoreNum;
    tiling->elemsPerCore = elemsPerCore;
    tiling->tailCoreElems = tailCoreElems;
    tiling->totalTasks = totalTasks;
    tiling->tasksPerCore = tasksPerCore;
    tiling->tailCoreTasks = tailCoreTasks;
    tiling->tileLen = tileLen;
    tiling->numTiles = numTiles;
    tiling->lastTileLen = lastTileLen;
    tiling->tileHW = tileHW;
    tiling->numTilesHW = numTilesHW;
    tiling->lastTileHW = lastTileHW;
    tiling->alignedC = alignedC;
    tiling->alignedC0 = alignedC0;

    // 将 epsilon float 位模式存入 int64_t（通过 memcpy 避免 strict aliasing 问题）
    int64_t epsBits = 0;
    float epsVal = epsilon;
    memcpy(&epsBits, &epsVal, sizeof(float));
    tiling->epsilonBits = epsBits;

    context->SetBlockDim(usedCoreNum);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, schMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBnInferGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BnInferGradCompileInfo {};

IMPL_OP_OPTILING(BnInferGrad)
    .Tiling(BnInferGradTilingFunc)
    .TilingParse<BnInferGradCompileInfo>(TilingParseForBnInferGrad);

} // namespace optiling
