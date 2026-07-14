/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file layer_normalization_grad_tiling.cpp
 * @brief LayerNormalizationGrad 算子 Tiling 策略实现
 *
 * 成本模型驱动多核行分片，UB 容量不足时自动降级为列分块模式。
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/layer_normalization_grad_tiling_data.h"
#include "../op_kernel/layer_normalization_grad_tiling_key.h"

namespace optiling {

// ---- 成本模型常量 ----
const uint32_t BLOCK_SIZE = 32;
const uint32_t MAX_TILE_NUM = 256;
const uint32_t MIN_TILE_LENGTH = 16;
const uint32_t MAX_TILE_LENGTH = 4096;
const uint32_t VECTOR_OPS_PER_CYCLE = 256;
const uint32_t HBM_BW_BYTES_PER_CYCLE = 32;
const uint32_t GM_TRANSFER_GRANULARITY = 512;
const uint32_t MIN_TRANSFER_FOR_PEAK_BW = 16384;

// ---- wholeRow UB 估算常量 ----
const uint32_t REPEAT_BYTES = 256;           // Vector 每 repeat 处理字节数
const uint32_t LOW_PRECISION_BYTES = 2;      // half/bf16 字节数
const uint32_t FLOAT_BYTES = 4;              // float 计算精度字节数
const uint32_t WHOLE_ROW_QUEUE_COUNT = 3;    // 整行模式 TQue 数（dy/x/dx）
const uint32_t WHOLE_ROW_CALC_BUF_COUNT = 8; // 整行模式 calc buffer 上界估计
const uint32_t WHOLE_ROW_BUFFER_NUM = 2;     // 整行模式预检双缓冲数

struct RowTilingConfig {
    uint64_t rowsPerCore;
    uint64_t tailCoreRows;
    uint64_t usedCoreNum;
    uint32_t bufferNum;
    double cost;
};

static double evaluate_row_cost(uint64_t rows_per_core, uint64_t tail_rows, uint64_t core_num, uint32_t buf_num,
                                uint64_t D_padded, uint32_t dtype_size, uint64_t total_N, uint32_t num_queues)
{
    uint32_t tile_length = static_cast<uint32_t>(D_padded);
    uint32_t tile_num = static_cast<uint32_t>(rows_per_core);

    if (tile_num == 0 || tile_length == 0)
        return 1e30;

    uint32_t gran = GM_TRANSFER_GRANULARITY;
    uint32_t aligned_tl = ((tile_length * dtype_size + gran - 1) / gran) * gran;
    double align_ratio = (double)aligned_tl / std::max((uint32_t)(tile_length * dtype_size), 1u);

    uint32_t mte2_bytes_per_row = tile_length * dtype_size * 3;
    double mte2_time_per_row = (double)mte2_bytes_per_row * align_ratio / HBM_BW_BYTES_PER_CYCLE;
    double total_mte2_time = tile_num * mte2_time_per_row;

    double s_vector = (double)HBM_BW_BYTES_PER_CYCLE / (VECTOR_OPS_PER_CYCLE * dtype_size);
    double vector_time_per_row = tile_length * s_vector * 10.0;
    double total_vector_time = tile_num * vector_time_per_row;

    double total_cost;
    if (buf_num >= 2) {
        total_cost = std::max(total_mte2_time, total_vector_time);
    } else {
        total_cost = total_mte2_time + total_vector_time;
    }

    if (core_num > 1) {
        double avg_load = (double)total_N / core_num;
        double max_load = (double)(rows_per_core + (tail_rows > 0 ? 1 : 0));
        double load_imbalance = (max_load - avg_load) / std::max(avg_load, 1.0);
        double parallel_efficiency = 1.0 / (1.0 + load_imbalance);

        double comm_overhead = 0.05 * std::log2((double)core_num);
        parallel_efficiency *= (1.0 - comm_overhead);
        parallel_efficiency = std::max(parallel_efficiency, 0.4);

        total_cost = total_cost / (core_num * parallel_efficiency);

        double sync_overhead = core_num * tile_num * 10 * s_vector;
        total_cost += sync_overhead;
    }

    if (tile_length * dtype_size < MIN_TRANSFER_FOR_PEAK_BW) {
        total_cost *= 2.0;
    }
    if (buf_num == 1) {
        total_cost += 100.0;
    }

    return total_cost;
}

static RowTilingConfig solve_row_tiling_optimal(uint64_t N, uint64_t D_padded, uint32_t dtype_size, uint64_t ub_size,
                                                uint32_t max_cores, uint32_t num_queues)
{
    RowTilingConfig best;
    best.cost = 1e30;
    best.rowsPerCore = N;
    best.tailCoreRows = 0;
    best.usedCoreNum = 1;
    best.bufferNum = 2;

    uint32_t max_core_search = std::min(static_cast<uint32_t>(max_cores), static_cast<uint32_t>(N));
    for (uint32_t cn = 1; cn <= max_core_search; ++cn) {
        uint64_t rpc = N / cn;
        uint64_t tail = N % cn;

        for (uint32_t bn = 1; bn <= 2; ++bn) {
            uint64_t max_rows_in_core = rpc + (tail > 0 ? 1 : 0);
            if (max_rows_in_core == 0)
                max_rows_in_core = 1;

            uint32_t calc_dtype_size = (dtype_size == LOW_PRECISION_BYTES) ? FLOAT_BYTES : dtype_size;
            uint64_t queue_ub = static_cast<uint64_t>(WHOLE_ROW_QUEUE_COUNT) * bn * D_padded * dtype_size;
            uint64_t calc_ub = static_cast<uint64_t>(WHOLE_ROW_CALC_BUF_COUNT) * D_padded * calc_dtype_size;
            uint64_t elem_per_repeat = REPEAT_BYTES / calc_dtype_size;
            uint64_t elem_per_block = BLOCK_SIZE / calc_dtype_size;
            uint64_t first_repeats = (D_padded + elem_per_repeat - 1) / elem_per_repeat;
            uint64_t aligned_repeats = ((first_repeats + elem_per_block - 1) / elem_per_block) * elem_per_block;
            uint64_t reduce_ub = aligned_repeats * elem_per_block * calc_dtype_size;

            uint64_t total_ub = queue_ub + calc_ub + reduce_ub;
            if (total_ub > ub_size)
                continue;

            double cost = evaluate_row_cost(rpc, tail, cn, bn, D_padded, dtype_size, N, num_queues);

            if (cost < best.cost) {
                best.cost = cost;
                best.rowsPerCore = rpc;
                best.tailCoreRows = tail;
                best.usedCoreNum = cn;
                best.bufferNum = bn;
            }
        }
    }

    return best;
}

static RowTilingConfig make_balanced_row_tiling(uint64_t N, uint32_t max_cores, uint32_t buffer_num)
{
    RowTilingConfig config;
    config.usedCoreNum = std::min<uint64_t>(N, static_cast<uint64_t>(max_cores));
    if (config.usedCoreNum == 0) {
        config.usedCoreNum = 1;
    }
    config.rowsPerCore = N / config.usedCoreNum;
    config.tailCoreRows = N % config.usedCoreNum;
    config.bufferNum = buffer_num;
    config.cost = 0.0;
    return config;
}

static uint64_t CeilAlign(uint64_t value, uint64_t align) { return ((value + align - 1U) / align) * align; }

static uint64_t CalcReduceUb(uint64_t elemCount, uint64_t elemBytes)
{
    const uint64_t elemPerRepeat = REPEAT_BYTES / elemBytes;
    const uint64_t elemPerBlock = BLOCK_SIZE / elemBytes;
    const uint64_t firstRepeats = (elemCount + elemPerRepeat - 1u) / elemPerRepeat;
    const uint64_t alignedRepeats = ((firstRepeats + elemPerBlock - 1u) / elemPerBlock) * elemPerBlock;
    return alignedRepeats * elemPerBlock * elemBytes;
}

static uint64_t CalcWholeRowUb(uint64_t dPadded, uint32_t dtypeSize)
{
    const uint32_t calcDtypeSize = (dtypeSize == LOW_PRECISION_BYTES) ? FLOAT_BYTES : dtypeSize;
    const uint64_t queueUb = static_cast<uint64_t>(WHOLE_ROW_QUEUE_COUNT) * WHOLE_ROW_BUFFER_NUM * dPadded * dtypeSize;
    const uint64_t calcUb = static_cast<uint64_t>(WHOLE_ROW_CALC_BUF_COUNT) * dPadded * calcDtypeSize;
    const uint64_t reduceUb = CalcReduceUb(dPadded, calcDtypeSize);
    return queueUb + calcUb + reduceUb;
}

static uint64_t CalcColSplitUb(uint64_t tileColsAligned, uint32_t dtypeSize, bool needFloatConvert,
                               uint64_t maxCoreRows)
{
    const uint64_t accumBytes = sizeof(float);
    const uint64_t queueUb = 6ULL * tileColsAligned * dtypeSize;
    const uint64_t gammaUb = tileColsAligned * accumBytes;
    const uint64_t xhatUb = tileColsAligned * accumBytes;
    const uint64_t dxhatUb = tileColsAligned * accumBytes;
    const uint64_t tmpUb = tileColsAligned * accumBytes;
    const uint64_t tileGradUb = 2ULL * tileColsAligned * accumBytes;
    const uint64_t rowScalarUb = 2ULL * maxCoreRows * accumBytes;
    const uint64_t reduceUb = CalcReduceUb(tileColsAligned, accumBytes);
    const uint64_t castScratchUb = needFloatConvert ? 2ULL * tileColsAligned * dtypeSize : 0ULL;
    return queueUb + gammaUb + xhatUb + dxhatUb + tmpUb + tileGradUb + rowScalarUb + reduceUb + castScratchUb;
}

static uint64_t SearchTileCols(uint64_t D, uint32_t dtypeSize, bool needFloatConvert, uint64_t ubLength,
                               uint64_t maxCoreRows)
{
    constexpr uint64_t kColSplitUbSafetyBytes = 2048;
    if (ubLength <= kColSplitUbSafetyBytes) {
        return 0;
    }
    const uint64_t effectiveUbLength = ubLength - kColSplitUbSafetyBytes;
    const uint64_t alignElems = 32u / dtypeSize;
    uint64_t lo = std::min(alignElems, D);
    uint64_t hi = D;
    uint64_t best = 0;
    while (lo <= hi) {
        uint64_t mid = lo + (hi - lo) / 2u;
        mid = (mid / alignElems) * alignElems;
        if (mid == 0) {
            mid = alignElems;
        }
        if (mid > D) {
            mid = (D / alignElems) * alignElems;
        }
        if (mid == 0) {
            break;
        }
        const uint64_t alignedMid = CeilAlign(mid, alignElems);
        if (CalcColSplitUb(alignedMid, dtypeSize, needFloatConvert, maxCoreRows) <= effectiveUbLength) {
            best = mid;
            lo = mid + alignElems;
        } else {
            if (mid <= alignElems) {
                break;
            }
            hi = mid - alignElems;
        }
    }
    return best;
}

struct LayerNormalizationGradCompileInfo {};

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

static ge::graphStatus LayerNormalizationGradTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台运行时信息
    uint64_t ubLength = 0;
    int64_t maxCoreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubLength, maxCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 获取输入形状 (dy: [N, D])
    auto dyShape = context->GetInputShape(0)->GetStorageShape();
    if (dyShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    uint64_t N = static_cast<uint64_t>(dyShape.GetDim(0));
    uint64_t D = static_cast<uint64_t>(dyShape.GetDim(1));

    if (N == 0 || D == 0) {
        return ge::GRAPH_FAILED;
    }

    // 3. 获取数据类型
    ge::DataType inputDataType = context->GetInputDesc(0)->GetDataType();
    // 设置 tiling key
    // 数据类型合法性校验，避免非法 dtype 触发后续异常。
    if (inputDataType != ge::DT_FLOAT && inputDataType != ge::DT_FLOAT16 && inputDataType != ge::DT_BF16) {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }

    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(inputDataType, dataTypeLength);

    uint64_t alignElems = 32u / dataTypeLength;
    uint64_t DPadded = ((D + alignElems - 1u) / alignElems) * alignElems;
    bool needFloatConvert = (dataTypeLength == 2);

    // 4. 基础 UB 容量预检（整行模式）
    const uint64_t wholeRowRequiredUb = CalcWholeRowUb(DPadded, dataTypeLength);
    {
        const uint32_t calc_dtype_size = (dataTypeLength == LOW_PRECISION_BYTES) ? FLOAT_BYTES : dataTypeLength;
        const uint64_t min_queue_ub = static_cast<uint64_t>(WHOLE_ROW_QUEUE_COUNT) * WHOLE_ROW_BUFFER_NUM * DPadded *
                                      dataTypeLength;
        const uint64_t min_calc_ub = static_cast<uint64_t>(WHOLE_ROW_CALC_BUF_COUNT) * DPadded * calc_dtype_size;
        const uint64_t min_reduce_ub = CalcReduceUb(DPadded, calc_dtype_size);
        OP_LOGD(context, "[LAYER_NORM_GRAD_TILING] ubLength=%lu wholeRowRequiredUB=%lu (queue=%lu calc=%lu reduce=%lu)",
                ubLength, wholeRowRequiredUb, min_queue_ub, min_calc_ub, min_reduce_ub);
    }

    // 5. 成本模型驱动的多核行分片
    uint32_t num_queues = needFloatConvert ? 11 : 8;

    RowTilingConfig optimal = solve_row_tiling_optimal(N, DPadded, dataTypeLength, ubLength,
                                                       static_cast<uint32_t>(maxCoreNum), num_queues);

    if (optimal.cost >= 1e29 || optimal.usedCoreNum == 0) {
        optimal.rowsPerCore = N;
        optimal.tailCoreRows = 0;
        optimal.usedCoreNum = 1;
        optimal.bufferNum = 1;
        optimal.cost = evaluate_row_cost(N, 0, 1, 1, DPadded, dataTypeLength, N, num_queues);
    }

    uint64_t usedCoreNum = optimal.usedCoreNum;
    uint64_t rowsPerCore = optimal.rowsPerCore;
    uint64_t tailCoreRows = optimal.tailCoreRows;
    uint32_t BUFFER_NUM = optimal.bufferNum;
    uint64_t maxCoreRows = rowsPerCore + (tailCoreRows > 0 ? 1ULL : 0ULL);
    if (maxCoreRows == 0) {
        maxCoreRows = 1;
    }

    uint32_t colSplitMode = 0;
    uint64_t tileCols = D;
    uint64_t tileColsAligned = DPadded;

    if (wholeRowRequiredUb > ubLength) {
        colSplitMode = 1;
        RowTilingConfig colSplitConfig = make_balanced_row_tiling(N, static_cast<uint32_t>(maxCoreNum), 1);
        usedCoreNum = colSplitConfig.usedCoreNum;
        rowsPerCore = colSplitConfig.rowsPerCore;
        tailCoreRows = colSplitConfig.tailCoreRows;
        BUFFER_NUM = colSplitConfig.bufferNum;
        maxCoreRows = rowsPerCore + (tailCoreRows > 0 ? 1ULL : 0ULL);
        if (maxCoreRows == 0) {
            maxCoreRows = 1;
        }
        tileCols = SearchTileCols(D, dataTypeLength, needFloatConvert, ubLength, maxCoreRows);
        if (tileCols == 0) {
            OP_LOGE(context,
                    "[LAYER_NORM_GRAD_TILING] no valid tileCols for ubLength=%lu D=%lu dtypeBytes=%u maxCoreRows=%lu",
                    ubLength, D, dataTypeLength, maxCoreRows);
            return ge::GRAPH_FAILED;
        }
        tileColsAligned = CeilAlign(tileCols, alignElems);
    }

    uint64_t numColTiles = (D + tileCols - 1u) / tileCols;
    uint64_t workspaceTileStride = CeilAlign(tileColsAligned, static_cast<uint64_t>(64u / sizeof(float)));

    // 6. 填写 tiling 数据
    LayerNormalizationGradTilingData* tiling = context->GetTilingData<LayerNormalizationGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LayerNormalizationGradTilingData), 0, sizeof(LayerNormalizationGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->N = N;
    tiling->D = D;
    tiling->DPadded = DPadded;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tailCoreRows = tailCoreRows;
    tiling->usedCoreNum = usedCoreNum;
    tiling->bufferNum = BUFFER_NUM;
    tiling->needFloatConvert = needFloatConvert ? 1 : 0;
    tiling->colSplitMode = colSplitMode;
    tiling->tileCols = tileCols;
    tiling->tileColsAligned = tileColsAligned;
    tiling->numColTiles = numColTiles;
    tiling->workspaceTileStride = workspaceTileStride;
    tiling->maxCoreRows = maxCoreRows;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(GET_TPL_TILING_KEY(LAYER_NORMALIZATION_GRAD_TPL_SCH_MODE_0));

    // 7. workspace 分配
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto sysWsSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (usedCoreNum > 1) {
        if (colSplitMode == 0) {
            uint64_t wsOffset = CeilAlign(DPadded, static_cast<uint64_t>(64 / sizeof(float)));
            currentWorkspace[0] = static_cast<size_t>(usedCoreNum) * 2ULL * wsOffset * sizeof(float) + sysWsSize;
        } else {
            currentWorkspace[0] = static_cast<size_t>(usedCoreNum) * numColTiles * 2ULL * workspaceTileStride *
                                      sizeof(float) +
                                  sysWsSize;
        }
    } else {
        currentWorkspace[0] = sysWsSize;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLayerNormalizationGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LayerNormalizationGrad)
    .Tiling(LayerNormalizationGradTilingFunc)
    .TilingParse<LayerNormalizationGradCompileInfo>(TilingParseForLayerNormalizationGrad);
} // namespace optiling
