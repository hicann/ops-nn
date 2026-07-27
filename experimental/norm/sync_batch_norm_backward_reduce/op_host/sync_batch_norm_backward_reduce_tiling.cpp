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
 * \file sync_batch_norm_backward_reduce_tiling.cpp
 * \brief SyncBatchNormBackwardReduce tiling: core split -> UB split -> tile split.
 *
 * element-wise op. schMode (tiling key) selects the compute dtype
 * (0 half, 1 float, 2 bfloat16). All dtypes are promoted to float on the UB
 * for computation, so每 tile 需要 3 个输入/输出队列 + 3 个 float 计算缓冲 + 1 个
 * mask 缓冲。The not-block-aligned remainder is absorbed by the last core.
 */

#include <string>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sync_batch_norm_backward_reduce_tiling_data.h"
#include "../op_kernel/sync_batch_norm_backward_reduce_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;

constexpr uint32_t BYTES_PER_BLOCK = 32u;
constexpr uint32_t BYTES_PER_CORE = 4096u;                            // minimal work granularity per core
constexpr uint32_t BLOCK_PER_CORE = BYTES_PER_CORE / BYTES_PER_BLOCK; // 128
constexpr uint32_t FOUR_BYTES = 4u;
constexpr uint32_t TWO_BYTES = 2u;

constexpr uint32_t INPUT_SUMDY_IDX = 0u;

// UB bytes consumed per processed element (single buffer accounted separately):
//   queues (4 inputs + 2 outputs) : 6 * dtypeSize * bufferNum
//   float compute buffers         : 4 * sizeof(float)  (sum_dy, sum_dy_dx_pad, mean, invert_std)
static uint64_t CalcTileLength(uint64_t ubSize, uint32_t dtypeSize, uint32_t bufferNum, uint64_t elemsPerBlock)
{
    constexpr uint64_t QUEUE_NUM = 6u;
    constexpr uint64_t FLOAT_BUF_BYTES = 4u * sizeof(float);
    // reserve a little UB for tiling struct / stack.
    uint64_t usable = (ubSize > 2048u) ? (ubSize - 2048u) : ubSize;
    uint64_t perElem = QUEUE_NUM * dtypeSize * bufferNum + FLOAT_BUF_BYTES;
    uint64_t maxElems = usable / perElem;
    uint64_t blockPerQue = FloorDiv(maxElems, elemsPerBlock);
    return (blockPerQue == 0u) ? 1u : blockPerQue;
}

// dtype of sum_dy selects the tiling key (schMode)。四个输入 dtype 由算子定义保证一致。
static ge::graphStatus SelectSchMode(gert::TilingContext* context, ge::DataType dtype, uint64_t& tilingKey,
                                     uint32_t& dtypeSize)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            dtypeSize = TWO_BYTES;
            tilingKey = GET_TPL_TILING_KEY(SYNCBNBR_TPL_SCH_MODE_0);
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT:
            dtypeSize = FOUR_BYTES;
            tilingKey = GET_TPL_TILING_KEY(SYNCBNBR_TPL_SCH_MODE_1);
            return ge::GRAPH_SUCCESS;
        case ge::DT_BF16:
            dtypeSize = TWO_BYTES;
            tilingKey = GET_TPL_TILING_KEY(SYNCBNBR_TPL_SCH_MODE_2);
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE(context, "unsupported dtype %d", static_cast<int32_t>(dtype));
            return ge::GRAPH_FAILED;
    }
}

// core split -> UB split -> tile split. The not-block-aligned remainder is absorbed by the last core.
static void FillSplitTiling(SyncBatchNormBackwardReduceTilingData* tiling, uint64_t aivCoreNum, uint64_t ubSize,
                            uint64_t totalLength, uint32_t dtypeSize, uint64_t elemsPerBlock)
{
    // core split.
    uint64_t totalBlocks = FloorDiv(totalLength, elemsPerBlock);
    uint64_t tailElems = totalLength % elemsPerBlock;
    uint64_t coreNum = aivCoreNum;
    if (totalBlocks < coreNum * BLOCK_PER_CORE) {
        coreNum = CeilDiv(totalBlocks, static_cast<uint64_t>(BLOCK_PER_CORE));
    }
    if (coreNum == 0u) {
        coreNum = 1u;
    }
    uint64_t blockPerCore = FloorDiv(totalBlocks, coreNum);
    uint64_t tailBlocks = totalBlocks % coreNum;

    // UB split (enable double buffering only when there is more than one tile).
    uint32_t bufferNum = 1u;
    uint64_t blockPerQue = CalcTileLength(ubSize, dtypeSize, bufferNum, elemsPerBlock);
    if (FloorDiv(blockPerCore, blockPerQue) > 1u) {
        bufferNum = 2u;
        blockPerQue = CalcTileLength(ubSize, dtypeSize, bufferNum, elemsPerBlock);
    }
    // CalcTileLength already guarantees blockPerQue >= 1; guard again so it can never be a zero divisor below.
    if (blockPerQue == 0u) {
        blockPerQue = 1u;
    }

    // tile split.
    uint64_t blockForLastCore = blockPerCore + tailBlocks;

    tiling->coreNum = coreNum;
    tiling->bufferNum = bufferNum;
    tiling->tailElems = tailElems;
    tiling->epochs = FloorDiv(blockPerCore, blockPerQue);
    tiling->epochsForLastCore = FloorDiv(blockForLastCore, blockPerQue);
    tiling->coreLength = blockPerCore * elemsPerBlock;
    tiling->tileLength = blockPerQue * elemsPerBlock;
    tiling->tailTileLength = (blockPerCore % blockPerQue) * elemsPerBlock;
    tiling->tailTileLengthForLastCore = (blockForLastCore % blockPerQue) * elemsPerBlock;
}

static ge::graphStatus SyncBatchNormBackwardReduceTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "SyncBatchNormBackwardReduce tiling starts.");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivCoreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0u;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    auto sumDyDesc = context->GetInputDesc(INPUT_SUMDY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumDyDesc);
    uint64_t tilingKey = 0u;
    uint32_t dtypeSize = 0u;
    if (SelectSchMode(context, sumDyDesc->GetDataType(), tilingKey, dtypeSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t elemsPerBlock = BYTES_PER_BLOCK / dtypeSize;

    // total element count (scalar shape -> 1).
    auto sumDyShape = context->GetInputShape(INPUT_SUMDY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumDyShape);
    uint64_t totalLength = static_cast<uint64_t>(sumDyShape->GetStorageShape().GetShapeSize());
    OP_CHECK_IF(totalLength == 0u, OP_LOGE(context, "input shape size must not be 0."), return ge::GRAPH_FAILED);

    SyncBatchNormBackwardReduceTilingData* tiling = context->GetTilingData<SyncBatchNormBackwardReduceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SyncBatchNormBackwardReduceTilingData), 0,
                         sizeof(SyncBatchNormBackwardReduceTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);
    FillSplitTiling(tiling, aivCoreNum, ubSize, totalLength, dtypeSize, elemsPerBlock);

    context->SetBlockDim(tiling->coreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0u;

    context->SetTilingKey(tilingKey);
    OP_LOGD(context,
            "SyncBatchNormBackwardReduce tiling: key=%lu coreNum=%lu bufferNum=%lu totalLen=%lu coreLength=%lu "
            "tileLength=%lu epochs=%lu "
            "epochsLast=%lu tailTile=%lu tailTileLast=%lu tailElems=%lu",
            tilingKey, tiling->coreNum, tiling->bufferNum, totalLength, tiling->coreLength, tiling->tileLength,
            tiling->epochs, tiling->epochsForLastCore, tiling->tailTileLength, tiling->tailTileLengthForLastCore,
            tiling->tailElems);
    return ge::GRAPH_SUCCESS;
}

struct SyncBatchNormBackwardReduceCompileInfo {};

static ge::graphStatus TilingParseForSyncBatchNormBackwardReduce([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SyncBatchNormBackwardReduce)
    .Tiling(SyncBatchNormBackwardReduceTilingFunc)
    .TilingParse<SyncBatchNormBackwardReduceCompileInfo>(TilingParseForSyncBatchNormBackwardReduce);

} // namespace optiling
