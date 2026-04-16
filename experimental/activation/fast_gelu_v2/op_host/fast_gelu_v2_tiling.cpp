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
 * \file fast_gelu_v2_tiling.cpp
 * \brief FastGeluV2 tiling implementation (arch35)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/fast_gelu_v2_tiling_data.h"
#include "../op_kernel/fast_gelu_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Double-buffer threshold: enable double buffer when total > 1024
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    OP_CHECK_IF(
        inputShapeX.GetShapeSize() != outShapeY.GetShapeSize(),
        OP_LOGE(context, "FastGeluV2: input and output shape size mismatch: x=%ld, y=%ld",
                inputShapeX.GetShapeSize(), outShapeY.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalIdx = inputShapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FastGeluV2TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape and attribute info
    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // Handle empty tensor: set safe tiling defaults so the kernel's Process()
    // computes loopCount = 0 and never enters the loop body.
    if (totalIdx == 0) {
        FastGeluV2TilingData* emptyTiling = context->GetTilingData<FastGeluV2TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, emptyTiling);
        OP_CHECK_IF(
            memset_s(emptyTiling, sizeof(FastGeluV2TilingData), 0, sizeof(FastGeluV2TilingData)) != EOK,
            OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
        emptyTiling->totalNum = 0;
        emptyTiling->blockFactor = 1;
        emptyTiling->ubFactor = 1;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // 4. Set tiling data
    FastGeluV2TilingData* tiling = context->GetTilingData<FastGeluV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(FastGeluV2TilingData), 0, sizeof(FastGeluV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Determine type size based on dtype
    int64_t typeSize = 4; // default float32
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        typeSize = 2;
    }

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);

    // Multi-core split: distribute elements evenly across available cores.
    // blockFactor is aligned to ubBlockSize to satisfy DMA alignment constraints.
    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    // UB split: calculate how many elements fit in one UB iteration.
    //
    // Buffer layout varies by dtype:
    //   FP16:  I/O buffers are FP16 (2B), compute buffers are FP32 (4B) for precision.
    //          Single-buffer: 1*2(in) + 1*2(out) + 3*4(tmp) + 2*4(cast) = 24 bytes/elem
    //          Double-buffer: 2*2(in) + 2*2(out) + 3*4(tmp) + 2*4(cast) = 28 bytes/elem
    //   BF16:  All buffers use bfloat16 (2B), no FP32 promotion.
    //          Single: 5 * 2 = 10 bytes/elem; Double: 7 * 2 = 14 bytes/elem
    //   FP32:  All buffers use float (4B).
    //          Single: 5 * 4 = 20 bytes/elem; Double: 7 * 4 = 28 bytes/elem
    //
    // Double-buffer is enabled when totalNum > MIN_SPLIT_THRESHOLD (1024) to allow
    // overlap of CopyIn/Compute/CopyOut via pipeline parallelism.
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;

    int64_t bytesPerElement;
    if (dataType == ge::DT_FLOAT16) {
        // FP16: I/O buffers are FP16 (2 bytes), compute buffers are FP32 (4 bytes)
        // Single: 1*2(in) + 1*2(out) + 3*4(tmp) + 2*4(cast) = 24
        // Double: 2*2(in) + 2*2(out) + 3*4(tmp) + 2*4(cast) = 28
        int64_t ioBufCount = useDoubleBuffer ? 4 : 2; // input + output, doubled if DB
        bytesPerElement = ioBufCount * 2 + 5 * 4; // 5 FP32 buffers (3 tmp + 2 cast)
    } else if (dataType == ge::DT_BF16) {
        // BF16: all buffers use bfloat16 (2 bytes), no cast to FP32
        // Single: 1 input + 3 tmp + 1 output = 5 buffers * 2 bytes = 10
        // Double: (1 input + 1 output)*2 + 3 tmp = 7 buffers * 2 bytes = 14
        int64_t bufferNum = useDoubleBuffer ? 7 : 5;
        bytesPerElement = bufferNum * typeSize;
    } else {
        // FP32: all buffers same size (4 bytes)
        // Single: 1 input + 3 tmp + 1 output = 5 buffers * 4 bytes = 20
        // Double: (1 input + 1 output)*2 + 3 tmp = 7 buffers * 4 bytes = 28
        int64_t bufferNum = useDoubleBuffer ? 7 : 5;
        bytesPerElement = bufferNum * typeSize;
    }
    tiling->ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize), bytesPerElement), ubBlockSize);

    // Safeguard: ubFactor must be at least 1 ubBlockSize to make progress
    if (tiling->ubFactor <= 0) {
        OP_LOGE(context, "FastGeluV2: ubFactor is 0; UB too small for even one block "
                "(ubSize=%lu, bytesPerElement=%ld)", ubSize, bytesPerElement);
        return ge::GRAPH_FAILED;
    }

    // Safeguard: blockFactor must be at least ubBlockSize
    if (tiling->blockFactor <= 0) {
        OP_LOGE(context, "FastGeluV2: blockFactor is 0; unexpected tiling calculation error");
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCoreNum);

    // 5. Set TilingKey
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, static_cast<uint32_t>(useDoubleBuffer));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFastGeluV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct FastGeluV2CompileInfo {};

IMPL_OP_OPTILING(FastGeluV2).Tiling(FastGeluV2TilingFunc).TilingParse<FastGeluV2CompileInfo>(TilingParseForFastGeluV2);

} // namespace optiling
