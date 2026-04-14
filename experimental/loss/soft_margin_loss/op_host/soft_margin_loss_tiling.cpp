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


/**
 * \file soft_margin_loss_tiling.cpp
 * \brief SoftMarginLoss Tiling implementation
 *
 * Computes tiling parameters for SoftMarginLoss operator:
 * - Multi-core splitting: totalNum divided evenly across AI Cores
 * - UB splitting: each core processes data in chunks of ubFactor elements
 * - TilingKey selection: based on input dtype and reduction mode
 *
 * Iteration 1: Implements float32 + none path, pre-embeds skeleton for other paths.
 */

#include <cstring>
#include <cmath>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/soft_margin_loss_tiling_data.h"
#include "../op_kernel/soft_margin_loss_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

// float32 + none: selfQueue(x2) + targetQueue(x2) + outputQueue(x2) + tmpBuf1(x1) + tmpBuf2(x1) = 8 float buffers
constexpr int64_t BUFFER_NUM_FP32_NONE = 8;
// float32 + reduce: selfQueue(x2) + targetQueue(x2) + tmpBuf1(x1) + tmpBuf2(x1) + tmpBuf3(x1) + reduceTmpBuf(x1) + partialSumBuf(x1) = 9
constexpr int64_t BUFFER_NUM_FP32_REDUCE = 9;
// float16 + none: selfQueue(x2,half=1eq) + targetQueue(x2,half=1eq) + outputQueue(x2,half=1eq) + tmpBuf1(float=2eq) + tmpBuf2(float=2eq) + tmpBuf3(float=2eq) = 6 float-equiv
constexpr int64_t BUFFER_NUM_FP16_NONE = 6;
// float16 + reduce: similar with reduce buffers = 7
constexpr int64_t BUFFER_NUM_FP16_REDUCE = 7;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, size_t usrWorkspaceSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SoftMarginLossTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get input shape and dtype
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = storageShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();

    // 3. Get reduction attribute
    // NOTE: The aclnn runtime stores Int attrs as strings ("none"/"mean"/"sum") in TilingContext,
    // while the UT framework stores them as raw int64. We try int64 first, and if the value is
    // out of valid range [0,2], fall back to reading as string.
    const auto* attrs = context->GetAttrs();
    int64_t reduction = -1;  // -1 means not yet determined
    if (attrs != nullptr) {
        const int64_t* intPtr = attrs->GetInt(0);
        if (intPtr != nullptr && *intPtr >= 0 && *intPtr <= 2) {
            reduction = *intPtr;
            OP_LOGI(context, "SoftMarginLoss: got reduction=%ld from int attr", reduction);
        } else {
            // Runtime may store as string: try string parsing
            const char* reductionStr = attrs->GetStr(0);
            if (reductionStr != nullptr) {
                if (strcmp(reductionStr, "none") == 0) {
                    reduction = 0;
                } else if (strcmp(reductionStr, "mean") == 0) {
                    reduction = 1;
                } else if (strcmp(reductionStr, "sum") == 0) {
                    reduction = 2;
                } else {
                    OP_LOGE(context, "SoftMarginLoss: invalid reduction string '%s', expected 0/1/2 or none/mean/sum", reductionStr);
                    return ge::GRAPH_FAILED;
                }
                OP_LOGI(context, "SoftMarginLoss: got reduction='%s' -> %ld", reductionStr, reduction);
            } else if (intPtr == nullptr || *intPtr < 0 || *intPtr > 2) {
                // Neither valid int64 nor valid string found
                OP_LOGE(context, "SoftMarginLoss: could not read valid reduction attr");
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        OP_LOGE(context, "SoftMarginLoss: context->GetAttrs() returned nullptr");
        return ge::GRAPH_FAILED;
    }

    // 4. Set TilingData
    SoftMarginLossTilingData* tiling = context->GetTilingData<SoftMarginLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftMarginLossTilingData), 0, sizeof(SoftMarginLossTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    // Handle empty tensor
    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        tiling->reductionMode = static_cast<int32_t>(reduction);
        tiling->invNumel = (reduction == 1) ? std::nanf("") : 0.0f;
        tiling->usedCoreNum = 1;
        context->SetBlockDim(1);
        // Select TilingKey based on dtype and reduction mode
        if (dtype == ge::DT_FLOAT) {
            context->SetTilingKey(GET_TPL_TILING_KEY(reduction == 0 ? SML_TPL_SCH_MODE_0 : SML_TPL_SCH_MODE_1));
        } else {
            context->SetTilingKey(GET_TPL_TILING_KEY(reduction == 0 ? SML_TPL_SCH_MODE_2 : SML_TPL_SCH_MODE_3));
        }
        OP_CHECK_IF(
            GetWorkspaceSize(context, 0) != ge::GRAPH_SUCCESS,
            OP_LOGE(context, "GetWorkspaceSize error"),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // 5. Multi-core splitting
    tiling->totalNum = totalNum;
    bool isReduce = (reduction != 0);

    // For reduce path: use single core to avoid cross-core SyncAll coordination.
    // The reduce path produces a single scalar output; all elements are processed
    // in tiles on core 0. This is correct and sufficient for iteration 2.
    // Multi-core reduce can be added as a performance optimization in a later iteration.
    int64_t usedCoreNum;
    if (isReduce) {
        tiling->blockFactor = totalNum;
        usedCoreNum = 1;
    } else {
        tiling->blockFactor = CeilDiv(totalNum, coreNum);
        usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);
    }
    tiling->usedCoreNum = usedCoreNum;

    // 6. Reduction mode
    tiling->reductionMode = static_cast<int32_t>(reduction);
    tiling->invNumel = (reduction == 1) ? (1.0f / static_cast<float>(totalNum)) : 0.0f;

    // 7. UB splitting and TilingKey selection
    int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t ubBlockSize = GetUbBlockSize(context);
    // Both paths compute internally in float32, so typeSize = 4
    constexpr int64_t typeSize = 4;
    size_t usrWorkspaceSize = 0;

    OP_LOGI(context, "SoftMarginLoss: dtype=%d, reduction=%ld, isReduce=%s, totalNum=%ld, coreNum=%ld, usedCoreNum=%ld",
            dtype, reduction, isReduce ? "true" : "false", totalNum, coreNum, usedCoreNum);

    if (dtype == ge::DT_FLOAT) {
        if (!isReduce) {
            tiling->ubFactor = FloorAlign(FloorDiv((ubCanUse / typeSize), BUFFER_NUM_FP32_NONE), ubBlockSize);
            auto tilingKey = GET_TPL_TILING_KEY(SML_TPL_SCH_MODE_0);
            OP_LOGI(context, "SoftMarginLoss: setting tiling key for FP32_NONE, key=%lu", tilingKey);
            context->SetTilingKey(tilingKey);
        } else {
            tiling->ubFactor = FloorAlign(FloorDiv((ubCanUse / typeSize), BUFFER_NUM_FP32_REDUCE), ubBlockSize);
            auto tilingKey = GET_TPL_TILING_KEY(SML_TPL_SCH_MODE_1);
            OP_LOGI(context, "SoftMarginLoss: setting tiling key for FP32_REDUCE, key=%lu", tilingKey);
            context->SetTilingKey(tilingKey);
            usrWorkspaceSize = static_cast<size_t>(usedCoreNum) * 32;  // 32-byte aligned per core
        }
    } else if (dtype == ge::DT_FLOAT16) {
        if (!isReduce) {
            tiling->ubFactor = FloorAlign(FloorDiv((ubCanUse / typeSize), BUFFER_NUM_FP16_NONE), ubBlockSize);
            context->SetTilingKey(GET_TPL_TILING_KEY(SML_TPL_SCH_MODE_2));
        } else {
            tiling->ubFactor = FloorAlign(FloorDiv((ubCanUse / typeSize), BUFFER_NUM_FP16_REDUCE), ubBlockSize);
            context->SetTilingKey(GET_TPL_TILING_KEY(SML_TPL_SCH_MODE_3));
            usrWorkspaceSize = static_cast<size_t>(usedCoreNum) * 32;
        }
    } else {
        OP_LOGE(context, "SoftMarginLoss: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    // 8. Set workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context, usrWorkspaceSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSoftMarginLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SoftMarginLossCompileInfo {};

IMPL_OP_OPTILING(SoftMarginLoss).Tiling(SoftMarginLossTilingFunc).TilingParse<SoftMarginLossCompileInfo>(TilingParseForSoftMarginLoss);

} // namespace optiling
