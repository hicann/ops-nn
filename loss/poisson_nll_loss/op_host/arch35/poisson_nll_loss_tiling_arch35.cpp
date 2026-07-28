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
 * \file poisson_nll_loss_tiling_arch35.cpp
 * \brief hand-written elementwise tiling (reduction=none stage-1). No atvoss reduce/elewise.
 */

#include <cstring>
#include <limits>
#include <string>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "poisson_nll_loss_tiling_arch35.h"
#include "poisson_nll_loss/op_kernel/arch35/poisson_nll_loss_tiling_def.h"
#include "poisson_nll_loss/op_kernel/arch35/poisson_nll_loss_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

namespace {
constexpr int64_t COMPUTE_TYPE_SIZE = 4; // fp32 compute
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr size_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL; // system reserved workspace
constexpr int64_t COMPARE_ALIGN_ELEMENTS = 256 / COMPUTE_TYPE_SIZE;
constexpr int64_t BUFFER_NUM_DB = 12; // double-buffer UB split count (x/t/y * 2 + work bufs)
constexpr int64_t BUFFER_NUM_SB = 9;  // single-buffer UB split count
// Attribute index in def.cpp: log_input(0), full(1), eps(2), reduction(3)
constexpr size_t ATTR_LOG_INPUT_IDX = 0;
constexpr size_t ATTR_FULL_IDX = 1;
constexpr size_t ATTR_EPS_IDX = 2;
constexpr size_t ATTR_REDUCTION_IDX = 3;
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_SUM = 1;
constexpr uint32_t REDUCTION_MEAN = 2;

const gert::Shape g_vec_1_shape = {1};

inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
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
} // namespace

ge::graphStatus PoissonNllLossTiling::RunTiling(const PoissonNllLossCompileInfo* compileInfo)
{
    (void)compileInfo;
    gert::TilingContext* context = tilingContext;

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    int64_t totalIdx = EnsureNotScalar(inputX->GetStorageShape()).GetShapeSize();
    // Empty tensor is supported, aligning with A2 (para_check.check_shape min_size=0 admits empty)
    // and torch: reduction=none -> empty output; sum -> 0; mean -> nan (0/0). See the empty branch
    // below (meanCof=inf makes the kernel's 0*meanCof produce nan for mean).

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    // Read attributes.
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* logInputPtr = attrs->GetAttrPointer<bool>(ATTR_LOG_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logInputPtr);
    const bool* fullPtr = attrs->GetAttrPointer<bool>(ATTR_FULL_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, fullPtr);
    const float* epsPtr = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsPtr);
    // eps must not be zero: it guards log(input+eps) on the log_input=false path.
    // Aligns with A2 TBE entry `if eps == 0: raise "Invalid eps which should not be zero."`
    // (canndev/ops/built-in/tbe/impl/poisson_nll_loss.py L148-149).
    OP_CHECK_IF(*epsPtr == 0.0f, OP_LOGE(context, "eps must not be zero"), return ge::GRAPH_FAILED);
    const char* reductionStr = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionStr);

    uint32_t reduction = REDUCTION_MEAN;
    if (strcmp(reductionStr, "none") == 0) {
        reduction = REDUCTION_NONE;
    } else if (strcmp(reductionStr, "sum") == 0) {
        reduction = REDUCTION_SUM;
    } else if (strcmp(reductionStr, "mean") == 0) {
        reduction = REDUCTION_MEAN;
    } else {
        OP_LOGE(context, "reduction must be none/sum/mean, got %s", reductionStr);
        return ge::GRAPH_FAILED;
    }

    PoissonNllLossTilingData* tiling = context->GetTilingData<PoissonNllLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(PoissonNllLossTilingData), 0, sizeof(PoissonNllLossTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->eps = *epsPtr;
    tiling->logInput = (*logInputPtr) ? 1U : 0U;
    tiling->full = (*fullPtr) ? 1U : 0U;
    tiling->reduction = reduction;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    int64_t usedCoreNum = 1;
    bool useDoubleBuffer = false;

    if (totalIdx > 0) {
        int64_t ubBlockSize = GetUbBlockSize(context);
        tiling->totalNum = totalIdx;
        tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
        usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);
        useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD);
        int64_t bufferNum = useDoubleBuffer ? BUFFER_NUM_DB : BUFFER_NUM_SB;
        int64_t alignUnit = (ubBlockSize > COMPARE_ALIGN_ELEMENTS) ? ubBlockSize : COMPARE_ALIGN_ELEMENTS;
        tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / COMPUTE_TYPE_SIZE, bufferNum), alignUnit);
        tiling->meanCof = 1.0f / static_cast<float>(totalIdx);
        // reduction=sum/mean stages one fp32 partial sum per core into workspace, each in its
        // own 32B(=8 fp32) block to avoid sub-block multi-core write races (WS_CORE_STRIDE=8).
        if (reduction != REDUCTION_NONE) {
            constexpr int64_t WS_CORE_STRIDE = 8;
            currentWorkspace[0] = static_cast<size_t>(usedCoreNum) * WS_CORE_STRIDE * sizeof(float) +
                                  SYS_WORKSPACE_SIZE;
        } else {
            currentWorkspace[0] = 0U;
        }
    } else {
        // Empty tensor (totalNum=0), block_dim stays 1. mean of empty = sum/N = 0/0 = nan;
        // meanCof=inf makes the kernel's total(0)*meanCof produce nan. sum of empty = 0 (meanCof
        // unused). reduction=sum/mean still stages one (zero) partial via workspace, so allocate
        // one core's 32B slot + sys workspace (mirrors the totalIdx>0 branch with usedCoreNum=1).
        tiling->meanCof = std::numeric_limits<float>::infinity();
        if (reduction != REDUCTION_NONE) {
            constexpr int64_t WS_CORE_STRIDE = 8;
            currentWorkspace[0] = WS_CORE_STRIDE * sizeof(float) + SYS_WORKSPACE_SIZE;
        } else {
            currentWorkspace[0] = 0U;
        }
    }

    context->SetBlockDim(usedCoreNum);
    // reduction=sum/mean does a two-phase cross-core reduction: each core writes its fp32 partial
    // sum to its own workspace slot, SyncAll, then block 0 sums the partials into the scalar output.
    // SyncAll requires all launched cores to be co-resident -> batch schedule mode.
    if (reduction != REDUCTION_NONE) {
        context->SetScheduleMode(1);
    }
    uint32_t doubleBufferKey = useDoubleBuffer ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), doubleBufferKey);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
