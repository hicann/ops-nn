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
constexpr int64_t MERGE_VL_FP32 = 64;                         // 跨核合并的矢量车道数(fp32)
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

const gert::Shape g_pnll_vec1_shape = {1};

inline const gert::Shape PnllEnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_pnll_vec1_shape;
    }
    return inShape;
}

ge::graphStatus GetPnllPlatformInfo(gert::TilingContext* context, uint64_t& pnllUbSize, int64_t& pnllCoreNum)
{
    fe::PlatFormInfos* pnllPlatformPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, pnllPlatformPtr);
    auto pnllPlatform = platform_ascendc::PlatformAscendC(pnllPlatformPtr);
    pnllCoreNum = pnllPlatform.GetCoreNumAiv();
    OP_CHECK_IF(pnllCoreNum == 0, OP_LOGE(context, "pnllCoreNum is 0"), return ge::GRAPH_FAILED);
    pnllPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, pnllUbSize);
    OP_CHECK_IF(pnllUbSize == 0, OP_LOGE(context, "pnllUbSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 读取并校验属性: log_input / full / eps / reduction。
ge::graphStatus ParseAttrs(gert::TilingContext* context, uint32_t& reduction, float& eps, uint32_t& logInput,
                           uint32_t& full)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* logInputPtr = attrs->GetAttrPointer<bool>(ATTR_LOG_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logInputPtr);
    const bool* fullPtr = attrs->GetAttrPointer<bool>(ATTR_FULL_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, fullPtr);
    const float* epsPtr = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsPtr);
    // eps must not be zero: it guards log(input+eps) on the log_input=false path.
    // Aligns with ascend910b TBE entry `if eps == 0: raise "Invalid eps which should not be zero."`
    // (canndev/ops/built-in/tbe/impl/poisson_nll_loss.py L148-149).
    OP_CHECK_IF(*epsPtr == 0.0f, OP_LOGE(context, "eps must not be zero"), return ge::GRAPH_FAILED);
    const char* pnllReductionStr = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pnllReductionStr);

    if (strcmp(pnllReductionStr, "none") == 0) {
        reduction = REDUCTION_NONE;
    } else if (strcmp(pnllReductionStr, "sum") == 0) {
        reduction = REDUCTION_SUM;
    } else if (strcmp(pnllReductionStr, "mean") == 0) {
        reduction = REDUCTION_MEAN;
    } else {
        OP_LOGE(context, "reduction must be none/sum/mean, got %s", pnllReductionStr);
        return ge::GRAPH_FAILED;
    }
    eps = *epsPtr;
    logInput = (*logInputPtr) ? 1U : 0U;
    full = (*fullPtr) ? 1U : 0U;
    return ge::GRAPH_SUCCESS;
}

// 切分与 workspace: 按真实核数算 blockFactor/ubFactor/partialUbElems 与 workspace 大小。
// reduction=sum/mean 每核一个 32B(=8 fp32)独占槽, 避免子块粒度的多核写竞争。
ge::graphStatus FillSplitAndWorkspace(gert::TilingContext* context, uint64_t pnllUbSize, int64_t pnllCoreNum,
                                      int64_t totalIdx, uint32_t reduction, PoissonNllLossTilingData* tiling,
                                      int64_t& usedCoreNum, bool& useDoubleBuffer)
{
    constexpr int64_t WS_CORE_STRIDE = 8;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0U;

    if (totalIdx == 0) {
        // Empty tensor (totalNum=0), block_dim stays 1. mean of empty = sum/N = 0/0 = nan;
        // meanCof=inf makes the kernel's total(0)*meanCof produce nan. sum of empty = 0 (meanCof
        // unused). reduction=sum/mean still stages one (zero) partial via workspace, so allocate
        // one core's 32B slot + sys workspace (mirrors the totalIdx>0 branch with usedCoreNum=1).
        tiling->meanCof = std::numeric_limits<float>::infinity();
        if (reduction != REDUCTION_NONE) {
            currentWorkspace[0] = WS_CORE_STRIDE * sizeof(float) + SYS_WORKSPACE_SIZE;
            tiling->partialUbElems = static_cast<uint32_t>(MERGE_VL_FP32);
        }
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = GetUbBlockSize(context);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, pnllCoreNum), ubBlockSize);
    usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);
    useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD);
    int64_t bufferNum = useDoubleBuffer ? BUFFER_NUM_DB : BUFFER_NUM_SB;
    int64_t alignUnit = (ubBlockSize > COMPARE_ALIGN_ELEMENTS) ? ubBlockSize : COMPARE_ALIGN_ELEMENTS;
    tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(pnllUbSize) / COMPUTE_TYPE_SIZE, bufferNum), alignUnit);
    tiling->meanCof = 1.0f / static_cast<float>(totalIdx);
    if (reduction != REDUCTION_NONE) {
        currentWorkspace[0] = static_cast<size_t>(usedCoreNum) * WS_CORE_STRIDE * sizeof(float) + SYS_WORKSPACE_SIZE;
        // 跨核合并整轮读 MERGE_VL_FP32 车道, UB 用量按整轮向上取整, 由 host 下发。
        tiling->partialUbElems = static_cast<uint32_t>(CeilAlign(usedCoreNum * WS_CORE_STRIDE, MERGE_VL_FP32));
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus PoissonNllLossTiling::RunTiling(const PoissonNllLossCompileInfo* compileInfo)
{
    (void)compileInfo;
    gert::TilingContext* context = tilingContext;

    uint64_t pnllUbSize = 0;
    int64_t pnllCoreNum = 0;
    OP_CHECK_IF(GetPnllPlatformInfo(context, pnllUbSize, pnllCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPnllPlatformInfo error"), return ge::GRAPH_FAILED);

    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    int64_t totalIdx = PnllEnsureNotScalar(inputX->GetStorageShape()).GetShapeSize();
    // Empty tensor is supported, aligning with ascend910b (para_check.check_shape min_size=0 admits empty)
    // and torch: reduction=none -> empty output; sum -> 0; mean -> nan (0/0). See FillSplitAndWorkspace
    // (meanCof=inf makes the kernel's 0*meanCof produce nan for mean).

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    uint32_t reduction = REDUCTION_MEAN;
    float eps = 0.0f;
    uint32_t logInput = 0U;
    uint32_t full = 0U;
    OP_CHECK_IF(ParseAttrs(context, reduction, eps, logInput, full) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ParseAttrs error"), return ge::GRAPH_FAILED);

    PoissonNllLossTilingData* tiling = context->GetTilingData<PoissonNllLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(PoissonNllLossTilingData), 0, sizeof(PoissonNllLossTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->eps = eps;
    tiling->logInput = logInput;
    tiling->full = full;
    tiling->reduction = reduction;

    int64_t usedCoreNum = 1;
    bool useDoubleBuffer = false;
    OP_CHECK_IF(FillSplitAndWorkspace(context, pnllUbSize, pnllCoreNum, totalIdx, reduction, tiling, usedCoreNum,
                                      useDoubleBuffer) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "FillSplitAndWorkspace error"), return ge::GRAPH_FAILED);

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
