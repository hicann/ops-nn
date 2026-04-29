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
 * \file apply_adamax_tiling.cpp
 * \brief ApplyAdamax tiling implementation (arch35 / Ascend950)
 *
 * TilingKey selection (D_T x IS_ALIGN):
 *   TilingKey_0: D_T=C_DT_FLOAT,   K_ALIGN=0  (FP32, unaligned)
 *   TilingKey_1: D_T=C_DT_FLOAT,   K_ALIGN=1  (FP32, aligned)
 *   TilingKey_2: D_T=C_DT_FLOAT16, K_ALIGN=0  (FP16, unaligned)
 *   TilingKey_3: D_T=C_DT_FLOAT16, K_ALIGN=1  (FP16, aligned)
 *
 * Scalar parameters (beta1Power, lr, beta1, beta2, epsilon) are passed via
 * TilingData (read from Attr).  The ACLNN layer reads aclScalar values and
 * passes them as Attrs in registration order: beta1Power=0, lr=1, beta1=2,
 * beta2=3, epsilon=4.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/apply_adamax_tiling_data.h"
#include "../op_kernel/apply_adamax_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

// System-reserved workspace (this op uses no extra workspace, set 0).
constexpr uint32_t WS_SYS_SIZE = 0U;

// Input index (consistent with op_def)
static constexpr size_t IDX_VAR = 0;

// Attr indices (matching op_def Attr registration order)
static constexpr size_t ATTR_BETA1_POWER = 0;
static constexpr size_t ATTR_LR = 1;
static constexpr size_t ATTR_BETA1 = 2;
static constexpr size_t ATTR_BETA2 = 3;
static constexpr size_t ATTR_EPSILON = 4;

static const gert::Shape K_VEC_1_SHAPE = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return K_VEC_1_SHAPE;
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

// Get var element count + dtype
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto varDesc = context->GetInputDesc(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    dataType = varDesc->GetDataType();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    OP_CHECK_IF(
        supportedDtype.count(dataType) == 0,
        OP_LOGE(context, "ApplyAdamax: unsupported dtype %d", static_cast<int>(dataType)),
        return ge::GRAPH_FAILED);

    auto varShapePtr = context->GetInputShape(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
    auto varShape = EnsureNotScalar(varShapePtr->GetStorageShape());
    totalIdx = varShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

// Read 5 scalar Attrs (beta1Power, lr, beta1, beta2, epsilon) into TilingData.
// Pre-compute oneMinusBeta1 and lrAdj = lr / (1 - beta1Power) for Kernel.
static ge::graphStatus ParseScalarAttrs(gert::TilingContext* context, ApplyAdamaxTilingData* tiling)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const float* beta1PowerPtr = attrs->GetAttrPointer<float>(ATTR_BETA1_POWER);
    OP_CHECK_NULL_WITH_CONTEXT(context, beta1PowerPtr);
    const float* lrPtr = attrs->GetAttrPointer<float>(ATTR_LR);
    OP_CHECK_NULL_WITH_CONTEXT(context, lrPtr);
    const float* beta1Ptr = attrs->GetAttrPointer<float>(ATTR_BETA1);
    OP_CHECK_NULL_WITH_CONTEXT(context, beta1Ptr);
    const float* beta2Ptr = attrs->GetAttrPointer<float>(ATTR_BETA2);
    OP_CHECK_NULL_WITH_CONTEXT(context, beta2Ptr);
    const float* epsPtr = attrs->GetAttrPointer<float>(ATTR_EPSILON);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsPtr);

    tiling->beta1Power = *beta1PowerPtr;
    tiling->lr = *lrPtr;
    tiling->beta1 = *beta1Ptr;
    tiling->beta2 = *beta2Ptr;
    tiling->epsilon = *epsPtr;
    tiling->oneMinusBeta1 = 1.0f - tiling->beta1;

    // Pre-compute lrAdj = lr / (1 - beta1Power) with zero-denominator guard.
    constexpr float SCALAR_EPS = 1e-30f;
    float denomPower = 1.0f - tiling->beta1Power;
    if (denomPower < SCALAR_EPS && denomPower > -SCALAR_EPS) {
        denomPower = (denomPower >= 0.0f) ? SCALAR_EPS : -SCALAR_EPS;
    }
    tiling->lrAdj = tiling->lr / denomPower;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// Initialize TilingData buffer (zero-fill).
static ge::graphStatus InitTilingData(gert::TilingContext* context, ApplyAdamaxTilingData*& tiling)
{
    tiling = context->GetTilingData<ApplyAdamaxTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ApplyAdamaxTilingData), 0, sizeof(ApplyAdamaxTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Compute multi-core split: blockFactor + usedCoreNum, write into tiling.
static ge::graphStatus ComputeMultiCoreSplit(gert::TilingContext* context, ge::DataType dataType,
    int64_t totalIdx, int64_t coreNum, ApplyAdamaxTilingData* tiling, int64_t& alignElem, int64_t& usedCoreNum)
{
    constexpr int64_t TYPE_SIZE_FP32 = 4;
    constexpr int64_t TYPE_SIZE_FP16 = 2;
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);  // 32B
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? TYPE_SIZE_FP32 : TYPE_SIZE_FP16;
    tiling->totalNum = totalIdx;
    if (totalIdx <= 0) {
        OP_LOGE(context, "ApplyAdamax: totalIdx must > 0, got %ld", totalIdx);
        return ge::GRAPH_FAILED;
    }
    alignElem = ubBlockSize / typeSize;          // fp32: 8, fp16: 16
    if (totalIdx < alignElem) {
        tiling->blockFactor = totalIdx;
        usedCoreNum = 1;
    } else {
        int64_t perCoreRaw = Ops::Base::CeilDiv(totalIdx, coreNum);
        tiling->blockFactor = Ops::Base::CeilAlign(perCoreRaw, alignElem);
        usedCoreNum = Ops::Base::CeilDiv(totalIdx, tiling->blockFactor);
    }
    OP_CHECK_IF(usedCoreNum == 0, OP_LOGE(context, "usedCoreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Compute UB split: ubFactor, write into tiling.
static ge::graphStatus ComputeUbSplit(gert::TilingContext* context, ge::DataType dataType,
    uint64_t ubSize, int64_t alignElem, ApplyAdamaxTilingData* tiling)
{
    constexpr int64_t RESERVED_UB = 48 * 1024;  // 48 KB reserved
    constexpr int64_t BYTE_PER_ELEM_FP32 = 64;
    constexpr int64_t BYTE_PER_ELEM_FP16 = 52;
    int64_t availableUb = static_cast<int64_t>(ubSize) - RESERVED_UB;
    OP_CHECK_IF(availableUb <= 0, OP_LOGE(context, "availableUb<=0"), return ge::GRAPH_FAILED);
    int64_t bytePerElem = (dataType == ge::DT_FLOAT) ? BYTE_PER_ELEM_FP32 : BYTE_PER_ELEM_FP16;
    int64_t tileElem = availableUb / bytePerElem;
    tileElem = Ops::Base::FloorAlign(tileElem, static_cast<int64_t>(256));
    if (tileElem < alignElem) {
        tileElem = alignElem;
    }
    tiling->ubFactor = tileElem;
    return ge::GRAPH_SUCCESS;
}

// Set TilingKey based on dtype x IS_ALIGN.
static void SelectTilingKey(gert::TilingContext* context, ge::DataType dataType, int64_t totalIdx, int64_t alignElem)
{
    uint32_t dtypeKey = (dataType == ge::DT_FLOAT) ?
                        static_cast<uint32_t>(C_DT_FLOAT) :
                        static_cast<uint32_t>(C_DT_FLOAT16);
    uint32_t isAlign = (alignElem != 0 && totalIdx % alignElem == 0) ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, dtypeKey, isAlign);
}

// Tiling entry
static ge::graphStatus ApplyAdamaxTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    ApplyAdamaxTilingData* tiling = nullptr;
    OP_CHECK_IF(InitTilingData(context, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "InitTilingData error"), return ge::GRAPH_FAILED);

    int64_t alignElem = 0;
    int64_t usedCoreNum = 0;
    OP_CHECK_IF(ComputeMultiCoreSplit(context, dataType, totalIdx, coreNum, tiling, alignElem, usedCoreNum)
        != ge::GRAPH_SUCCESS, OP_LOGE(context, "ComputeMultiCoreSplit error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeUbSplit(context, dataType, ubSize, alignElem, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ComputeUbSplit error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseScalarAttrs(context, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ParseScalarAttrs error"), return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);
    SelectTilingKey(context, dataType, totalIdx, alignElem);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyAdamax([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyAdamaxCompileInfo {};

IMPL_OP_OPTILING(ApplyAdamax)
    .Tiling(ApplyAdamaxTilingFunc)
    .TilingParse<ApplyAdamaxCompileInfo>(TilingParseForApplyAdamax);

} // namespace optiling
