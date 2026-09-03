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
 * \file huber_loss_tiling.cpp
 * \brief HuberLoss tiling -- framework glue only.
 *
 * Everything that decides anything lives in huber_loss_tiling_calc.h, which
 * has no framework dependency and is unit tested on a plain host compiler.
 * This file fetches shapes, dtype, attributes and platform limits, calls it
 * once, and forwards the result.
 */
#include <cmath>
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/huber_loss_tiling_data.h"
#include "../op_kernel/huber_loss_tiling_key.h"
#include "huber_loss_tiling_calc.h"

namespace optiling {

struct HuberLossCompileInfo {};

static constexpr size_t IDX_INPUT = 0;
static constexpr size_t IDX_TARGET = 1;
static constexpr size_t IDX_OUT = 0;
// Attribute order is fixed in the OpDef: reduction 0, delta 1.
static constexpr size_t ATTR_REDUCTION = 0;
static constexpr size_t ATTR_DELTA = 1;

// Batch scheduling. Not a framework constant -- every operator in this
// repository that needs it declares its own. Required whenever the kernel
// calls SyncAll: the barrier assumes every launched core is resident at once,
// and without batch mode they start in waves and the barrier deadlocks
// probabilistically. Precedent: loss/multilabel_margin_loss,
// loss/mse_loss_v2, loss/cosine_embedding_loss.
static constexpr uint32_t HUBER_LOSS_BATCH_MODE = 1;

static ge::graphStatus FetchInputs(gert::TilingContext* context, int64_t& numel, uint32_t& typeLength)
{
    auto inputShape = context->GetInputShape(IDX_INPUT);
    auto targetShape = context->GetInputShape(IDX_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    // infershape lets unknown dims pass so dynamic-shape graphs are not
    // rejected; here the concrete shapes exist, so this is where a would-be
    // broadcast is stopped.
    OP_CHECK_IF(inputShape->GetStorageShape() != targetShape->GetStorageShape(),
                OP_LOGE(context, "input and target shapes must match"), return ge::GRAPH_FAILED);

    numel = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(numel < 0, OP_LOGE(context, "negative shape size %ld", numel), return ge::GRAPH_FAILED);
    // numel == 0 is legal and not an error here: the kernel handles an empty
    // tensor through the divisor alone.

    auto inputDesc = context->GetInputDesc(IDX_INPUT);
    auto targetDesc = context->GetInputDesc(IDX_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    OP_CHECK_IF(inputDesc->GetDataType() != targetDesc->GetDataType(),
                OP_LOGE(context, "input and target dtypes must match"), return ge::GRAPH_FAILED);
    // The dtype whitelist is explicit rather than inferred from the byte
    // width. Accepting anything two or four bytes wide would admit DT_INT32
    // and DT_INT16, which the kernel has no instantiation for.
    const ge::DataType inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(inputDtype != ge::DT_FLOAT && inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16,
                OP_LOGE(context, "unsupported dtype %d", static_cast<int>(inputDtype)), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::TypeUtils::GetDataTypeLength(inputDtype, typeLength) != true,
                OP_LOGE(context, "cannot size dtype %d", static_cast<int>(inputDtype)), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FetchAttrs(gert::TilingContext* context, int32_t& reduction, float& delta)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* reductionPtr = attrs->GetInt(ATTR_REDUCTION);
    const float* deltaPtr = attrs->GetFloat(ATTR_DELTA);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, deltaPtr);
    // reduction is validated in the WIDER type, before narrowing: integer
    // narrowing wraps, so 2^32 would arrive as a legal 0. delta is checked
    // after narrowing (double to float) for the opposite reason: a tiny
    // positive double such as 1e-300 narrows to 0, and checking beforehand
    // would let it through. The rule is not "always check after narrowing" --
    // it is "check where the illegal values cannot be disguised".
    OP_CHECK_IF(*reductionPtr != HUBER_LOSS_REDUCE_NONE && *reductionPtr != HUBER_LOSS_REDUCE_MEAN &&
                    *reductionPtr != HUBER_LOSS_REDUCE_SUM,
                OP_LOGE(context, "reduction must be 0, 1 or 2, got %ld", static_cast<long>(*reductionPtr)),
                return ge::GRAPH_FAILED);
    reduction = static_cast<int32_t>(*reductionPtr);
    delta = *deltaPtr;
    return ge::GRAPH_SUCCESS;
}

// The output is validated here and not in infershape, because infershape
// computes the output shape rather than receiving one. On the aclnn path the
// caller supplies the output tensor directly and infershape never sees it, so
// without this check nothing constrains it.
static ge::graphStatus CheckOutput(gert::TilingContext* context, int32_t reduction)
{
    auto outShape = context->GetOutputShape(IDX_OUT);
    auto outDesc = context->GetOutputDesc(IDX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    OP_CHECK_IF(outDesc->GetDataType() != context->GetInputDesc(IDX_INPUT)->GetDataType(),
                OP_LOGE(context, "out dtype must match input"), return ge::GRAPH_FAILED);
    if (reduction == HUBER_LOSS_REDUCE_NONE) {
        OP_CHECK_IF(outShape->GetStorageShape() != context->GetInputShape(IDX_INPUT)->GetStorageShape(),
                    OP_LOGE(context, "reduction=none requires out to have the input shape"), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    // A reduced result is one element. Rank 0 is what infershape produces
    // and what the scalar contract calls for; a rank-1 {1} is accepted too
    // because it is equally safe and callers commonly spell a scalar that
    // way. Anything else is refused rather than silently over- or
    // under-written.
    const int64_t outNumel = outShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(outNumel != 1 || outShape->GetStorageShape().GetDimNum() > 1,
                OP_LOGE(context, "reduction=%d requires a scalar out, got numel %ld rank %zu", reduction, outNumel,
                        outShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HuberLossTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    int64_t numel = 0;
    uint32_t typeLength = 0;
    int32_t reduction = 0;
    float delta = 0.0f;
    // Each helper logs the check it fails.
    if (FetchInputs(context, numel, typeLength) != ge::GRAPH_SUCCESS ||
        FetchAttrs(context, reduction, delta) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());

    const huber_loss::HuberTilingPlan plan = huber_loss::CalcTiling(static_cast<uint64_t>(numel), reduction, delta,
                                                                    typeLength, ubSize, aivNum);
    OP_CHECK_IF(!plan.valid,
                OP_LOGE(context, "tiling rejected: numel=%ld reduction=%d delta=%f dtypeBytes=%u ub=%lu aiv=%u", numel,
                        reduction, static_cast<double>(delta), typeLength, static_cast<unsigned long>(ubSize), aivNum),
                return ge::GRAPH_FAILED);

    if (CheckOutput(context, reduction) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    HuberLossTilingData* tiling = context->GetTilingData<HuberLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = plan.data;

    context->SetBlockDim(plan.blockDim);
    // Route through GET_TPL_TILING_KEY, the convention every TPL operator in
    // this repository follows, instead of publishing the raw mode value: this
    // keeps working if the template-key encoding ever changes. With a single
    // width-1 field the encoding is currently the value itself.
    context->SetTilingKey(plan.tilingKey == HUBER_LOSS_SCH_MODE_REDUCE ?
                              GET_TPL_TILING_KEY(HUBER_LOSS_SCH_MODE_REDUCE) :
                              GET_TPL_TILING_KEY(HUBER_LOSS_SCH_MODE_NONE));

    // A non-zero workspace means the kernel will cross-core reduce, which
    // means it will call SyncAll.
    if (plan.workspaceSize > 0) {
        context->SetScheduleMode(HUBER_LOSS_BATCH_MODE);
    }

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(plan.workspaceSize) + ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHuberLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HuberLoss).Tiling(HuberLossTilingFunc).TilingParse<HuberLossCompileInfo>(TilingParseForHuberLoss);

} // namespace optiling
