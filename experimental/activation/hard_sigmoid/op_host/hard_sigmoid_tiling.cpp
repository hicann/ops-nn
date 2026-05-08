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
 * \file hard_sigmoid_tiling.cpp
 * \brief HardSigmoid Host Tiling (atvoss Elewise mode, ElewiseBaseTiling)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "hard_sigmoid/op_kernel/hard_sigmoid_dag.h"
#include "hard_sigmoid/op_kernel/hard_sigmoid_struct.h"

namespace optiling {

using namespace ge;

static ge::graphStatus HardSigmoidTilingFunc(gert::TilingContext* context)
{
    auto tilingData = context->GetTilingData<HardSigmoidTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context, "HardSigmoid: GetTilingData returned nullptr");
        return ge::GRAPH_FAILED;
    }

    float alpha = 1.0f / 6.0f;
    float beta = 0.5f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* alphaPtr = attrs->GetFloat(0);
        if (alphaPtr != nullptr) {
            alpha = *alphaPtr;
        }
        const float* betaPtr = attrs->GetFloat(1);
        if (betaPtr != nullptr) {
            beta = *betaPtr;
        }
    }
    tilingData->alpha = alpha;
    tilingData->beta = beta;

    ::Ops::Base::ElewiseBaseTiling elewiseBaseTiling(context);

    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE(context, "HardSigmoid: GetInputDesc(0) returned nullptr");
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype = inputDesc->GetDataType();
    ge::graphStatus ret;

    if (dtype == ge::DT_INT32) {
        ret = elewiseBaseTiling.DoTiling<
            NsHardSigmoid::HardSigmoidInt32::OpDag>(tilingData->baseTiling);
    } else if (dtype == ge::DT_FLOAT16) {
        ret = elewiseBaseTiling.DoTiling<
            NsHardSigmoid::HardSigmoidWithCast<half>::OpDag>(tilingData->baseTiling);
    } else if (dtype == ge::DT_BF16) {
        ret = elewiseBaseTiling.DoTiling<
            NsHardSigmoid::HardSigmoidWithCast<bfloat16_t>::OpDag>(tilingData->baseTiling);
    } else if (dtype == ge::DT_FLOAT) {
        // fp32 - direct compute, no Cast
        ret = elewiseBaseTiling.DoTiling<
            NsHardSigmoid::HardSigmoidCompute<float>::OpDag>(tilingData->baseTiling);
    } else {
        OP_LOGE(context, "HardSigmoid: unsupported dtype=%d", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "HardSigmoid: ElewiseBaseTiling DoTiling failed");
        return ret;
    }

    // Critical: ElewiseBaseTiling does NOT call SetBlockDim internally; caller must do it.
    // The DoTiling result populates baseTiling.blockNum with the actually-used block count.
    int64_t blockDim = tilingData->baseTiling.blockNum;
    if (blockDim <= 0) {
        OP_LOGE(context, "HardSigmoid: invalid blockDim=%lld from baseTiling", static_cast<long long>(blockDim));
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // Workspace: no extra workspace needed
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }

    uint32_t tilingKey = GET_TPL_TILING_KEY(
        (uint64_t)tilingData->baseTiling.scheMode);
    context->SetTilingKey(tilingKey);

    OP_LOGI(context, "HardSigmoid: Tiling success, scheMode=%u, blockDim=%lld",
            tilingData->baseTiling.scheMode, static_cast<long long>(blockDim));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardSigmoidCompileInfo {};

IMPL_OP_OPTILING(HardSigmoid)
    .Tiling(HardSigmoidTilingFunc)
    .TilingParse<HardSigmoidCompileInfo>(TilingParseForHardSigmoid);

}  // namespace optiling
