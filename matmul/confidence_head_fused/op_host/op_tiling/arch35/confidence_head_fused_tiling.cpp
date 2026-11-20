/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "confidence_head_fused_tiling.h"
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <string>
#include "securec.h"
#include "confidence_head_fused_tiling_key.h"
#include "error_util.h"

namespace optiling::confidence_head_fused {
namespace {
constexpr size_t HIDDEN_INDEX = 0;
constexpr size_t WEIGHT_INDEX = 1;
constexpr size_t MARKOV_INDEX = 2;
constexpr size_t TEMPERATURE_INDEX = 3;
constexpr size_t BIAS_INDEX = 4;
constexpr const char* INPUT_NAMES[] = {"hidden", "weight", "markov_emb", "temperature", "bias"};
constexpr size_t CLAMP_LOW_ATTR = 0;
constexpr size_t CLAMP_HIGH_ATTR = 1;
constexpr size_t HIDDEN_RANK_WITHOUT_BATCH = 2;
constexpr size_t HIDDEN_RANK_WITH_BATCH = 3;
constexpr uint64_t CUBE_ALIGN = 16;
constexpr uint32_t BASE_N = 128;
constexpr uint32_t BASE_K = 128;
constexpr uint32_t K_L1 = 256;
constexpr uint32_t L1_STAGES = 4;
constexpr uint64_t INPUT_ELEMENT_SIZE = 2;
constexpr uint64_t FLOAT_SIZE = 4;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t UB_ROWS_PER_SLOT = 3; // Two accumulator rows and one output row.
constexpr uint64_t AIC_AIV_RATIO = 2;
constexpr uint64_t MAX_DIM = std::numeric_limits<uint32_t>::max();
constexpr float CLAMP_LIMIT = static_cast<float>(std::numeric_limits<int32_t>::max());

uint64_t AlignK(uint64_t k) { return (k + CUBE_ALIGN - 1) / CUBE_ALIGN * CUBE_ALIGN; }
} // namespace

bool ConfidenceHeadFusedTiling::IsCapable() { return true; }

ge::graphStatus ConfidenceHeadFusedTiling::GetPlatformInfo()
{
    auto platform = context_->GetPlatformInfo();
    if (platform != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platform);
        GetConfidenceHeadFusedPlatformInfo(ascendcPlatform, platformInfo_);
    } else {
        auto compileInfo = context_->GetCompileInfo<ConfidenceHeadFusedCompileInfo>();
        OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        platformInfo_ = *compileInfo;
    }
    OP_TILING_CHECK(platformInfo_.socVersion != platform_ascendc::SocVersion::ASCEND950 &&
                        platformInfo_.socVersion != platform_ascendc::SocVersion::ASCEND350,
                    OP_LOGE(context_->GetNodeName(), "ConfidenceHeadFused only supports Ascend950 and Ascend350."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(platformInfo_.aicNum == 0 || platformInfo_.aivNum != AIC_AIV_RATIO * platformInfo_.aicNum,
                    OP_LOGE(context_->GetNodeName(), "ConfidenceHeadFused requires an AIC:AIV ratio of 1:2."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfidenceHeadFusedTiling::CheckArgs()
{
    for (size_t index : {HIDDEN_INDEX, WEIGHT_INDEX}) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(index));
        OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(index));
    }
    for (size_t index : {MARKOV_INDEX, TEMPERATURE_INDEX, BIAS_INDEX}) {
        auto desc = context_->GetOptionalInputDesc(index);
        auto shape = context_->GetOptionalInputShape(index);
        if (desc == nullptr && shape == nullptr) {
            continue;
        }
        OPS_CHECK_NULL_WITH_CONTEXT(context_, desc);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, shape);
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetFloat(CLAMP_LOW_ATTR));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetFloat(CLAMP_HIGH_ATTR));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfidenceHeadFusedTiling::CheckDtypeValid()
{
    auto inputDtype = context_->GetInputDesc(HIDDEN_INDEX)->GetDataType();
    OP_TILING_CHECK(inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16,
                    OP_LOGE(context_->GetNodeName(), "hidden must be FP16 or BF16."), return ge::GRAPH_FAILED);
    const auto weightDtype = context_->GetInputDesc(WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(weightDtype != inputDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), INPUT_NAMES[WEIGHT_INDEX],
                                                          Ops::Base::ToString(weightDtype),
                                                          "The dtype of weight must match hidden"),
                    return ge::GRAPH_FAILED);
    for (size_t index : {MARKOV_INDEX, TEMPERATURE_INDEX, BIAS_INDEX}) {
        auto desc = context_->GetOptionalInputDesc(index);
        auto expectedDtype = index == MARKOV_INDEX ? inputDtype : ge::DT_FLOAT;
        OP_TILING_CHECK(desc != nullptr && desc->GetDataType() != expectedDtype,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context_->GetNodeName(), INPUT_NAMES[index], Ops::Base::ToString(desc->GetDataType()),
                            index == MARKOV_INDEX ? "The dtype must match hidden" : "The dtype must be FLOAT32"),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(context_->GetOutputDesc(0)->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "Output must be FP32."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfidenceHeadFusedTiling::CheckShapeValid()
{
    const auto& hidden = context_->GetInputShape(HIDDEN_INDEX)->GetOriginShape();
    size_t rank = hidden.GetDimNum();
    OP_TILING_CHECK(rank != HIDDEN_RANK_WITHOUT_BATCH && rank != HIDDEN_RANK_WITH_BATCH,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), INPUT_NAMES[HIDDEN_INDEX],
                                                             std::to_string(rank), "hidden must have rank 2 or 3"),
                    return ge::GRAPH_FAILED);
    for (size_t index = HIDDEN_INDEX; index <= BIAS_INDEX; ++index) {
        auto shape = index >= MARKOV_INDEX ? context_->GetOptionalInputShape(index) : context_->GetInputShape(index);
        if (shape == nullptr) {
            continue;
        }
        const auto& originShape = shape->GetOriginShape();
        if (index != HIDDEN_INDEX) {
            size_t expectedRank = index == MARKOV_INDEX ? rank : 1;
            OP_TILING_CHECK(originShape.GetDimNum() != expectedRank,
                            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                                context_->GetNodeName(), INPUT_NAMES[index], std::to_string(originShape.GetDimNum()),
                                "The rank must be " + std::to_string(expectedRank)),
                            return ge::GRAPH_FAILED);
        }
        for (size_t i = 0; i < originShape.GetDimNum(); ++i) {
            auto dim = originShape.GetDim(i);
            OP_TILING_CHECK(dim < 0 || static_cast<uint64_t>(dim) > MAX_DIM,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), INPUT_NAMES[index],
                                                                  Ops::Base::ToString(originShape),
                                                                  "Each dimension must be in [0, UINT32_MAX]"),
                            return ge::GRAPH_FAILED);
        }
    }
    const auto& out = context_->GetOutputShape(0)->GetOriginShape();
    OP_TILING_CHECK(
        out.GetDimNum() != rank - 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "out", std::to_string(out.GetDimNum()),
                                                 "The rank of out must equal the rank of hidden minus one"),
        return ge::GRAPH_FAILED);
    auto temperature = context_->GetOptionalInputShape(TEMPERATURE_INDEX);
    OP_TILING_CHECK(temperature != nullptr && temperature->GetOriginShape().GetDim(0) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), INPUT_NAMES[TEMPERATURE_INDEX],
                                                          Ops::Base::ToString(temperature->GetOriginShape()),
                                                          "temperature must have shape [1]"),
                    return ge::GRAPH_FAILED);
    auto bias = context_->GetOptionalInputShape(BIAS_INDEX);
    OP_TILING_CHECK(
        bias != nullptr && bias->GetOriginShape().GetDim(0) != hidden.GetDim(rank - 2),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), INPUT_NAMES[BIAS_INDEX], Ops::Base::ToString(bias->GetOriginShape()),
            "bias must have shape [" + std::to_string(hidden.GetDim(rank - 2)) + "], shared across batches"),
        return ge::GRAPH_FAILED);
    auto markov = context_->GetOptionalInputShape(MARKOV_INDEX);
    uint64_t rows = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        uint64_t dim = static_cast<uint64_t>(hidden.GetDim(i));
        OP_TILING_CHECK(out.GetDim(i) != hidden.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "out", Ops::Base::ToString(out),
                                                              "The leading dimensions of out must match hidden"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(markov != nullptr && markov->GetOriginShape().GetDim(i) != hidden.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), INPUT_NAMES[MARKOV_INDEX],
                                                              Ops::Base::ToString(markov->GetOriginShape()),
                                                              "The leading dimensions of markov_emb must match hidden"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(dim != 0 && rows > MAX_DIM / dim,
                        OP_LOGE(context_->GetNodeName(), "Flattened rows must fit UINT32."), return ge::GRAPH_FAILED);
        rows *= dim;
    }
    uint64_t k = static_cast<uint64_t>(hidden.GetDim(rank - 1));
    if (markov != nullptr) {
        k += static_cast<uint64_t>(markov->GetOriginShape().GetDim(rank - 1));
    }
    const auto& weight = context_->GetInputShape(WEIGHT_INDEX)->GetOriginShape();
    OP_TILING_CHECK(k > MAX_DIM || static_cast<uint64_t>(weight.GetDim(0)) != k,
                    OP_LOGE(context_->GetNodeName(), "weight length must equal K1+K2 and fit UINT32."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfidenceHeadFusedTiling::CheckFormat()
{
    for (size_t index = HIDDEN_INDEX; index <= BIAS_INDEX; ++index) {
        auto desc = index >= MARKOV_INDEX ? context_->GetOptionalInputDesc(index) : context_->GetInputDesc(index);
        OP_TILING_CHECK(desc != nullptr && ge::GetPrimaryFormat(desc->GetStorageFormat()) != ge::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), INPUT_NAMES[index],
                                                               Ops::Base::ToString(desc->GetStorageFormat()),
                                                               "The storage format must be ND"),
                        return ge::GRAPH_FAILED);
    }
    const auto outFormat = context_->GetOutputDesc(0)->GetStorageFormat();
    OP_TILING_CHECK(
        ge::GetPrimaryFormat(outFormat) != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), "out", Ops::Base::ToString(outFormat),
                                               "The storage format must be ND"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfidenceHeadFusedTiling::GetArgs()
{
    const auto& hidden = context_->GetInputShape(HIDDEN_INDEX)->GetOriginShape();
    size_t rank = hidden.GetDimNum();
    auto markov = context_->GetOptionalInputShape(MARKOV_INDEX);
    auto& params = tilingData_.params;
    uint64_t rows = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        rows *= static_cast<uint64_t>(hidden.GetDim(i));
    }
    params.n = static_cast<uint32_t>(rows);
    params.kEmbed = static_cast<uint32_t>(hidden.GetDim(rank - 1));
    params.hasMarkov = markov != nullptr;
    params.kMarkov = markov == nullptr ? 0 : static_cast<uint32_t>(markov->GetOriginShape().GetDim(rank - 1));
    params.hasBias = context_->GetOptionalInputShape(BIAS_INDEX) != nullptr;
    params.biasN = static_cast<uint32_t>(hidden.GetDim(rank - 2));
    auto attrs = context_->GetAttrs();
    params.lowLimit = *attrs->GetFloat(CLAMP_LOW_ATTR);
    params.highLimit = *attrs->GetFloat(CLAMP_HIGH_ATTR);
    // Reserved default bounds disable the corresponding clamp operation.
    params.hasLowLimit = params.lowLimit != -CLAMP_LIMIT;
    params.hasHighLimit = params.highLimit != CLAMP_LIMIT;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfidenceHeadFusedTiling::GetShapeAttrsInfo()
{
    if (CheckArgs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtypeValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return GetArgs();
}

ge::graphStatus ConfidenceHeadFusedTiling::DoOpTiling()
{
    auto& params = tilingData_.params;
    auto& cube = tilingData_.cubeTiling;
    cube.baseN = BASE_N;
    cube.baseK = BASE_K;
    cube.kL1 = K_L1;
    cube.l1BufferNum = L1_STAGES;
    uint64_t tiles = (static_cast<uint64_t>(params.n) + BASE_N - 1) / BASE_N;
    if (params.hasBias != 0 && params.biasN != 0) {
        // Bias[N] repeats per batch; no tile may cross its broadcast boundary.
        tiles = (static_cast<uint64_t>(params.n) / params.biasN) *
                ((static_cast<uint64_t>(params.biasN) + BASE_N - 1) / BASE_N);
    }
    params.usedCoreNum = static_cast<uint32_t>(std::max<uint64_t>(1, std::min(tiles, platformInfo_.aicNum)));
    // K=0 reuses the accumulator buffer for bias; no separate UB bias row is needed.
    constexpr uint64_t ubBytes = DOUBLE_BUFFER * UB_ROWS_PER_SLOT * BASE_N * FLOAT_SIZE;
    OP_TILING_CHECK(platformInfo_.ubSize < ubBytes,
                    OP_LOGE(context_->GetNodeName(), "UB is insufficient for ConfidenceHeadFused."),
                    return ge::GRAPH_FAILED);
    if (params.n != 0 && (params.kEmbed != 0 || params.kMarkov != 0)) {
        uint64_t weightBytes = (AlignK(params.kEmbed) + AlignK(params.kMarkov)) * INPUT_ELEMENT_SIZE;
        uint64_t featureBytes = static_cast<uint64_t>(K_L1) * BASE_N * INPUT_ELEMENT_SIZE * L1_STAGES;
        uint64_t biasBytes = params.hasBias != 0 ? static_cast<uint64_t>(BASE_N) * FLOAT_SIZE * L1_STAGES : 0;
        OP_TILING_CHECK(weightBytes + featureBytes + biasBytes >= platformInfo_.l1Size ||
                            DOUBLE_BUFFER * CUBE_ALIGN * BASE_K * INPUT_ELEMENT_SIZE > platformInfo_.l0aSize ||
                            DOUBLE_BUFFER * BASE_K * BASE_N * INPUT_ELEMENT_SIZE > platformInfo_.l0bSize ||
                            DOUBLE_BUFFER * CUBE_ALIGN * BASE_N * FLOAT_SIZE > platformInfo_.l0cSize,
                        OP_LOGE(context_->GetNodeName(), "The full-load GEMV tile exceeds on-chip memory capacity."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfidenceHeadFusedTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ConfidenceHeadFusedTiling::GetTilingKey() const
{
    return ConfidenceHeadFusedTilingKey().SetHasMarkov(tilingData_.params.hasMarkov != 0).GetTilingKey();
}

ge::graphStatus ConfidenceHeadFusedTiling::GetWorkspaceSize()
{
    auto workspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    // The accumulator is passed directly through UB; no GM intermediate is required.
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfidenceHeadFusedTiling::PostTiling()
{
    auto raw = context_->GetRawTilingData();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, raw);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, raw->GetData());
    static_assert(sizeof(CHF::ConfidenceHeadFusedTilingData) % sizeof(uint64_t) == 0);
    OP_TILING_CHECK(memcpy_s(raw->GetData(), raw->GetCapacity(), &tilingData_, sizeof(tilingData_)) != EOK,
                    OP_LOGE(context_->GetNodeName(), "Failed to copy ConfidenceHeadFused tiling data."),
                    return ge::GRAPH_FAILED);
    raw->SetDataSize(sizeof(tilingData_));
    context_->SetBlockDim(tilingData_.params.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling::confidence_head_fused
