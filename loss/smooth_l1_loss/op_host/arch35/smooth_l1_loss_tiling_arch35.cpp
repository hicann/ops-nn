/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include <cmath>
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/util/math_util.h"
#include "smooth_l1_loss_tiling_arch35.h"
#include <set>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t COMPUTE_TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t COMPARE_ALIGN_ELEMENTS = 256 / COMPUTE_TYPE_SIZE;
constexpr int64_t BUFFER_NUM_DB = 9;
constexpr int64_t BUFFER_NUM_SB = 7;
constexpr float NEGTIVE_CONST_HALF = -0.5f;
constexpr size_t MAX_DIM_NUM = 8;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SmoothL1LossTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto predictShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    auto predictStorageShape = EnsureNotScalar(predictShape->GetStorageShape());
    int64_t totalIdx = predictStorageShape.GetShapeSize();

    OP_CHECK_IF(predictStorageShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context, "predict dim num must be <= 8, got %zu", predictStorageShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    auto labelShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);
    auto labelStorageShape = EnsureNotScalar(labelShape->GetStorageShape());
    OP_CHECK_IF(labelStorageShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context, "label dim num must be <= 8, got %zu", labelStorageShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(predictStorageShape.GetDimNum() != labelStorageShape.GetDimNum(),
                OP_LOGE(context, "predict and label dim num must be the same, predict %zu, label %zu",
                        predictStorageShape.GetDimNum(), labelStorageShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < predictStorageShape.GetDimNum(); i++) {
        int64_t predictDim = predictStorageShape.GetDim(i);
        int64_t labelDim = labelStorageShape.GetDim(i);
        if (predictDim != -1 && labelDim != -1 && predictDim != labelDim) {
            OP_LOGE(context, "predict and label shape mismatch at dim %zu, predict %ld, label %ld", i, predictDim,
                    labelDim);
            return ge::GRAPH_FAILED;
        }
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedDtypes.count(dataType) == 0,
                OP_LOGE(context, "predict only support FP16/FP32/BF16, got %d", static_cast<int32_t>(dataType)),
                return ge::GRAPH_FAILED);

    auto labelDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelDesc);
    OP_CHECK_IF(dataType != labelDesc->GetDataType(), OP_LOGE(context, "predict and label dtype must be the same"),
                return ge::GRAPH_FAILED);

    SmoothL1Loss::SmoothL1LossTilingData* tiling = context->GetTilingData<SmoothL1Loss::SmoothL1LossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SmoothL1Loss::SmoothL1LossTilingData), 0,
                         sizeof(SmoothL1Loss::SmoothL1LossTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* sigmaPtr = attrs->GetAttrPointer<float>(0);
    float sigma = (sigmaPtr == nullptr) ? 1.0f : *sigmaPtr;
    OP_CHECK_IF(
        sigma < 0,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "sigma", std::to_string(sigma).c_str(), "non-negative"),
        return ge::GRAPH_FAILED);

    tiling->Sigma = sigma;
    tiling->MultiplyValue = fabsf(sigma) < 1e-6f ? 0.0f : 0.5f / sigma;
    tiling->AddsValue = NEGTIVE_CONST_HALF * sigma;

    int64_t usedCoreNum = 1;

    if (totalIdx > 0) {
        int64_t ubBlockSize = GetUbBlockSize(context);
        int64_t inputTypeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
        tiling->totalNum = totalIdx;
        tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
        usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);
        int64_t bufferNum = BUFFER_NUM_SB;
        int64_t alignUnit = (ubBlockSize > COMPARE_ALIGN_ELEMENTS) ? ubBlockSize : COMPARE_ALIGN_ELEMENTS;
        tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / inputTypeSize, bufferNum), alignUnit);
        int64_t maxUbElems = static_cast<int64_t>(ubSize) / 4 / 6;
        if (tiling->ubFactor > maxUbElems) {
            tiling->ubFactor = FloorAlign(maxUbElems, alignUnit);
        }
        currentWorkspace[0] = WS_SYS_SIZE;
    } else {
        currentWorkspace[0] = 0;
    }

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSmoothL1Loss(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SmoothL1Loss)
    .Tiling(SmoothL1LossTilingFunc)
    .TilingParse<SmoothL1LossCompileInfo>(TilingParseForSmoothL1Loss);

} // namespace optiling
