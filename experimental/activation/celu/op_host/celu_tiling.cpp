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
 * \file celu_tiling.cpp
 * \brief Celu Tiling implementation (arch35 = Ascend950)
 *
 * Tiling strategy:
 * 1. Get platform info (UB size, AI Core count)
 * 2. Get shape/dtype info
 * 3. Get scalar parameters (alpha1/alpha2/alpha3) via Attrs
 * 4. Multi-core split: blockFactor = ceil(totalNum / coreNum)
 * 5. UB split: ubFactor = floor_align(floor_div(ubSize / typeSize / bufferNum), ubBlockSize)
 *    Celu needs ~5 LocalTensor equivalent spaces (input, output, negResult, zeros, mask overhead)
 * 6. Set TilingKey (dtype template parameter selection via ASCENDC_TPL_SEL_PARAM)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "../op_kernel/celu_tiling_data.h"
#include "../op_kernel/celu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr size_t WS_SYS_SIZE = 0U;
constexpr int64_t BUFFER_NUM = 5;

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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    OP_CHECK_IF(inputShapeX.GetShapeSize() != outShapeY.GetShapeSize(),
        OP_LOGE(context, "Celu: input and output shape size mismatch"),
        return ge::GRAPH_FAILED);

    totalNum = inputShapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetScalarParams(gert::TilingContext* context, float& alpha1, float& alpha2, float& alpha3)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const float* pA1 = attrs->GetAttrPointer<float>(0);
    const float* pA2 = attrs->GetAttrPointer<float>(1);
    const float* pA3 = attrs->GetAttrPointer<float>(2);
    if (pA1 == nullptr || pA2 == nullptr || pA3 == nullptr) {
        OP_LOGE(context, "Celu: GetAttrPointer for alpha1/alpha2/alpha3 returned null");
        return ge::GRAPH_FAILED;
    }

    alpha1 = *pA1;
    alpha2 = *pA2;
    alpha3 = *pA3;

    OP_CHECK_IF(alpha2 <= 0.0f,
        OP_LOGE(context, "Celu: alpha2 must be positive, got %f", alpha2),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CeluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    float alpha1, alpha2, alpha3;
    OP_CHECK_IF(GetScalarParams(context, alpha1, alpha2, alpha3) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetScalarParams error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    CeluTilingData* tiling = context->GetTilingData<CeluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CeluTilingData), 0, sizeof(CeluTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        uint32_t dType = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dType);
        return ge::GRAPH_SUCCESS;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    int64_t typeSize = ge::GetSizeByDataType(dataType);
    int64_t ubBlockSize = GetUbBlockSize(context);
    if (typeSize <= 0 || ubBlockSize <= 0) {
        OP_LOGE(context, "Celu: invalid typeSize=%ld or ubBlockSize=%ld", typeSize, ubBlockSize);
        return ge::GRAPH_FAILED;
    }
    tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, BUFFER_NUM), ubBlockSize);

    tiling->alpha1 = alpha1;
    tiling->alpha2 = alpha2;
    tiling->alpha3 = alpha3;

    context->SetBlockDim(usedCoreNum);

    uint32_t dType = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dType);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CeluCompileInfo {};

IMPL_OP_OPTILING(Celu).Tiling(CeluTilingFunc).TilingParse<CeluCompileInfo>(TilingParseForCelu);

} // namespace optiling
