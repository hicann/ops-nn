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
 * \file hard_swish_grad_tiling.cpp
 * \brief HardSwishGrad 算子 Tiling 实现
 */

#include <algorithm>
#include <set>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/hard_swish_grad_tiling_data.h"
#include "../op_kernel/hard_swish_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t VECTOR_ALIGN_ELEM = 64;
constexpr int64_t UB_MASK_RESERVE = 1024;
constexpr int64_t BYTES_PER_ELEMENT = 48;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t idx = 0; idx < lhs.GetDimNum(); ++idx) {
        if (lhs.GetDim(idx) != rhs.GetDim(idx)) {
            return false;
        }
    }
    return true;
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeDtypeInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto gradShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShapePtr);
    auto gradShape = EnsureNotScalar(gradShapePtr->GetStorageShape());

    auto xShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    OP_CHECK_IF(!IsSameShape(xShape, gradShape), OP_LOGE(context, "HardSwishGrad: x and grad shape mismatch"),
                return ge::GRAPH_FAILED);

    auto gradDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
    auto xDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);

    dataType = gradDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(dataType) == 0, OP_LOGE(context, "HardSwishGrad: unsupported input dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDesc->GetDataType() != dataType, OP_LOGE(context, "HardSwishGrad: x and grad dtype mismatch"),
                return ge::GRAPH_FAILED);

    totalNum = gradShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetTilingKeyByDtype(ge::DataType dataType)
{
    if (dataType == ge::DT_FLOAT16) {
        return GET_TPL_TILING_KEY(HARDSWISHGRAD_TPL_SCH_MODE_FP16);
    }
    if (dataType == ge::DT_BF16) {
        return GET_TPL_TILING_KEY(HARDSWISHGRAD_TPL_SCH_MODE_BF16);
    }
    return GET_TPL_TILING_KEY(HARDSWISHGRAD_TPL_SCH_MODE_FP32);
}

static ge::graphStatus HardSwishGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeDtypeInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeDtypeInfo error"), return ge::GRAPH_FAILED);

    HardSwishGradTilingData* tiling = context->GetTilingData<HardSwishGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(HardSwishGradTilingData), 0, sizeof(HardSwishGradTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    if (totalNum == 0) {
        context->SetBlockDim(1);
        context->SetTilingKey(GetTilingKeyByDtype(dataType));
        return ge::GRAPH_SUCCESS;
    }

    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    int64_t availableUbSize = static_cast<int64_t>(ubSize) - UB_MASK_RESERVE;
    OP_CHECK_IF(availableUbSize <= 0, OP_LOGE(context, "HardSwishGrad: available UB size is invalid"),
                return ge::GRAPH_FAILED);

    int64_t alignUnit = std::max<int64_t>(GetUbBlockSize(context), VECTOR_ALIGN_ELEM);
    int64_t ubFactorRaw = FloorDiv(availableUbSize, BYTES_PER_ELEMENT);
    tiling->ubFactor = FloorAlign(ubFactorRaw, alignUnit);
    OP_CHECK_IF(tiling->ubFactor <= 0, OP_LOGE(context, "HardSwishGrad: ubFactor is invalid"), return ge::GRAPH_FAILED);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    context->SetTilingKey(GetTilingKeyByDtype(dataType));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardSwishGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardSwishGradCompileInfo {};

IMPL_OP_OPTILING(HardSwishGrad)
    .Tiling(HardSwishGradTilingFunc)
    .TilingParse<HardSwishGradCompileInfo>(TilingParseForHardSwishGrad);

} // namespace optiling
