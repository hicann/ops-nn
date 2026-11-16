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
 * \file foreach_atan_tiling.cpp
 * \brief tiling implementation for foreach_atan (arch35 / Ascend950)
 */

#include <limits>
#include <string>
#include <algorithm>
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "graph/utils/type_utils.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/foreach_atan_tiling_data.h"
#include "../../op_kernel/arch35/foreach_atan_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int32_t INPUT_IDX_0 = 0;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t MIN_PER_CORE = 1024;
constexpr int32_t MAX_TENSOR_NUM = 256;
constexpr size_t MAX_DIM_NUM = 8U;

struct ForeachAtanCompileInfo {};

static bool IsSupportedInputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
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

static ge::graphStatus ValidateForeachAtan(gert::TilingContext* context)
{
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(INPUT_IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxInstanceInfoPtr);
    uint64_t tensorNum = idxInstanceInfoPtr->GetInstanceNum();

    OP_CHECK_IF(tensorNum > static_cast<uint64_t>(MAX_TENSOR_NUM),
                OP_LOGE_FOR_INVALID_TENSORNUM(context->GetNodeName(), "x", static_cast<int64_t>(tensorNum),
                                              ("within the range [0, " + std::to_string(MAX_TENSOR_NUM) + "]").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetComputeNodeOutputNum() != tensorNum,
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context->GetNodeName(), "x and out",
                    (std::to_string(tensorNum) + " and " + std::to_string(context->GetComputeNodeOutputNum())).c_str(),
                    "The tensor nums in {x, out} must be the same"),
                return ge::GRAPH_FAILED);

    ge::DataType listDtype = ge::DT_UNDEFINED;
    int64_t totalElements = 0;
    for (uint64_t i = 0; i < tensorNum; i++) {
        auto inputDesc = context->GetDynamicInputDesc(INPUT_IDX_0, i);
        auto inputShape = context->GetDynamicInputShape(INPUT_IDX_0, i);
        OP_CHECK_IF(inputDesc == nullptr || inputShape == nullptr,
                    OP_LOGE(context, "The input %lu desc or shape is null.", i), return ge::GRAPH_FAILED);

        ge::DataType inputDtype = inputDesc->GetDataType();
        OP_CHECK_IF(
            !IsSupportedInputDtype(inputDtype),
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x",
                                      ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(), "FP32, FP16 or BF16"),
            return ge::GRAPH_FAILED);
        if (listDtype == ge::DT_UNDEFINED) {
            listDtype = inputDtype;
        } else {
            OP_CHECK_IF(inputDtype != listDtype,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(),
                            ("The dtypes of all tensors in the tensor list must be the same. "
                             "Currently, the dtype of the " +
                             std::to_string(i) + "th tensor is inconsistent with that (" +
                             ge::TypeUtils::DataTypeToSerialString(listDtype) + ") of other tensors")
                                .c_str()),
                        return ge::GRAPH_FAILED);
        }

        const gert::Shape& inputStorageShape = inputShape->GetStorageShape();
        OP_CHECK_IF(inputStorageShape.GetDimNum() > MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(inputStorageShape.GetDimNum()).c_str(),
                        ("The shape dim of the " + std::to_string(i) +
                         "th tensor in the tensor list should be less than or equal to " + std::to_string(MAX_DIM_NUM))
                            .c_str()),
                    return ge::GRAPH_FAILED);
        int64_t dataCount = inputStorageShape.GetShapeSize();
        OP_CHECK_IF(dataCount < 0 || totalElements > std::numeric_limits<int64_t>::max() - dataCount,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(dataCount).c_str(),
                        ("The shape size of the " + std::to_string(i) +
                         "th tensor in the tensor list must be 0 or a positive number, and the total element "
                         "count of the tensor list must not overflow")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        totalElements += dataCount;

        auto outputDesc = context->GetOutputDesc(i);
        auto outputShape = context->GetOutputShape(i);
        OP_CHECK_IF(outputDesc == nullptr || outputShape == nullptr,
                    OP_LOGE(context, "The output %lu desc or shape is null.", i), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            outputDesc->GetDataType() != listDtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "out", ge::TypeUtils::DataTypeToSerialString(outputDesc->GetDataType()).c_str(),
                "The dtype of out must be the same as x"),
            return ge::GRAPH_FAILED);
        const gert::Shape& outputStorageShape = outputShape->GetStorageShape();
        OP_CHECK_IF(outputStorageShape.GetDimNum() > MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "out", std::to_string(outputStorageShape.GetDimNum()).c_str(),
                        ("The shape dim of the " + std::to_string(i) +
                         "th tensor in tensor list out must be less than or equal to " + std::to_string(MAX_DIM_NUM))
                            .c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputStorageShape.GetShapeSize() != outputStorageShape.GetShapeSize(),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        context->GetNodeName(), "x and out",
                        (std::to_string(inputStorageShape.GetShapeSize()) + " and " +
                         std::to_string(outputStorageShape.GetShapeSize()))
                            .c_str(),
                        ("The shape size of the " + std::to_string(i) +
                         "th tensor in tensor list out must be equal to that of the tensor in the same position "
                         "of the tensor list x")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ForeachAtanTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get tensor list info
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(INPUT_IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxInstanceInfoPtr);
    uint64_t tensorNum = idxInstanceInfoPtr->GetInstanceNum();

    // 3. Validate input/output tensor lists
    OP_CHECK_IF(ValidateForeachAtan(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateForeachAtan failed"),
                return ge::GRAPH_FAILED);

    // 4. Get dtype for tiling key
    auto inputDesc = context->GetDynamicInputDesc(INPUT_IDX_0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    // 5. Fill tiling data
    ForeachAtanTilingData* tiling = context->GetTilingData<ForeachAtanTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ForeachAtanTilingData), 0, sizeof(ForeachAtanTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->tensorCount = static_cast<int32_t>(tensorNum);
    int64_t totalElements = 0;

    for (uint64_t i = 0; i < tensorNum; i++) {
        auto shapePtr = context->GetDynamicInputShape(INPUT_IDX_0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
        auto storageShape = shapePtr->GetStorageShape();
        int64_t numel = storageShape.GetShapeSize();
        tiling->tensorElements[i] = numel;
        totalElements += numel;
    }
    tiling->totalElements = totalElements;

    // 6. Calculate core split
    if (totalElements == 0) {
        tiling->needCoreNum = 1;
        context->SetBlockDim(1);
    } else {
        int64_t needCoreNum = std::min(coreNum, Ops::Base::CeilDiv(totalElements, MIN_PER_CORE));
        tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
        context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    }

    // 7. Set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    // 8. Set local memory size
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));

    // 9. Set tiling key by dtype
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ATAN_TPL_SCH_MODE_FLOAT);
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ATAN_TPL_SCH_MODE_FLOAT16);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ATAN_TPL_SCH_MODE_BF16);
    } else {
        OP_LOGE(context, "unsupported dtype for foreach_atan");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForForeachAtan([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachAtan)
    .Tiling(ForeachAtanTilingFunc)
    .TilingParse<ForeachAtanCompileInfo>(TilingParseForForeachAtan);

} // namespace optiling
