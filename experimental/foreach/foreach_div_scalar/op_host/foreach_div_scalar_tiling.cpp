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

/**
 * \file foreach_div_scalar_tiling.cpp
 * \brief ForeachDivScalar Tiling implementation (arch32 / Ascend910B)
 *
 * x is IR index 0 (DYNAMIC), scalar is IR index 1 (REQUIRED).
 * Scalar value is NOT read in tiling; kernel reads it from GM directly.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/foreach_div_scalar_tiling_data.h"
#include "../op_kernel/foreach_div_scalar_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

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

static int64_t GetTypeSizeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:  return 4;
        case ge::DT_FLOAT16: return 2;
        case ge::DT_BF16:   return 2;
        case ge::DT_DOUBLE: return 8;
        default:             return 4;
    }
}

static ge::graphStatus ForeachDivScalarTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // x is IR index 0 (DYNAMIC)
    auto inputDesc = context->GetDynamicInputDesc(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    auto irInputInfo = context->GetIrInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, irInputInfo);
    uint32_t tensorNum = static_cast<uint32_t>(irInputInfo->GetInstanceNum());

    ForeachDivScalarTilingData* tiling = context->GetTilingData<ForeachDivScalarTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ForeachDivScalarTilingData), 0, sizeof(ForeachDivScalarTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->tensorNum = tensorNum;

    // Determine scalar dtype for kernel to interpret GM data correctly
    tiling->scalarDtype = 0;  // default: float
    {
        // scalar is at physical input index after all dynamic x tensors
        // Use GetInputDesc with physical index = tensorNum (x takes indices 0..tensorNum-1, scalar is next)
        auto scalarDesc = context->GetInputDesc(tensorNum);
        if (scalarDesc != nullptr) {
            ge::DataType sDtype = scalarDesc->GetDataType();
            if (sDtype == ge::DT_FLOAT16) tiling->scalarDtype = 1;
            else if (sDtype == ge::DT_DOUBLE) tiling->scalarDtype = 2;
            else tiling->scalarDtype = 0;
        }
    }

    int64_t totalElements = 0;
    for (uint32_t i = 0; i < tensorNum && i < MAX_TENSOR_NUM; i++) {
        auto shape = context->GetDynamicInputShape(0, i);
        int64_t elemCount = 0;
        if (shape != nullptr) {
            elemCount = shape->GetStorageShape().GetShapeSize();
        }
        tiling->tensorLengths[i] = static_cast<int32_t>(elemCount);
        totalElements += elemCount;
    }
    tiling->totalElements = totalElements;

    // Partition by elements (element-granularity multi-core splitting)
    int64_t blockFactor = 0;
    int64_t usedCoreNum = 1;
    if (totalElements > 0) {
        blockFactor = CeilDiv(totalElements, coreNum);
        usedCoreNum = CeilDiv(totalElements, blockFactor);
    }
    tiling->blockFactor = blockFactor;

    int64_t typeSize = GetTypeSizeBytes(dataType);
    int64_t ubBlockSize = GetUbBlockSize(context);

    // For bf16: need 3 buffers (input bf16 + output bf16 + intermediate fp32)
    // Total per element = 2 + 2 + 4 = 8 bytes, so divide UB by 4 (not 2)
    int64_t ubDivisor = 2;
    if (dataType == ge::DT_BF16) {
        ubDivisor = 4;  // 2 bf16 buffers + 1 fp32 buffer = 2+2+4=8 bytes/elem, 8/2=4
    }
    int64_t ubFactorElems = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, ubDivisor), ubBlockSize);
    tiling->ubFactor = ubFactorElems;

    context->SetBlockDim(usedCoreNum);
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForForeachDivScalar([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ForeachDivScalarCompileInfo {};

IMPL_OP_OPTILING(ForeachDivScalar)
    .Tiling(ForeachDivScalarTilingFunc, sizeof(ForeachDivScalarTilingData))
    .TilingParse<ForeachDivScalarCompileInfo>(TilingParseForForeachDivScalar)
    .InputsDataDependency({1});

} // namespace optiling
