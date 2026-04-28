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
 * \file apply_rms_prop_tiling.cpp
 * \brief ApplyRmsProp 算子 Host Tiling 实现
 */

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <set>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/apply_rms_prop_tiling_data.h"
#include "../op_kernel/apply_rms_prop_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr size_t INPUT_IDX_VAR  = 0;
constexpr size_t INPUT_IDX_MS   = 1;
constexpr size_t INPUT_IDX_MOM  = 2;
constexpr size_t INPUT_IDX_GRAD = 3;

constexpr size_t ATTR_IDX_LR       = 0;
constexpr size_t ATTR_IDX_RHO      = 1;
constexpr size_t ATTR_IDX_MOMENTUM = 2;
constexpr size_t ATTR_IDX_EPSILON  = 3;

constexpr uint32_t WS_SYS_SIZE = 0U;

struct PlatformResource {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    int64_t ubBlockSize = 0;
};

static int64_t EstimateBytesPerElem(ge::DataType dtype, bool isCastBranch)
{
    int64_t typeSize = 4;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) typeSize = 2;
    if (isCastBranch) {
        return 7 * typeSize + 8 * 4;
    }
    return 11 * 4;
}

static ge::graphStatus FetchPlatform(gert::TilingContext* context, PlatformResource& res)
{
    auto* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    res.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(res.coreNum == 0, OP_LOGE(context, "ApplyRmsProp: coreNum is 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, res.ubSize);
    OP_CHECK_IF(res.ubSize == 0, OP_LOGE(context, "ApplyRmsProp: ubSize is 0"),
                return ge::GRAPH_FAILED);
    res.ubBlockSize = GetUbBlockSize(context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FetchAttrs(gert::TilingContext* context, ApplyRmsPropTilingData* td)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* lrPtr  = attrs->GetFloat(ATTR_IDX_LR);
    const float* rhoPtr = attrs->GetFloat(ATTR_IDX_RHO);
    const float* momPtr = attrs->GetFloat(ATTR_IDX_MOMENTUM);
    const float* epsPtr = attrs->GetFloat(ATTR_IDX_EPSILON);
    OP_CHECK_IF(lrPtr == nullptr || rhoPtr == nullptr || momPtr == nullptr || epsPtr == nullptr,
                OP_LOGE(context, "ApplyRmsProp: attr ptr nullptr"),
                return ge::GRAPH_FAILED);
    td->lr    = *lrPtr;
    td->rho1m = 1.0f - (*rhoPtr);
    td->mom   = *momPtr;
    td->eps   = *epsPtr;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeSplit(gert::TilingContext* context, ge::DataType dtype,
                                    int64_t totalLength, const PlatformResource& res,
                                    ApplyRmsPropTilingData* td)
{
    int64_t typeSize = (dtype == ge::DT_FLOAT) ? 4 : 2;
    OP_CHECK_IF(typeSize == 0 || res.ubBlockSize == 0,
                OP_LOGE(context, "ApplyRmsProp: invalid typeSize/ubBlockSize"),
                return ge::GRAPH_FAILED);
    int64_t elemPerBlock = res.ubBlockSize / typeSize;
    OP_CHECK_IF(elemPerBlock == 0,
                OP_LOGE(context, "ApplyRmsProp: elemPerBlock is 0"),
                return ge::GRAPH_FAILED);

    int64_t blockLength = CeilAlign(CeilDiv(totalLength, res.coreNum), elemPerBlock);
    OP_CHECK_IF(blockLength == 0,
                OP_LOGE(context, "ApplyRmsProp: blockLength is 0"),
                return ge::GRAPH_FAILED);
    int64_t usedCoreNum = CeilDiv(totalLength, blockLength);
    int64_t lastBlockLength = totalLength - blockLength * (usedCoreNum - 1);
    if (lastBlockLength <= 0) lastBlockLength = blockLength;

    bool isCastBranch = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    int64_t bytesPerElem = EstimateBytesPerElem(dtype, isCastBranch);
    if (bytesPerElem <= 0) {
        OP_LOGE(context, "ApplyRmsProp: bytesPerElem=%ld invalid (divide-by-zero guard)",
                (long)bytesPerElem);
        return ge::GRAPH_FAILED;
    }
    int64_t safeBytesPerElem = std::max<int64_t>(bytesPerElem, 1);
    int64_t maxTileElems = FloorAlign(static_cast<int64_t>(res.ubSize) / safeBytesPerElem, elemPerBlock);
    OP_CHECK_IF(maxTileElems <= 0,
                OP_LOGE(context, "ApplyRmsProp: maxTileElems=%ld <= 0 (ubSize=%lu, bytesPerElem=%ld)",
                        (long)maxTileElems, (unsigned long)res.ubSize, (long)bytesPerElem),
                return ge::GRAPH_FAILED);

    int64_t tileLength = maxTileElems;
    if (tileLength > blockLength) tileLength = CeilAlign(blockLength, elemPerBlock);
    if (tileLength > maxTileElems) tileLength = maxTileElems;

    int64_t tileNum = CeilDiv(blockLength, tileLength);
    int64_t lastTileLength = blockLength - tileLength * (tileNum - 1);
    if (lastTileLength <= 0) lastTileLength = tileLength;

    int64_t lastBlockTileNum = CeilDiv(lastBlockLength, tileLength);
    int64_t lastBlockLastTileLength = lastBlockLength - tileLength * (lastBlockTileNum - 1);
    if (lastBlockLastTileLength <= 0) lastBlockLastTileLength = tileLength;

    td->totalLength             = totalLength;
    td->blockLength             = blockLength;
    td->lastBlockLength         = lastBlockLength;
    td->tileLength              = tileLength;
    td->tileNum                 = tileNum;
    td->lastTileLength          = lastTileLength;
    td->lastBlockTileNum        = lastBlockTileNum;
    td->lastBlockLastTileLength = lastBlockLastTileLength;
    td->usedCoreNum             = static_cast<int32_t>(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FinalizeTiling(gert::TilingContext* context, ge::DataType dtype, uint32_t blockDim)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WS_SYS_SIZE;
    context->SetBlockDim(blockDim);
    uint32_t dTypeVar = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dTypeVar);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyRmsPropTilingFunc(gert::TilingContext* context)
{
    PlatformResource res;
    if (FetchPlatform(context, res) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    auto* td = context->GetTilingData<ApplyRmsPropTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(ApplyRmsPropTilingData), 0,
                         sizeof(ApplyRmsPropTilingData)) != EOK,
                OP_LOGE(context, "ApplyRmsProp: memset tiling failed"),
                return ge::GRAPH_FAILED);

    auto varDesc = context->GetInputDesc(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    ge::DataType dtype = varDesc->GetDataType();
    const std::set<ge::DataType> supported = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(supported.count(dtype) == 0,
                OP_LOGE(context, "ApplyRmsProp: unsupported dtype=%d", static_cast<int>(dtype)),
                return ge::GRAPH_FAILED);

    auto varShape = context->GetInputShape(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    int64_t totalLength = varShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalLength < 0,
                OP_LOGE(context, "ApplyRmsProp: totalLength=%ld invalid", (long)totalLength),
                return ge::GRAPH_FAILED);
    if (totalLength == 0) {
        OP_LOGI(context, "ApplyRmsProp: empty tensor, skip computation");
        return FinalizeTiling(context, dtype, 1U);
    }

    if (FetchAttrs(context, td) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (ComputeSplit(context, dtype, totalLength, res, td) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    OP_LOGI(context,
        "ApplyRmsProp Tiling: dtype=%d total=%ld block=%ld lastBlock=%ld tile=%ld tileNum=%ld "
        "lastTile=%ld lastBlockTileNum=%ld lastBlockLastTile=%ld cores=%d lr=%f rho1m=%f mom=%f eps=%f",
        (int)dtype, (long)td->totalLength, (long)td->blockLength, (long)td->lastBlockLength,
        (long)td->tileLength, (long)td->tileNum, (long)td->lastTileLength,
        (long)td->lastBlockTileNum, (long)td->lastBlockLastTileLength,
        (int)td->usedCoreNum, td->lr, td->rho1m, td->mom, td->eps);

    return FinalizeTiling(context, dtype, static_cast<uint32_t>(td->usedCoreNum));
}

static ge::graphStatus TilingParseForApplyRmsProp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyRmsPropCompileInfo {};

IMPL_OP_OPTILING(ApplyRmsProp)
    .Tiling(ApplyRmsPropTilingFunc)
    .TilingParse<ApplyRmsPropCompileInfo>(TilingParseForApplyRmsProp);

}  // namespace optiling
