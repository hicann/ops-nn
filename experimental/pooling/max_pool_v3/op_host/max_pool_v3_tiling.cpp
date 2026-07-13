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
 * \file max_pool_v3_tiling.cpp
 * \brief max_pool_v3 tiling implementation
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "max_pool_v3/op_kernel/max_pool_v3_tiling_data.h"
#include "max_pool_v3/op_kernel/max_pool_v3_tiling_key.h"
#include "max_pool_v3_util.h"

namespace optiling {

struct MaxPoolV3CompileInfo {};

// Compute per-core distribution for output elements.
// For max_pool_v3, each output element is an independent tile (tileDataNum = 1).
// The function distributes totalElements across up to coreNum cores using a
// "big core / small core" load-balancing scheme: first formerNum cores get
// ceil(total/cores) elements, remaining get floor(total/cores) elements.
static inline uint64_t ComputeCoreDistribution(uint64_t totalElements, uint64_t coreNum, MaxPoolV3TilingData* tiling)
{
    uint64_t usedCoreNum = (coreNum < totalElements) ? coreNum : totalElements;
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // bigCoreDataNum = ceil(totalElements / usedCoreNum)
    uint64_t bigCoreDataNum = (totalElements + usedCoreNum - 1) / usedCoreNum;
    // smallCoreDataNum = floor(totalElements / usedCoreNum)
    uint64_t smallCoreDataNum = totalElements / usedCoreNum;
    uint64_t formerNum = totalElements % usedCoreNum; // cores that get the "big" share

    // tileDataNum = 1 (one element per tile): tileNum = coreDataNum, tailDataNum = 1
    tiling->smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum);
    tiling->tileDataNum = 1;
    tiling->finalBigTileNum = static_cast<uint32_t>(bigCoreDataNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(smallCoreDataNum);
    tiling->smallTailDataNum = 1;
    tiling->bigTailDataNum = 1;
    tiling->tailBlockNum = static_cast<uint32_t>(formerNum);
    return usedCoreNum;
}

static inline void PopulateTilingFields(MaxPoolV3TilingData* tiling, int64_t n, int64_t c, int64_t hIn, int64_t wIn,
                                        int64_t hOut, int64_t wOut, int64_t kH, int64_t kW, int64_t sH, int64_t sW,
                                        int64_t padT, int64_t padL)
{
    tiling->n = static_cast<uint32_t>(n);
    tiling->c = static_cast<uint32_t>(c);
    tiling->hIn = static_cast<uint32_t>(hIn);
    tiling->wIn = static_cast<uint32_t>(wIn);
    tiling->hOut = static_cast<uint32_t>(hOut);
    tiling->wOut = static_cast<uint32_t>(wOut);
    tiling->kH = static_cast<uint32_t>(kH);
    tiling->kW = static_cast<uint32_t>(kW);
    tiling->sH = static_cast<uint32_t>(sH);
    tiling->sW = static_cast<uint32_t>(sW);
    tiling->padT = static_cast<uint32_t>(padT);
    tiling->padL = static_cast<uint32_t>(padL);
    tiling->inHW = static_cast<uint32_t>(hIn * wIn);
    tiling->outHW = static_cast<uint32_t>(hOut * wOut);
}

// Finish tiling setup: distribute cores, populate fields, configure block dim and workspace.
static inline void FinishTilingSetup(gert::TilingContext* context, MaxPoolV3TilingData* tiling, uint64_t totalElements,
                                     uint64_t coreNum, int64_t n, int64_t c, int64_t hIn, int64_t wIn, int64_t hOut,
                                     int64_t wOut, int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t padT,
                                     int64_t padL, uint32_t sysWorkspaceSize)
{
    uint64_t usedCoreNum = ComputeCoreDistribution(totalElements, coreNum, tiling);
    PopulateTilingFields(tiling, n, c, hIn, wIn, hOut, wOut, kH, kW, sH, sW, padT, padL);
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(TPL_SCH_MODE_0));
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
}

// Parsed tiling inputs extracted from context.
struct TilingInputs {
    MaxPoolV3TilingData* tiling;
    uint64_t ubSize, coreNum, totalElements, windowElements, blockSize;
    uint32_t typeLength;
    uint32_t sysWorkspaceSize; // cached once from PlatformAscendC
    int64_t n, c, hIn, wIn, hOut, wOut;
    int64_t kH, kW, sH, sW, padT, padL;
};

// Get block size, tiling data, UB/core info, workspace size, and validate BF16 SoC.
// Fills in->blockSize, in->tiling, in->ubSize, in->coreNum, in->sysWorkspaceSize.
static inline ge::graphStatus GetPlatformInfo(gert::TilingContext* context, TilingInputs* in)
{
    in->blockSize = Ops::Base::GetUbBlockSize(context);
    if (in->blockSize == 0) {
        return ge::GRAPH_FAILED;
    }

    in->tiling = context->GetTilingData<MaxPoolV3TilingData>();
    auto pf = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, pf);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(pf);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, in->ubSize);
    in->coreNum = ascendcPlatform.GetCoreNum();
    in->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto socVersion = ascendcPlatform.GetSocVersion();

    if (socVersion != platform_ascendc::SocVersion::ASCEND910B &&
        socVersion != platform_ascendc::SocVersion::ASCEND310B &&
        context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        OP_LOGE(context, "BF16 not supported on this SoC version.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// Validate spatial dims, compute output dimensions, check bounds and UB capacity.
// Fills in->hOut, in->wOut, in->totalElements, in->windowElements.
static inline ge::graphStatus ValidatePoolingOutput(gert::TilingContext* context, TilingInputs* in,
                                                    const int64_t* ksizeData, const int64_t* stridesData,
                                                    const int64_t* padsData, bool ceilMode, ge::DataType dataType)
{
    if (!ValidateSpatialDims(ksizeData, stridesData, 2, 3, "MaxPoolV3")) {
        return ge::GRAPH_FAILED;
    }

    int64_t padB = padsData[PAD_BOTTOM], padR = padsData[PAD_RIGHT];
    in->hOut = CalculateUpdateDim(in->kH, in->padT, padB, in->sH, ceilMode, in->hIn);
    in->wOut = CalculateUpdateDim(in->kW, in->padL, padR, in->sW, ceilMode, in->wIn);

    if (in->hOut == UNKNOWN_DIM_VALUE || in->wOut == UNKNOWN_DIM_VALUE || in->n == UNKNOWN_DIM_VALUE ||
        in->c == UNKNOWN_DIM_VALUE) {
        OP_LOGE(context, "Dynamic shape not fully resolved.");
        return ge::GRAPH_FAILED;
    }
    if (in->hOut <= 0 || in->wOut <= 0) {
        OP_LOGE(context, "Invalid output dims: hOut=%ld, wOut=%ld", in->hOut, in->wOut);
        return ge::GRAPH_FAILED;
    }

    in->totalElements = static_cast<uint64_t>(in->n) * in->c * in->hOut * in->wOut;
    if (in->totalElements == 0) {
        OP_LOGE(context, "totalElements is 0");
        return ge::GRAPH_FAILED;
    }

    in->windowElements = static_cast<uint64_t>(in->kH) * in->kW;

    // Verify kernel window fits in UB.
    // Each 32B block holds elementsPerBlock elements of type T:
    //   float32 (4B): 2 elements/block;  float16/bf16 (2B): 4 elements/block
    uint64_t elementsPerBlock = (dataType == ge::DT_FLOAT) ? 2 : 4;
    uint64_t tileBlockNum = (in->ubSize / in->blockSize / MAX_POOL_V3_BUFFER_NUM) / elementsPerBlock;
    uint64_t maxUbElements = (tileBlockNum * in->blockSize) / in->typeLength;
    if (in->windowElements > maxUbElements) {
        OP_LOGE(context, "Kernel window too large for UB: %lu > %lu", in->windowElements, maxUbElements);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// Parse shape, attributes, validate, and compute output dims + window size.
// Returns GRAPH_FAILED and logs on error.
static inline ge::graphStatus ParseTilingInputs(gert::TilingContext* context, TilingInputs* in)
{
    if (GetPlatformInfo(context, in) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    ge::TypeUtils::GetDataTypeLength(dataType, in->typeLength);

    auto originShape = inputShape->GetOriginShape();
    in->n = originShape.GetDim(0);
    in->c = originShape.GetDim(1);
    in->hIn = originShape.GetDim(2);
    in->wIn = originShape.GetDim(3);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksizeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksizeAttr);
    auto ksizeData = static_cast<const int64_t*>(ksizeAttr->GetData());
    auto stridesAttr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesAttr);
    auto stridesData = static_cast<const int64_t*>(stridesAttr->GetData());

    // pads is OPTIONAL (default {0,0,0,0}); fall back to DEFAULT_PADS if not provided
    auto padsAttr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
    const int64_t* padsData;
    if (padsAttr != nullptr) {
        padsData = static_cast<const int64_t*>(padsAttr->GetData());
    } else {
        padsData = DEFAULT_PADS;
    }

    auto ceilModeAttr = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
    bool ceilMode = (ceilModeAttr != nullptr) ? *ceilModeAttr : false;

    in->kH = ksizeData[2];
    in->kW = ksizeData[3];
    in->sH = stridesData[2];
    in->sW = stridesData[3];
    in->padT = padsData[PAD_TOP];
    in->padL = padsData[PAD_LEFT];

    return ValidatePoolingOutput(context, in, ksizeData, stridesData, padsData, ceilMode, dataType);
}

static ge::graphStatus MaxPoolV3TilingFunc(gert::TilingContext* context)
{
    TilingInputs in;
    if (ParseTilingInputs(context, &in) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FinishTilingSetup(context, in.tiling, in.totalElements, in.coreNum, in.n, in.c, in.hIn, in.wIn, in.hOut, in.wOut,
                      in.kH, in.kW, in.sH, in.sW, in.padT, in.padL, in.sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaxPoolV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPoolV3).Tiling(MaxPoolV3TilingFunc).TilingParse<MaxPoolV3CompileInfo>(TilingParseForMaxPoolV3);
} // namespace optiling
