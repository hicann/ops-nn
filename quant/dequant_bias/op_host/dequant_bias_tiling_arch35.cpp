/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dequant_bias_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_util.h"
#include <securec.h>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "error_util.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

using namespace ge;

namespace optiling {

using namespace NsDequantBias;
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::NN::OpTiling::IsRegbaseSocVersion;

constexpr int64_t CACHE_LINE = 128;
constexpr int64_t RESERVED_UB = 2048;
constexpr int64_t SIZE_INT32 = sizeof(int32_t);
constexpr int64_t SIZE_FLOAT = sizeof(float);
constexpr int64_t ALIGN_ELEM = 16;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum,
                                       uint32_t* blockSize, uint32_t* vectorLength)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum=0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF((*ubSize <= RESERVED_UB),
                OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u", static_cast<uint32_t>(*ubSize)),
                return ge::GRAPH_FAILED);
    *blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        (*blockSize <= 0),
        OP_LOGE(context->GetNodeName(), "Get block Size failed, block size: %u", static_cast<uint32_t>(*blockSize)),
        return ge::GRAPH_FAILED);

    *vectorLength = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF((*vectorLength <= 0),
                OP_LOGE(context->GetNodeName(), "Get vector Length failed, vector Length: %u",
                        static_cast<uint32_t>(*vectorLength)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void CalcBlockSplit(DequantBiasArch35TilingData* td, int64_t M, int64_t N, int64_t coreNum)
{
    int64_t blockFactor = CeilDiv(M, coreNum);
    if (blockFactor < 1)
        blockFactor = 1;
    int64_t usedCoreNum = CeilDiv(M, blockFactor);
    int64_t blockTailFactor = M - blockFactor * (usedCoreNum - 1);
    if (blockTailFactor <= 0) {
        blockTailFactor = M;
        usedCoreNum = 1;
    }

    td->blockFactor = blockFactor;
    td->blockTailFactor = blockTailFactor;
    td->numCore = static_cast<int32_t>(usedCoreNum);
    td->dim0 = M;
    td->dim1 = N;
}

static void CalcUbCut(DequantBiasArch35TilingData* td, int64_t N, int64_t ubSize, bool hasActivate, bool hasBias,
                      ge::DataType biasDtype, int64_t outputDtype, uint32_t blockSize, uint32_t vectorLength)
{
    int64_t availableUB = static_cast<int64_t>(ubSize) - RESERVED_UB;
    int64_t bytesPerElem = SIZE_INT32 + SIZE_FLOAT + static_cast<int64_t>(sizeof(int16_t));
    if (hasActivate)
        bytesPerElem += SIZE_FLOAT;
    if (hasBias) {
        if (biasDtype == ge::DT_INT32 || biasDtype == ge::DT_FLOAT) {
            bytesPerElem += SIZE_INT32;
        } else {
            bytesPerElem += static_cast<int64_t>(sizeof(int16_t));
        }
    }

    int64_t maxBaseElem = availableUB / bytesPerElem;
    int64_t alignVL = vectorLength / SIZE_FLOAT;
    int64_t align_elems = blockSize / SIZE_INT32;
    int64_t baseLen = FloorAlign(FloorAlign(maxBaseElem, alignVL), align_elems);
    if (baseLen < align_elems)
        baseLen = align_elems;
    if (baseLen > N)
        baseLen = N;

    int64_t tileNum = 1;
    int64_t baseLenTail = N;
    if (N > baseLen) {
        tileNum = CeilDiv(N, baseLen);
        baseLenTail = N - baseLen * (tileNum - 1);
        if (baseLenTail < align_elems) {
            tileNum--;
            baseLenTail = N - baseLen * (tileNum - 1);
        }
    } else {
        baseLen = N;
        baseLenTail = N;
    }

    int64_t alignLenVal = (baseLen + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
    int64_t maxElemsPerRow = alignLenVal * bytesPerElem;
    int64_t baseN = (maxElemsPerRow > 0) ? (availableUB / maxElemsPerRow) : 1;
    baseN = std::min(baseN, td->blockFactor);
    if (baseN < 1)
        baseN = 1;

    td->baseN = baseN;
    td->baseLen = baseLen;
    td->baseLenTail = baseLenTail;
    td->tileNum = tileNum;
    td->hasActivateScale = hasActivate ? 1 : 0;
    td->hasBias = hasBias ? 1 : 0;
    td->outputDtype = static_cast<int32_t>(outputDtype);
}

static int32_t GetBiasDtypeId(bool hasBias, ge::DataType biasDtype)
{
    if (!hasBias) {
        return TPL_BIAS_FLOAT;
    }
    if (biasDtype == ge::DT_FLOAT) {
        return TPL_BIAS_FLOAT;
    }
    if (biasDtype == ge::DT_FLOAT16) {
        return TPL_BIAS_HALF;
    }
    if (biasDtype == ge::DT_BF16) {
        return TPL_BIAS_BFLOAT;
    }
    if (biasDtype == ge::DT_INT32) {
        return TPL_BIAS_INT;
    }
    return TPL_BIAS_FLOAT;
}

static ge::graphStatus DequantBiasTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    uint32_t blockSize = 0;
    uint32_t vectorLength = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum, &blockSize, &vectorLength) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    OP_TILING_CHECK(xShape.GetDimNum() < 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Dims of input should be greater than 2."),
                    return ge::GRAPH_FAILED);

    int64_t M = xShape.GetDim(0);
    int64_t N = xShape.GetDim(1);

    auto asShapePtr = context->GetOptionalInputShape(2);
    bool hasActivate = (asShapePtr != nullptr);

    auto biasShapePtr = context->GetOptionalInputShape(3);
    bool hasBias = (biasShapePtr != nullptr);
    ge::DataType biasDtype = ge::DT_FLOAT;
    if (hasBias) {
        auto biasDesc = context->GetOptionalInputDesc(3);
        if (biasDesc != nullptr) {
            biasDtype = biasDesc->GetDataType();
        }
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* outputDtypePtr = attrs->GetAttrPointer<int64_t>(0);
    int64_t outputDtype = (outputDtypePtr == nullptr) ? 1 : *outputDtypePtr;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = sysWorkspaceSize;

    auto* td = context->GetTilingData<DequantBiasArch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(*td), 0, sizeof(*td)) != EOK, OP_LOGE(context, "memset tiling failed"),
                return ge::GRAPH_FAILED);

    CalcBlockSplit(td, M, N, coreNum);
    CalcUbCut(td, N, static_cast<int64_t>(ubSize), hasActivate, hasBias, biasDtype, outputDtype, blockSize,
              vectorLength);

    context->SetBlockDim(static_cast<uint32_t>(td->numCore));

    int32_t dtypeBiasId = GetBiasDtypeId(hasBias, biasDtype);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(hasBias ? TPL_HAS_BIAS : TPL_NO_BIAS),
                          static_cast<uint32_t>(hasActivate ? TPL_HAS_ACTIVATE : TPL_NO_ACTIVATE),
                          static_cast<uint32_t>(dtypeBiasId));

    return ge::GRAPH_SUCCESS;
}

/* ---- Exported wrapper functions (called from base dequant_bias_tiling.cpp) ---- */
ge::graphStatus Arch35DequantBiasTilingImpl(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingParseContext is nullptr."),
                    return ge::GRAPH_FAILED);
    return DequantBiasTilingFunc(context);
}

ge::graphStatus Arch35DequantBiasTilingParseImpl(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(context == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingParseContext is nullptr."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
