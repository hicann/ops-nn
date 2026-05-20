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
 * \file ascend_anti_quant_tiling_arch35.cpp
 * \brief Pure regbase tiling for AscendAntiQuant on Ascend950 / arch35.
 *        Self-computes blockNum / blockFormer / ubFormer; no dependency on
 *        atvoss ElewiseBaseTiling / DAGSch.
 */
#include <algorithm>
#include <cstdint>
#include <graph/utils/type_utils.h>
#include "ascend_anti_quant_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"

#include "quant/ascend_anti_quant/op_kernel/arch35/ascend_anti_quant_tilingdata.h"
#include "quant/ascend_anti_quant/op_kernel/arch35/ascend_anti_quant_struct.h"

namespace optiling {
using namespace AscendAntiQuantOp;

static constexpr uint64_t ASCEND_WORKSPACE         = 0;
static constexpr int32_t  ATTR_AAQ_SCALE_POS       = 0;
static constexpr int32_t  ATTR_AAQ_OFFSET_POS      = 1;
static constexpr int32_t  ATTR_AAQ_DTYPE_POS       = 2;
static constexpr int32_t  ATTR_AAQ_SQRT_MODE_POS   = 3;

// Cache line of A5 is 128B. Reserve ub for scalar buf + stack/spill margin.
static constexpr int64_t  AAQ_CACHE_LINE_BYTES     = 128;
static constexpr int64_t  AAQ_UB_RESERVE_BYTES     = 2048;
// Kernel uses BUFFER_NUM=2 (double-buffer for in/out queues).
static constexpr int64_t  AAQ_BUFFER_NUM           = 2;

struct AscendAntiQuantCompileInfo {
    int32_t  vectorCoreNum = 0;
    uint64_t ubSize        = 0;
};

class AscendAntiQuantTiling {
public:
    explicit AscendAntiQuantTiling(gert::TilingContext* ctx)
        : tilingContext(ctx) {
        tiling = tilingContext->GetTilingData<AscendAntiQuantTilingData>();
    }
    ge::graphStatus RunTiling();
    AscendAntiQuantTilingData* tiling = nullptr;

protected:
    ge::graphStatus GetCompileInfo();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus FetchAttrs();
    void            CalcBaseTiling(int64_t elemNum);
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext;
    ge::DataType inputDtype  = ge::DT_UNDEFINED;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    int64_t  coreNum_ = 0;
    uint64_t ubSize_  = 0;
    int64_t  elemNum_ = 0;

    float    scale_  = 1.0f;
    float    offset_ = 0.0f;
    int32_t  sqrtMode_ = 0;
};

ge::graphStatus AscendAntiQuantTiling::GetCompileInfo() {
    auto ci = tilingContext->GetCompileInfo<AscendAntiQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, ci);
    coreNum_ = ci->vectorCoreNum;
    ubSize_  = ci->ubSize;
    OP_CHECK_IF(coreNum_ <= 0 || ubSize_ == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "compile_info",
                ("coreNum=" + std::to_string(coreNum_) + ", ubSize=" + std::to_string(ubSize_)).c_str(),
                "coreNum must be greater than 0 and ubSize must be greater than 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantTiling::CalcInputDtype() {
    auto d = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, d);
    inputDtype = d->GetDataType();
    OP_CHECK_IF(inputDtype != ge::DT_INT8 && inputDtype != ge::DT_HIFLOAT8 &&
                inputDtype != ge::DT_FLOAT8_E5M2 && inputDtype != ge::DT_FLOAT8_E4M3FN,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x",
                ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(),
                "int8, hifloat8, float8_e5m2 or float8_e4m3fn"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantTiling::CalcOutputDtype() {
    auto d = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, d);
    outputDtype = d->GetDataType();
    OP_CHECK_IF(outputDtype != ge::DT_FLOAT16 && outputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
                ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str(),
                "float16 or float"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantTiling::CheckShape() {
    auto xs = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xs);
    const gert::Shape& xShape = Ops::NN::OpTiling::EnsureNotScalar(xs->GetStorageShape());
    auto ys = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, ys);
    const gert::Shape& yShape = Ops::NN::OpTiling::EnsureNotScalar(ys->GetStorageShape());
    OP_CHECK_IF(xShape != yShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x and y",
                (Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(yShape)).c_str(),
                "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    elemNum_ = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        int64_t d = xShape.GetDim(i);
        OP_CHECK_IF(d == 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "x",
                    ("dim[" + std::to_string(i) + "] is 0").c_str(),
                    "All axes of x must be greater than 0"),
            return ge::GRAPH_FAILED);
        elemNum_ *= d;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantTiling::FetchAttrs() {
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* scalePtr  = attrs->GetAttrPointer<float>(ATTR_AAQ_SCALE_POS);
    const float* offsetPtr = attrs->GetAttrPointer<float>(ATTR_AAQ_OFFSET_POS);
    const bool*  sqrtPtr   = attrs->GetAttrPointer<bool>(ATTR_AAQ_SQRT_MODE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, scalePtr);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, offsetPtr);
    scale_    = *scalePtr;
    offset_   = *offsetPtr;
    sqrtMode_ = (sqrtPtr != nullptr && *sqrtPtr) ? 1 : 0;

    const int32_t* dstTypePtr = attrs->GetAttrPointer<int32_t>(ATTR_AAQ_DTYPE_POS);
    if (dstTypePtr != nullptr) {
        const auto dstType = static_cast<ge::DataType>(*dstTypePtr);
        OP_CHECK_IF(dstType != ge::DT_FLOAT16 && dstType != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "dtype",
                    ge::TypeUtils::DataTypeToSerialString(dstType).c_str(),
                    "float16 or float"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(dstType != outputDtype,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "dtype and y",
                    (ge::TypeUtils::DataTypeToSerialString(dstType) + " and " +
                     ge::TypeUtils::DataTypeToSerialString(outputDtype)).c_str(),
                    "The dtype attribute must match the dtype of y"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// per-elem byte budget for the regbase kernel:
//   in queue  : sizeof(T) (T is always 1B for int8/fp8/hifloat8) * BUFFER_NUM
//   out queue : sizeof(U) (2 or 4)                              * BUFFER_NUM
static int64_t CalcMaxUbElem(uint64_t ubSize, ge::DataType outputDtype)
{
    const int64_t inBytes  = 1;
    const int64_t outBytes = (outputDtype == ge::DT_FLOAT) ? 4 : 2;
    const int64_t perElemBytes = (inBytes + outBytes) * AAQ_BUFFER_NUM;
    int64_t usableUb = static_cast<int64_t>(ubSize) - AAQ_UB_RESERVE_BYTES;
    if (usableUb < AAQ_CACHE_LINE_BYTES) {
        usableUb = AAQ_CACHE_LINE_BYTES;
    }
    // align down to cache-line worth of elements (128 elems = 128B for 1B input)
    int64_t maxUbElem = Ops::Base::FloorAlign(usableUb / perElemBytes, AAQ_CACHE_LINE_BYTES);
    return (maxUbElem > 0) ? maxUbElem : AAQ_CACHE_LINE_BYTES;
}

static void CalcCoreSlice(int64_t elemNum, int64_t coreNum, int64_t& blockFormer, int64_t& blockNum)
{
    blockFormer = Ops::Base::CeilAlign(Ops::Base::CeilDiv(elemNum, coreNum), AAQ_CACHE_LINE_BYTES);
    if (blockFormer <= 0) {
        blockFormer = AAQ_CACHE_LINE_BYTES;
    }
    blockNum = Ops::Base::CeilDiv(elemNum, blockFormer);
    if (blockNum > coreNum) {
        blockNum = coreNum;
    }
    if (blockNum <= 0) {
        blockNum = 1;
    }
}

static void CalcUbLoopTail(int64_t blockLen, int64_t ubFormer, int64_t& loop, int64_t& tail)
{
    if (blockLen <= 0) {
        loop = 0;
        tail = 0;
        return;
    }
    loop = Ops::Base::CeilDiv(blockLen, ubFormer);
    int64_t rem = blockLen - ubFormer * (loop - 1);
    tail = (rem == ubFormer) ? 0 : rem;
}

void AscendAntiQuantTiling::CalcBaseTiling(int64_t elemNum) {
    int64_t maxUbElem = CalcMaxUbElem(ubSize_, outputDtype);

    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    CalcCoreSlice(elemNum, coreNum_, blockFormer, blockNum);

    int64_t ubFormer = std::min<int64_t>(blockFormer, maxUbElem);
    if (ubFormer <= 0) {
        ubFormer = AAQ_CACHE_LINE_BYTES;
    }

    int64_t tailBlock = elemNum - blockFormer * (blockNum - 1);
    if (tailBlock <= 0) {
        tailBlock = blockFormer;
    }

    int64_t ubLoopOfFormerBlock = 0, ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock   = 0, ubTailOfTailBlock   = 0;
    CalcUbLoopTail(blockFormer, ubFormer, ubLoopOfFormerBlock, ubTailOfFormerBlock);
    CalcUbLoopTail(tailBlock,   ubFormer, ubLoopOfTailBlock,   ubTailOfTailBlock);

    auto& bt = tiling->baseTiling;
    bt.dim0                 = elemNum;
    bt.coreNum              = static_cast<int32_t>(blockNum);
    bt.ubFormer             = static_cast<int32_t>(ubFormer);
    bt.blockFormer          = blockFormer;
    bt.blockNum             = blockNum;
    bt.ubLoopOfFormerBlock  = ubLoopOfFormerBlock;
    bt.ubLoopOfTailBlock    = ubLoopOfTailBlock;
    bt.ubTailOfFormerBlock  = ubTailOfFormerBlock;
    bt.ubTailOfTailBlock    = ubTailOfTailBlock;
    bt.elemNum              = elemNum;
    bt.scheMode             = 0;  // single regbase strategy
}

ge::graphStatus AscendAntiQuantTiling::SetTilingData() {
    size_t* ws = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, ws);
    ws[0] = static_cast<size_t>(ASCEND_WORKSPACE);

    tiling->scale    = scale_;
    tiling->offset   = offset_;
    tiling->sqrtMode = sqrtMode_;
    tiling->reserved = 0;

    const uint64_t tilingKey = static_cast<uint64_t>(sqrtMode_ ? 1 : 0);
    OP_LOGD(tilingContext->GetNodeName(),
            "[AscendAntiQuant] tilingKey=%lu sqrtMode=%lu blockNum=%ld blockFormer=%ld ubFormer=%d elemNum=%ld",
            tilingKey, tilingKey,
            tiling->baseTiling.blockNum, tiling->baseTiling.blockFormer,
            tiling->baseTiling.ubFormer, tiling->baseTiling.elemNum);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(static_cast<uint32_t>(tiling->baseTiling.blockNum));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantTiling::RunTiling() {
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    OP_CHECK_IF(GetCompileInfo()  == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "Failed to get compile info"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcInputDtype()  == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "Failed to check input dtype"),  return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "Failed to check output dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape()      == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "Failed to check shape"),       return ge::GRAPH_FAILED);
    OP_CHECK_IF(FetchAttrs()      == ge::GRAPH_FAILED,
        OP_LOGE(tilingContext, "Failed to fetch attrs"),       return ge::GRAPH_FAILED);

    CalcBaseTiling(elemNum_);
    return SetTilingData();
}

ge::graphStatus Tiling4AscendAntiQuant(gert::TilingContext* context) {
    OP_CHECK_IF(context == nullptr,
        OP_LOGE("AscendAntiQuant", "Tiling context is null."),
        return ge::GRAPH_FAILED);
    AscendAntiQuantTiling t(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, t.tiling);
    return t.RunTiling();
}

ge::graphStatus TilingPrepareForAscendAntiQuant(gert::TilingParseContext* context) {
    OP_CHECK_IF(context == nullptr,
        OP_LOGE("AscendAntiQuant", "TilingParse context is null."),
        return ge::GRAPH_FAILED);
    auto ci = context->GetCompiledInfo<AscendAntiQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ci->ubSize);
    OP_CHECK_IF(ci->vectorCoreNum <= 0 || ci->ubSize == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("AscendAntiQuant", "hardware_info",
                ("vectorCoreNum=" + std::to_string(ci->vectorCoreNum) +
                 ", ubSize=" + std::to_string(ci->ubSize)).c_str(),
                "vectorCoreNum must be greater than 0 and ubSize must be greater than 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AscendAntiQuant)
    .Tiling(Tiling4AscendAntiQuant)
    .TilingParse<AscendAntiQuantCompileInfo>(TilingPrepareForAscendAntiQuant);

}  // namespace optiling
