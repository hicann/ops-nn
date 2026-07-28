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
 * \file l2_normalize_grad_tiling.cpp
 * \brief L2NormalizeGrad arch35 (Ascend950) tiling.
 *
 * Splits the outer groups across AI cores (no cross-core reduction). Selects the DX template by the
 * [outer, D, inner] decomposition: inner==1 -> full-load (7000) or split-D (7010) by D vs UB;
 * inner>1 -> strided (7020). Empty tensor -> 8000.
 */

#include <algorithm>
#include <graph/utils/type_utils.h>
#include "l2_normalize_grad_tiling.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "error_util.h"
#include "../../op_kernel/arch35/l2_normalize_grad_tiling_data.h"

using namespace ge;

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::ToString;

constexpr int32_t INPUT_X_IDX = 0;
constexpr int32_t INPUT_Y_IDX = 1;
constexpr int32_t INPUT_DY_IDX = 2;
constexpr int32_t OUTPUT_DX_IDX = 0;
constexpr int64_t ATTR_DIM_IDX = 0;
constexpr int64_t ATTR_EPS_IDX = 1;
constexpr int64_t DX_UB_FACTOR = 6144; // full-load threshold (fp32 elements), matches kernel
constexpr int64_t BUFFER_NUM = 2;      // double buffer
constexpr int64_t STRIDED_BUF_NUM = 4; // x + y + dy + dx
constexpr int64_t FLOAT_BYTE = 4;
constexpr int64_t UB_RESERVED = 8 * 1024;
constexpr int64_t STRIDED_VL_SLACK = 64; // 内核 strided tileElems += V_LENGTH(=VRegSize/4=64 fp32 lanes)
constexpr int64_t FP32_BLOCK = 8;        // FLOAT_NUM_BLOCK(fp32 每 32B 块元素数)
constexpr int64_t FP16_BLOCK = 16;       // HALF_NUM_BLOCK(fp16 每 32B 块元素数)
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t EMPTY_TILING_KEY = 8000;
constexpr uint32_t FULL_LOAD_KEY = 7000;
constexpr uint32_t SPLIT_D_KEY = 7010;
constexpr uint32_t STRIDED_KEY = 7020;

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

static ge::graphStatus CheckDtype(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF((xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", ToString(xDtype).c_str(), "FLOAT or FLOAT16"),
                return ge::GRAPH_FAILED);

    auto yDesc = context->GetInputDesc(INPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    auto dyDesc = context->GetInputDesc(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyDesc);
    auto dxDesc = context->GetOutputDesc(OUTPUT_DX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxDesc);
    if (yDesc->GetDataType() != xDtype || dyDesc->GetDataType() != xDtype || dxDesc->GetDataType() != xDtype) {
        OP_LOGE(context->GetNodeName(), "The dtypes of x, y, dy and dx must all be the same.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(gert::TilingContext* context)
{
    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto yShapePtr = context->GetInputShape(INPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto dyShapePtr = context->GetInputShape(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const gert::Shape& yShape = yShapePtr->GetStorageShape();
    const gert::Shape& dyShape = dyShapePtr->GetStorageShape();

    if (xShape.GetDimNum() != yShape.GetDimNum() || xShape.GetDimNum() != dyShape.GetDimNum()) {
        OP_LOGE(context->GetNodeName(), "The shapes of x, y and dy must be the same (rank mismatch).");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (xShape.GetDim(i) != yShape.GetDim(i) || xShape.GetDim(i) != dyShape.GetDim(i)) {
            OP_LOGE(context->GetNodeName(), "The shapes of x, y and dy must be the same (dim %zu mismatch).", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// Reads dim/eps attrs and derives [outer, D, inner] and totalNum from the x shape.
static ge::graphStatus ResolveDimAndShape(gert::TilingContext* context, int64_t& outer, int64_t& dimLen, int64_t& inner,
                                          int64_t& totalNum, float& eps)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::ContinuousVector* dimAttr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_DIM_IDX);
    const float* epsAttr = attrs->GetFloat(ATTR_EPS_IDX);
    eps = (epsAttr != nullptr) ? *epsAttr : 1e-4f;

    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());

    int64_t dimVal = 1; // A2 impl default when dim is unset/empty
    if (dimAttr != nullptr && dimAttr->GetSize() > 0) {
        if (dimAttr->GetSize() > 1) {
            // Multi-axis dim (e.g. legacy 5HD [1,4]) is not supported by this arch35 ND-only kernel.
            OP_LOGE(context->GetNodeName(),
                    "L2NormalizeGrad arch35 supports a single normalization axis only; got dim of size %zu.",
                    dimAttr->GetSize());
            return ge::GRAPH_FAILED;
        }
        const int64_t* dimData = reinterpret_cast<const int64_t*>(dimAttr->GetData());
        dimVal = dimData[0];
    }
    if (dimVal < 0) {
        dimVal += rank;
    }
    if (dimVal < 0 || dimVal >= rank) {
        OP_LOGE(context->GetNodeName(), "Resolved dim %ld is out of range for rank %ld.", dimVal, rank);
        return ge::GRAPH_FAILED;
    }

    outer = 1;
    dimLen = xShape.GetDim(dimVal);
    inner = 1;
    totalNum = 1;
    for (int64_t i = 0; i < dimVal; i++) {
        outer *= xShape.GetDim(i);
    }
    for (int64_t i = dimVal + 1; i < rank; i++) {
        inner *= xShape.GetDim(i);
    }
    for (int64_t i = 0; i < rank; i++) {
        totalNum *= xShape.GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}

static void SelectTemplate(int64_t outer, int64_t dimLen, int64_t inner, int64_t totalNum, int64_t coreNum,
                           uint64_t ubSize, int64_t block, uint32_t& tilingKey, int64_t& blockFactor,
                           int64_t& usedCoreNum, int64_t& colFactor)
{
    colFactor = 0;
    if (totalNum == 0) {
        tilingKey = EMPTY_TILING_KEY;
        blockFactor = 0;
        usedCoreNum = 1;
        return;
    }
    blockFactor = CeilDiv(outer, coreNum);
    if (blockFactor < 1) {
        blockFactor = 1;
    }
    usedCoreNum = CeilDiv(outer, blockFactor);
    if (inner == 1) {
        tilingKey = (dimLen <= DX_UB_FACTOR) ? FULL_LOAD_KEY : SPLIT_D_KEY;
    } else {
        tilingKey = STRIDED_KEY;
        // inner-column tile so 4 fp32 buffers (double-buffered) fit UB。内核每 buffer 实际元素数 =
        // dimLen * AlignUp(colFactor, block) + STRIDED_VL_SLACK(见 strided kernel tileElems)。**预算必须按
        // 对齐后大小 + VL slack 算**:否则 dimLen 大(如 128/256)时对齐溢出 (align_gap*dimLen) 超过 UB_RESERVED
        // → UB 越界 → VEC_ERROR / NO_OUTPUT。maxCol 对齐下取整,保证 AlignUp(colFactor) 不超预算。
        int64_t ubBudget = static_cast<int64_t>(ubSize) - UB_RESERVED;
        int64_t perBufElems = ubBudget / (STRIDED_BUF_NUM * BUFFER_NUM * FLOAT_BYTE);
        int64_t maxAlignedCol = (dimLen > 0) ? ((perBufElems - STRIDED_VL_SLACK) / dimLen) : inner;
        int64_t maxCol = (maxAlignedCol / block) * block;
        if (maxCol < block) {
            maxCol = block;
        }
        colFactor = std::min(inner, maxCol);
    }
}

static ge::graphStatus L2NormalizeGradTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Inputs dtype invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Inputs shape invalid."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t outer = 1;
    int64_t dimLen = 1;
    int64_t inner = 1;
    int64_t totalNum = 0;
    float eps = 1e-4f;
    OP_CHECK_IF(ResolveDimAndShape(context, outer, dimLen, inner, totalNum, eps) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to resolve dim/shape."), return ge::GRAPH_FAILED);

    uint32_t tilingKey = 0;
    int64_t blockFactor = 0;
    int64_t usedCoreNum = 0;
    int64_t colFactor = 0;
    // strided 路径 UB 预算按 dtype 的块对齐(fp32=8 / fp16=16),与 kernel colFactorAlign 一致
    auto xDescForBlock = context->GetInputDesc(INPUT_X_IDX);
    int64_t block = (xDescForBlock != nullptr && xDescForBlock->GetDataType() == ge::DataType::DT_FLOAT) ? FP32_BLOCK :
                                                                                                           FP16_BLOCK;
    SelectTemplate(outer, dimLen, inner, totalNum, coreNum, ubSize, block, tilingKey, blockFactor, usedCoreNum,
                   colFactor);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    L2NormalizeGradTilingData* tiling = context->GetTilingData<L2NormalizeGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->outer = outer;
    tiling->dimLen = dimLen;
    tiling->inner = inner;
    tiling->blockFactor = blockFactor;
    tiling->usedCoreNum = usedCoreNum;
    tiling->colFactor = colFactor;
    tiling->eps = eps;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum > 0 ? usedCoreNum : 1));
    context->SetTilingKey(tilingKey);

    OP_LOGI(context->GetNodeName(),
            "L2NormalizeGrad tiling: key=%u outer=%ld D=%ld inner=%ld blockFactor=%ld usedCore=%ld colFactor=%ld "
            "eps=%f",
            tilingKey, outer, dimLen, inner, blockFactor, usedCoreNum, colFactor, eps);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForL2NormalizeGrad(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<L2NormalizeGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(L2NormalizeGrad)
    .Tiling(L2NormalizeGradTilingFunc)
    .TilingParse<L2NormalizeGradCompileInfo>(TilingParseForL2NormalizeGrad);

} // namespace optiling
