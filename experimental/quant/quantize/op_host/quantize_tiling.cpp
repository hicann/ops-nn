/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize_tiling.cpp
 * \brief Quantize host tiling, ascend910b (DAV_2201) standard model.
 *
 * Folds x to [dimBefore, dimAxis, dimAfter] around the (already positive) axis.
 *   per-tensor  (scales.numel == 1): one scalar scale/offset over all elements; split total elements over cores.
 *   per-channel (scales.numel  > 1): scale[c] applies to axis-index c; rows = dimBefore*dimAxis, cols = dimAfter;
 *                                    row r uses channel = r % dimAxis; split rows over cores.
 */

#include "quantize_tiling.h"
#include "quantize/op_kernel/quantize_tiling_key.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {
// perMode template-argument values; single source of truth is op_kernel/quantize_tiling_key.h, which both the
// host (declare-params for GET_TPL_TILING_KEY) and the kernel (template instantiation) include.
constexpr uint64_t KEY_PER_CHANNEL = QUANTIZE_PER_CHANNEL;
constexpr uint64_t KEY_PER_TENSOR = QUANTIZE_PER_TENSOR;

constexpr size_t INPUT_IDX_X = 0;
constexpr size_t INPUT_IDX_SCALES = 1;
constexpr size_t INPUT_IDX_ZERO_POINTS = 2;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t ATTR_IDX_AXIS = 1;

constexpr int64_t DEFAULT_AXIS = 1;
constexpr int64_t RESERVE_UB = 16 * 1024;
constexpr int64_t ALIGN_ELEMS = 64;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t CALC_BUF_BYTES_PER_ELEM = 4; // fp32 compute buffer

// x folded around the quantization axis
struct QuantizeShape {
    int64_t total = 1;
    int64_t dimBefore = 1;
    int64_t dimAxis = 1;
    int64_t dimAfter = 1;
};

// how the work is spread over the cores
struct QuantizeSplit {
    uint64_t tilingKey = KEY_PER_TENSOR;
    uint32_t numCore = 1;
    int64_t channelNum = 1;
    int64_t rowLen = 1;
    int64_t totalRows = 1;
    int64_t blockFactor = 0;
    int64_t blockTailFactor = 0;
};

inline int64_t CeilDiv(int64_t a, int64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }

inline int64_t FloorAlign(int64_t a, int64_t b) { return (b == 0) ? a : (a / b) * b; }

inline int64_t DtypeSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1;
        default:
            return 4;
    }
}

// Normalize axis into [0, rank) and fold x to [dimBefore, dimAxis, dimAfter].
inline QuantizeShape FoldShape(const gert::Shape& xShape, int64_t axis)
{
    QuantizeShape shape;
    int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    if (rank <= 0) {
        return shape; // scalar x: total = 1, all dims = 1
    }
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0) {
        axis = 0;
    }
    if (axis >= rank) {
        axis = rank - 1;
    }

    for (int64_t i = 0; i < rank; ++i) {
        int64_t d = xShape.GetDim(i);
        shape.total *= d;
        if (i < axis) {
            shape.dimBefore *= d;
        } else if (i == axis) {
            shape.dimAxis = d;
        } else {
            shape.dimAfter *= d;
        }
    }
    return shape;
}

// UB tile length in elements: inX(bufNum*x) + outY(bufNum*y) + calc(fp32).
// The scale is applied as a scalar reciprocal Muls, so no per-element scale-broadcast buffer is needed.
inline int64_t CalcBaseLen(ge::DataType xDtype, ge::DataType yDtype, int64_t ubSize)
{
    int64_t perElemBytes = BUFFER_NUM * (DtypeSize(xDtype) + DtypeSize(yDtype)) + CALC_BUF_BYTES_PER_ELEM;
    int64_t available = ubSize - RESERVE_UB;
    int64_t baseLen = FloorAlign(available / perElemBytes, ALIGN_ELEMS);
    return (baseLen < ALIGN_ELEMS) ? ALIGN_ELEMS : baseLen;
}

// per-tensor splits elements over cores; per-channel splits output rows.
inline QuantizeSplit CalcSplit(const QuantizeShape& shape, int64_t scalesSize, int64_t coreNum)
{
    QuantizeSplit split;
    if (shape.total == 0) {
        return split; // empty tensor: a single no-op core
    }

    if (scalesSize == 1) {
        split.tilingKey = KEY_PER_TENSOR;
        split.blockFactor = CeilDiv(shape.total, coreNum);
        if (split.blockFactor <= 0) {
            split.blockFactor = shape.total;
        }
        split.numCore = static_cast<uint32_t>(CeilDiv(shape.total, split.blockFactor));
        split.blockTailFactor = shape.total - split.blockFactor * (static_cast<int64_t>(split.numCore) - 1);
        return split;
    }

    split.tilingKey = KEY_PER_CHANNEL;
    split.channelNum = shape.dimAxis;
    split.rowLen = shape.dimAfter;
    split.totalRows = shape.dimBefore * shape.dimAxis;
    split.blockFactor = CeilDiv(split.totalRows, coreNum);
    if (split.blockFactor <= 0) {
        split.blockFactor = split.totalRows;
    }
    split.numCore = static_cast<uint32_t>(CeilDiv(split.totalRows, split.blockFactor));
    split.blockTailFactor = split.totalRows - split.blockFactor * (static_cast<int64_t>(split.numCore) - 1);
    return split;
}

// Kernel selection keys on (x, scales, y) dtypes only; the optional zero_points dtype is not part of the
// key, so the dispatched binary's compile-time zero_points type is unreliable. Carry the real runtime
// zero_points dtype to the kernel so it reads the buffer with the correct element size.
inline uint32_t GetZeroPointsDtype(gert::TilingContext* context, bool hasZeroPoint)
{
    if (!hasZeroPoint) {
        return 0U;
    }
    auto zeroPointsDesc = context->GetOptionalInputDesc(INPUT_IDX_ZERO_POINTS);
    return (zeroPointsDesc == nullptr) ? 0U : static_cast<uint32_t>(zeroPointsDesc->GetDataType());
}

inline void FillTilingData(QuantizeTilingData& tilingData, const QuantizeShape& shape, const QuantizeSplit& split,
                           bool hasZeroPoint, uint32_t zpDtype, int64_t baseLen)
{
    tilingData.set_numCore(split.numCore);
    tilingData.set_hasZeroPoint(hasZeroPoint ? 1U : 0U);
    tilingData.set_channelNum(split.channelNum);
    tilingData.set_rowLen(split.rowLen);
    tilingData.set_totalRows(split.totalRows);
    tilingData.set_totalElems(shape.total);
    tilingData.set_blockFactor(split.blockFactor);
    tilingData.set_blockTailFactor(split.blockTailFactor);
    tilingData.set_baseLen(baseLen);
    tilingData.set_zpDtype(zpDtype);
}
} // namespace

static ge::graphStatus Tiling4Quantize(gert::TilingContext* context)
{
    auto compileInfo = context->GetCompileInfo<QuantizeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    int64_t coreNum = compileInfo->coreNum;
    int64_t ubSize = static_cast<int64_t>(compileInfo->ubSize);
    OP_CHECK_IF(
        (coreNum <= 0 || ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Quantize tiling: bad platform info coreNum=%ld ubSize=%ld", coreNum, ubSize),
        return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto scalesShapePtr = context->GetInputShape(INPUT_IDX_SCALES);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalesShapePtr);
    auto xDesc = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto yDesc = context->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_AXIS);
    int64_t axis = (axisPtr == nullptr) ? DEFAULT_AXIS : *axisPtr;
    bool hasZeroPoint = (context->GetOptionalInputShape(INPUT_IDX_ZERO_POINTS) != nullptr);

    QuantizeShape shape = FoldShape(xShapePtr->GetStorageShape(), axis);
    int64_t scalesSize = scalesShapePtr->GetStorageShape().GetShapeSize();
    QuantizeSplit split = CalcSplit(shape, scalesSize, coreNum);
    int64_t baseLen = CalcBaseLen(xDesc->GetDataType(), yDesc->GetDataType(), ubSize);

    QuantizeTilingData tilingData;
    FillTilingData(tilingData, shape, split, hasZeroPoint, GetZeroPointsDtype(context, hasZeroPoint), baseLen);

    context->SetBlockDim(split.numCore);
    // Compose the templated tiling key from the perMode template argument (see op_kernel/quantize_tiling_key.h).
    context->SetTilingKey(GET_TPL_TILING_KEY(split.tilingKey));
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0U;

    OP_LOGD(context->GetNodeName(),
            "Quantize tiling: key=%lu numCore=%u hasZp=%d channelNum=%ld rowLen=%ld totalRows=%ld total=%ld "
            "blockFactor=%ld blockTailFactor=%ld baseLen=%ld",
            split.tilingKey, split.numCore, static_cast<int>(hasZeroPoint), split.channelNum, split.rowLen,
            split.totalRows, shape.total, split.blockFactor, split.blockTailFactor, baseLen);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4Quantize(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<QuantizeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF((compileInfo->coreNum <= 0 || compileInfo->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "Quantize TilingParse: bad platform coreNum=%ld ubSize=%lu",
                        compileInfo->coreNum, compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Quantize).Tiling(Tiling4Quantize).TilingParse<QuantizeCompileInfo>(TilingParse4Quantize);

} // namespace optiling
