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
 * \file dynamic_quant_update_scatter_v2_tiling.cpp
 * \brief
 */
#include "dynamic_quant_update_scatter_v2_tiling.h"
#include <algorithm>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "util/math_util.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t INDICES_INDEX = 1;
constexpr uint32_t VAR_INDEX = 2;
constexpr uint32_t VAR_SCALE_INDEX = 3;
constexpr uint32_t VAR_OFFSET_INDEX = 4;
constexpr uint32_t VAR_OUT_INDEX = 0;
constexpr uint32_t VAR_SCALE_OUT_INDEX = 1;
constexpr uint32_t VAR_OFFSET_OUT_INDEX = 2;
constexpr uint32_t FP16_DB_UB_SIZE = 13;
constexpr uint32_t FP16_UB_SIZE = 11;
constexpr uint32_t BF16_DB_UB_SIZE = 13;
constexpr uint32_t BF16_UB_SIZE = 11;

constexpr uint32_t SYS_WORKSPACE_SIZE = static_cast<uint32_t>(16) * 1024 * 1024;
constexpr uint32_t COMPARE_INT = 255;
constexpr uint32_t NUM_SIXTEEN = 16;
constexpr uint32_t RESERVED_LENGTH = static_cast<uint32_t>(2 * 1024);
constexpr int64_t TILING_KEY_BF16 = 0;
constexpr int64_t TILING_KEY_HALF = 1;
constexpr int64_t TILING_KEY_DB_BF16 = 2;
constexpr int64_t TILING_KEY_DB_HALF = 3;
constexpr int64_t EVEN_FACTOR = 2;
constexpr size_t INPUT_X_DIM_NUM = 3;

class DynamicQuantUpdateScatterV2Tiling {
public:
    DynamicQuantUpdateScatterV2Tiling() = default;
    ~DynamicQuantUpdateScatterV2Tiling() = default;
    DynamicQuantUpdateScatterV2TilingData tilingData;
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);

private:
    uint32_t AlignUp(uint32_t base, uint32_t a) const;
    void SetTilingKey(gert::TilingContext* context, ge::DataType dataType, bool useDb) const;
    ge::graphStatus CheckInputDtype(gert::TilingContext* context) const;
    ge::graphStatus CheckOutputDtype(gert::TilingContext* context);
    ge::graphStatus CheckOpInputShape(gert::TilingContext* context) const;
    ge::graphStatus CheckAttrs(gert::TilingContext* context);
    ge::graphStatus CheckOpShape(gert::TilingContext* context);
    ge::graphStatus CheckOpDim(const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape2Dim) const;
    ge::graphStatus CheckOpParams(gert::TilingContext* context);
    ge::graphStatus SetTilingData(gert::TilingContext* context, ge::DataType xDtype);
    void CalculateMaxUbSizePerRow(ge::DataType xDtype);
    ge::graphStatus GetCompileInfo(gert::TilingContext* context);

private:
    uint32_t vectorCoreNum{0};
    uint64_t ubSize{0};

    uint32_t coreNum;
    uint32_t rowLen;
    uint32_t headCoreNum;    // head core number
    uint32_t rowPerHeadCore; // head core row number: head core比 tail core多一row
    uint32_t rowPerTailCore; // tail core row number
    uint32_t multiRowNumHeadCore;
    uint32_t multiRowNumTailCore;

    uint32_t rowNum;
    uint32_t ubPerRowDB; // w DB
    uint32_t ubPerRow;   // w/o DB

    uint32_t batchSize; // dim0
    uint32_t dstSeqLen; // var dim1

    int32_t varDtype;
};

uint32_t DynamicQuantUpdateScatterV2Tiling::AlignUp(uint32_t base, uint32_t a) const
{
    if (unlikely(base == static_cast<uint32_t>(0))) {
        return a;
    }

    return static_cast<uint32_t>(a + base - static_cast<uint32_t>(1)) / base * base;
}

void DynamicQuantUpdateScatterV2Tiling::SetTilingKey(gert::TilingContext* context, ge::DataType dataType, bool useDb) const
{
    if (useDb) {
        if (dataType == ge::DT_BF16) {
            context->SetTilingKey(TILING_KEY_DB_BF16);
            return;
        }
        context->SetTilingKey(TILING_KEY_DB_HALF);
        return;
    }

    if (dataType == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_BF16);
        return;
    }
    context->SetTilingKey(TILING_KEY_HALF);
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::CheckOpDim(
    const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape2Dim) const
{
    for (uint32_t i = 0; i < shape2Dim; i++) {
        if (shape1->GetStorageShape().GetDim(i) != shape2->GetStorageShape().GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::CheckOpInputShape(gert::TilingContext* context) const
{
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum != INPUT_X_DIM_NUM) {
        OP_LOGE(context, "x shape dimension is only support 3!");
        return ge::GRAPH_FAILED;
    }
    int64_t xDimLast = xShape->GetStorageShape().GetDim(xDimNum - 1);

    if (varDtype == ge::DT_INT4) {
        OP_CHECK_IF(
            (xDimLast % EVEN_FACTOR),
            OP_LOGE(context, "if var datatype is int4, the last dim(%ld) of x should be an even number", xDimLast),
            return ge::GRAPH_FAILED);
    }

    // check indices
    auto indicesShape = context->GetInputShape(INDICES_INDEX);
    size_t indicesDimNum = indicesShape->GetStorageShape().GetDimNum();
    if (indicesDimNum != static_cast<size_t>(1)) {
        OP_LOGE(context, "indices shape dimension should be 1");
        return ge::GRAPH_FAILED;
    }

    auto varShape = context->GetInputShape(VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);

    auto scaleShape = context->GetInputShape(VAR_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    size_t scaleDimNum = scaleShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (CheckOpDim(varShape, scaleShape, scaleDimNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "scale shape is wrong, must be same as var first 2 dim."), return ge::GRAPH_FAILED);

    auto offsetShape = context->GetInputShape(VAR_OFFSET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetShape);
    size_t offsetDimNum = offsetShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (CheckOpDim(varShape, offsetShape, offsetDimNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "offset shape is wrong, must be same as var first 2 dim."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::CheckOpShape(gert::TilingContext* context)
{
    OP_CHECK_IF(
        (CheckOpInputShape(context) != ge::GRAPH_SUCCESS), OP_LOGE(context, "input shape check failed!"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::CheckOutputDtype(gert::TilingContext* context)
{
    auto varDesc = context->GetOutputDesc(VAR_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    varDtype = varDesc->GetDataType();
    if (varDtype != ge::DataType::DT_INT4) {
        OP_LOGE(context, "var dtype is only support int4!");
        return ge::GRAPH_FAILED;
    }

    auto scaleDesc = context->GetOutputDesc(VAR_SCALE_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
    auto scaleDtype = scaleDesc->GetDataType();
    if (scaleDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context, "scale dtype is only support fp32!");
        return ge::GRAPH_FAILED;
    }

    auto offsetDesc = context->GetOutputDesc(VAR_OFFSET_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetDesc);
    auto offsetDtype = scaleDesc->GetDataType();
    if (offsetDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context, "offset dtype is only support fp32!");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::CheckInputDtype(gert::TilingContext* context) const
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    if (xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16) {
        OP_LOGE(context, "x dtype is only support fp16 or bf16.");
        return ge::GRAPH_FAILED;
    }

    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    auto indicesDtype = indicesDesc->GetDataType();
    if (indicesDtype != ge::DataType::DT_INT32) {
        OP_LOGE(context, "indices dtype is only support int32.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::CheckOpParams(gert::TilingContext* context)
{
    OP_CHECK_IF(
        (CheckInputDtype(context) != ge::GRAPH_SUCCESS), OP_LOGE(context, "x or indices dtype is invalid"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOutputDtype(context) != ge::GRAPH_SUCCESS), OP_LOGE(context, "op output dtype is invalid"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpShape(context) != ge::GRAPH_SUCCESS), OP_LOGE(context, "input or output shape is invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::SetTilingData(gert::TilingContext* context, ge::DataType xDtype)
{
    uint64_t maxUseUbSize = ubSize - RESERVED_LENGTH;                     // Bytes
    uint32_t maxRow = static_cast<uint32_t>(maxUseUbSize / ubPerRow);     // with out DB max row number
    uint32_t maxRowDB = static_cast<uint32_t>(maxUseUbSize / ubPerRowDB); // with DB max row number

    OP_CHECK_IF((maxRowDB == 0), OP_LOGE(context, "last dim size exceed!"), return ge::GRAPH_FAILED);

    // copy in multi with DB
    bool useDb = true;
    uint32_t multiRowNum = maxRowDB;

    // copy in once without DB
    if (maxRow > rowPerHeadCore) {
        useDb = false;
        multiRowNum = maxRow;
    }

    SetTilingKey(context, xDtype, useDb);
    tilingData.set_coreNum(coreNum);
    tilingData.set_rowLen(rowLen);
    tilingData.set_headCoreNum(headCoreNum);
    tilingData.set_rowPerHeadCore(rowPerHeadCore);
    tilingData.set_rowPerTailCore(rowPerTailCore);
    tilingData.set_multiRowNumHeadCore(std::min({COMPARE_INT, multiRowNum, rowPerHeadCore}));
    tilingData.set_multiRowNumTailCore(std::min({COMPARE_INT, multiRowNum, rowPerTailCore}));
    tilingData.set_batchSize(batchSize);
    tilingData.set_dstSeqLen(dstSeqLen);

    return ge::GRAPH_SUCCESS;
}

/**
 * Calculate the maximum ub space required for each row
 * @param context: ge::TilingContext
 */
void DynamicQuantUpdateScatterV2Tiling::CalculateMaxUbSizePerRow(ge::DataType xDtype)
{
    // Align RowLen BS = 1
    uint32_t alignedRowLen = AlignUp(NUM_SIXTEEN, rowLen);
    if (xDtype == ge::DT_BF16) {
        ubPerRowDB = BF16_DB_UB_SIZE * alignedRowLen;
        ubPerRow = BF16_UB_SIZE * alignedRowLen;
    } else {
        ubPerRowDB = FP16_DB_UB_SIZE * alignedRowLen;
        ubPerRow = FP16_UB_SIZE * alignedRowLen;
    }
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::GetCompileInfo(gert::TilingContext* context)
{
    auto compileInfo =context->GetCompileInfo<DynamicQuantUpdateScatterV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    vectorCoreNum = compileInfo->vectorCoreNum;
    ubSize = compileInfo->ubSize;
    OP_CHECK_IF(
        (vectorCoreNum <= 0 || ubSize <= 0),
        OP_LOGE(context, "RunFusionKernelTiling GetCompileInfo Failed, coreNum:%u, ubSize:%lu.", vectorCoreNum, ubSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantUpdateScatterV2Tiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    OP_CHECK_IF(
        (GetCompileInfo(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "RunFusionKernelTiling GetCompileInfo failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckOpParams(context) != ge::GRAPH_SUCCESS), OP_LOGE(context, "RunFusionKernelTiling CheckOpParams failed."),
        return ge::GRAPH_FAILED);

    // 合并B*S
    const gert::StorageShape* xShape = context->GetInputShape(X_INDEX);
    size_t dimNum = xShape->GetStorageShape().GetDimNum() - 1;
    rowLen = xShape->GetStorageShape().GetDim(dimNum);
    uint64_t tempRowNum = 1;
    for (size_t i = 0; i < dimNum; i++) {
        tempRowNum *= xShape->GetStorageShape().GetDim(i);
    }
    rowNum = tempRowNum;

    const gert::StorageShape* varShape = context->GetOutputShape(VAR_OUT_INDEX);
    batchSize = varShape->GetStorageShape().GetDim(0);
    dstSeqLen = varShape->GetStorageShape().GetDim(1);

    rowPerHeadCore = static_cast<uint32_t>(rowNum + vectorCoreNum - static_cast<uint32_t>(1)) / vectorCoreNum;
    coreNum = static_cast<uint32_t>(rowNum + rowPerHeadCore - static_cast<uint32_t>(1)) / rowPerHeadCore;
    headCoreNum = static_cast<uint32_t>(coreNum - static_cast<uint32_t>(1));
    rowPerTailCore = rowNum - headCoreNum * rowPerHeadCore;

    auto x = context->GetInputDesc(X_INDEX);
    CalculateMaxUbSizePerRow(x->GetDataType()); // per row
    SetTilingData(context, x->GetDataType());

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = SYS_WORKSPACE_SIZE;
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTilingContext(gert::TilingContext* context)
{
    if (context == nullptr || (context->GetRawTilingData() == nullptr) ||
        (context->GetRawTilingData()->GetData() == nullptr) || (context->GetWorkspaceSizes(1) == nullptr) ||
        context->GetWorkspaceNum() <= 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForDynamicQuantUpdateScatterV2(gert::TilingContext* context)
{
    if (CheckTilingContext(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    static thread_local DynamicQuantUpdateScatterV2Tiling tiling;
    auto ret = tiling.RunFusionKernelTiling(context);
    return ret;
}

static ge::graphStatus TilingPrepareForDynamicQuantUpdateScatterV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DynamicQuantUpdateScatterV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF(
        (compileInfo->vectorCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context, "DynamicQuantUpdateScatterV2 GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%lu.",
            compileInfo->vectorCoreNum, compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "GetCoreNum:%d, ubSize:%lu", compileInfo->vectorCoreNum, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicQuantUpdateScatterV2)
    .Tiling(TilingForDynamicQuantUpdateScatterV2)
    .TilingParse<DynamicQuantUpdateScatterV2CompileInfo>(TilingPrepareForDynamicQuantUpdateScatterV2);
} // namespace optiling