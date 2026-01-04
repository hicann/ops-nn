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
 * \file dynamic_quant_tiling.cpp
 * \brief
 */
#include "dynamic_quant_tiling.h"
#include "dynamic_quant_tiling_310P.h"
#include "dynamic_quant_tiling_310P.h"
#include "dynamic_quant_tiling_arch35.h"

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

constexpr uint32_t OUTPUT_NUM_DYNAMIC_QUANT = 2;
constexpr uint32_t OUTPUT_NUM_DYNAMIC_QUANT_V2 = 3;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t SMOOTH_INDEX = 1;
constexpr uint32_t GROUP_INDEX = 2;
constexpr uint32_t DST_TYPE_ATTR_INDEX = 0;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t SCALE_INDEX = 1;
constexpr uint32_t OFFSET_INDEX = 2;
constexpr uint32_t ONE = 1;
constexpr uint32_t FP16_DB_SMOOTH_UB_SIZE = 19;
constexpr uint32_t FP16_DB_UB_SIZE = 13;
constexpr uint32_t FP16_SMOOTH_UB_SIZE = 17;
constexpr uint32_t FP16_UB_SIZE = 11;

constexpr uint32_t MOE_UB_SIZE = 17;

constexpr uint32_t BF16_DB_SMOOTH_UB_SIZE = 19;
constexpr uint32_t BF16_DB_UB_SIZE = 13;
constexpr uint32_t BF16_SMOOTH_UB_SIZE = 17;
constexpr uint32_t BF16_UB_SIZE = 11;

constexpr uint32_t SYS_WORKSPACE_SIZE = 16777216;
constexpr uint32_t COMPARE_INT = 255;
constexpr uint32_t B_ASCEND_HW_NUMCORES = 40;
constexpr uint32_t P_ASCEND_HUW_NUMCORE = 8;
constexpr uint32_t NUM_SIXTEEN = 16;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t ALIGEN_EIGHT = 8;
constexpr uint32_t SEVEN = 7;
constexpr uint32_t RESERVED_LENGTH = 1024;
constexpr uint32_t UB_ALIG_NUM = 32;
constexpr uint32_t FLOAT_NUM_ONE_RPT = 128;
constexpr uint32_t MAX_EXPERT_NUM = 1024;
constexpr uint32_t MOE_SMOOTH_NUM = 2;
constexpr int64_t TILING_KEY_BF16 = 0;
constexpr int64_t TILING_KEY_HALF = 1;
constexpr int64_t TILING_KEY_DB_BF16 = 2;
constexpr int64_t TILING_KEY_DB_HALF = 3;
constexpr int64_t TILING_KEY_LARGE_SHAPE = 6;
constexpr int64_t TILING_KEY_MOE = 7;
constexpr int64_t TILING_KEY_MOE_LARGE_SHAPE = 8;
constexpr int64_t EVEN_FACTOR = 2;

static map<const ge::DataType, const uint32_t> g_dTypeLen = {{ge::DT_INT32, 4}, {ge::DT_INT64, 8}};

template <uint32_t base, typename T = uint32_t>
auto AlignUp(T a) -> T
{
    return (a + base - 1) / base * base;
}

class DynamicQuantTiling
{
public:
    DynamicQuantTiling() = default;
    ~DynamicQuantTiling() = default;
    DynamicQuantTilingData tilingData;
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);

private:
    void SetTilingKey(gert::TilingContext* context, ge::DataType dataType, bool useDb);
    ge::graphStatus CheckInputDtype(gert::TilingContext* context);
    ge::graphStatus CheckOutputDtype(gert::TilingContext* context);
    ge::graphStatus CheckOpInpuShape(gert::TilingContext* context);
    ge::graphStatus CheckOpOutputShape(gert::TilingContext* context);
    ge::graphStatus CheckAttrs(gert::TilingContext* context);
    ge::graphStatus CheckOpShape(gert::TilingContext* context);
    ge::graphStatus CheckOpDim(
        const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape1Dim, uint32_t shape2Dim);
    ge::graphStatus CheckOpParams(gert::TilingContext* context);
    void ResetLargeTilingParams();
    void SetTilingData(gert::TilingContext* context, ge::DataType xDtype);
    void CalculateMaxUbSizePerRow(ge::DataType xDtype);
    ge::graphStatus GetCompileInfo(gert::TilingContext* context);

private:
    uint32_t vectorCoreNum{0};
    uint64_t ubSize{0};

    uint32_t coreNum;
    uint32_t rowLen;
    uint32_t headCoreNum;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t multiRowNumHeadCore;
    uint32_t multiRowNumTailCore;

    uint32_t rowNum;
    uint32_t rowNumPerMinTask;
    uint32_t scaleNumPerMinTask;
    uint32_t rowNumPerTask;
    uint32_t taskNum;
    uint32_t wholeTaskNum;
    uint32_t ubPerRow;
    uint32_t ubPerRowNew;

    uint32_t innerLoopEle = 0;
    uint32_t innerLoopTimes = 0;
    uint32_t innerLoopTail = 0;

    uint32_t groupNum = 0;
    uint32_t groupDtypeSize = 0;
    bool hasSmooth = false;

    int32_t yDtype;
};

void DynamicQuantTiling::SetTilingKey(gert::TilingContext* context, ge::DataType dataType, bool useDb)
{
    if (groupNum > 0U) {
        if (innerLoopTimes > 0U) {
            context->SetTilingKey(TILING_KEY_MOE_LARGE_SHAPE);
        } else {
            context->SetTilingKey(TILING_KEY_MOE);
        }
    } else {
        if (innerLoopTimes > 0U) {
            context->SetTilingKey(TILING_KEY_LARGE_SHAPE);
            return;
        }
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
}

ge::graphStatus DynamicQuantTiling::CheckOpDim(
    const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape1Dim, uint32_t shape2Dim)
{
    if (shape1Dim != shape2Dim) {
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < shape1Dim; i++) {
        if (shape1->GetStorageShape().GetDim(i) != shape2->GetStorageShape().GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOpInpuShape(gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum == 1U) {
        OP_LOGE(context, "x shape dimension is less than 2!");
        return ge::GRAPH_FAILED;
    }
    int64_t xDimLast = xShape->GetStorageShape().GetDim(xDimNum - 1);

    if (yDtype == ge::DT_INT4) {
        OP_CHECK_IF(
            (xDimLast % EVEN_FACTOR),
            OP_LOGE(context, "if y datatype is int4, the last dim(%ld) of x should be an even number",
                xDimLast),
            return ge::GRAPH_FAILED);
    }

    auto smoothShape = context->GetOptionalInputShape(SMOOTH_INDEX);
    if (smoothShape != nullptr) {
        auto groupShape = context->GetOptionalInputShape(GROUP_INDEX);
        if (groupShape != nullptr) {
            groupNum = groupShape->GetStorageShape().GetDim(groupShape->GetStorageShape().GetDimNum() - 1);
            OP_CHECK_IF(
                (groupNum > MAX_EXPERT_NUM),
                OP_LOGE(context, "moe expert nums exceed maximum, the maximum is 1024."),
                return ge::GRAPH_FAILED);
        }

        size_t smoothDimNum = smoothShape->GetStorageShape().GetDimNum();
        // 针对moe场景下的校验
        if (groupNum >= 1U) {
            OP_CHECK_IF(
                (smoothDimNum != MOE_SMOOTH_NUM),
                OP_LOGE(context, "In moe scenario, smooth_scales dim num must be two."),
                return ge::GRAPH_FAILED);
            int64_t smoothDimFirst = smoothShape->GetStorageShape().GetDim(0);
            if (groupNum != static_cast<uint32_t>(smoothDimFirst)) {
                OP_LOGE(context,
                    "moe expert and smooth_scales first dim is not equal! expert nums is :%u, smooth_scales is:%ld",
                    groupNum, smoothDimFirst);
                return ge::GRAPH_FAILED;
            }
        } else {
            if (smoothDimNum != 1U) {
                OP_LOGE(context, "smooth_scales dim num not one");
                return ge::GRAPH_FAILED;
            }
        }
        int64_t smoothDimLast = smoothShape->GetStorageShape().GetDim(smoothDimNum - 1);
        if (xDimLast != smoothDimLast) {
            OP_LOGE(context, "x and smooth_scales last dim is not equal! x is :%ld, smooth_scales is:%ld",
                xDimLast, smoothDimLast);
            return ge::GRAPH_FAILED;
        }
        hasSmooth = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOpOutputShape(gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(X_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    auto yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (CheckOpDim(xShape, yShape, xDimNum, yDimNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "x and y shape is inconsistency."),
        return ge::GRAPH_FAILED);

    auto scaleShape = context->GetOutputShape(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    size_t scaleDimNum = scaleShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (CheckOpDim(xShape, scaleShape, xDimNum - 1, scaleDimNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "scale shape is wrong, must be same as before x exclude the last dim."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOpShape(gert::TilingContext* context)
{
    OP_CHECK_IF(
        (CheckOpInpuShape(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "input shape check failed!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpOutputShape(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "output shape check failed!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOutputDtype(gert::TilingContext* context)
{
    auto yDesc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    yDtype = yDesc->GetDataType();
    if (yDtype != ge::DataType::DT_INT8 && yDtype != ge::DataType::DT_INT4) {
        OP_LOGE(context, "y dtype is support int8 or int4!");
        return ge::GRAPH_FAILED;
    }

    auto scaleDesc = context->GetOutputDesc(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
    auto scaleDtype = scaleDesc->GetDataType();
    if (scaleDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context, "scale dtype is only support fp32!");
        return ge::GRAPH_FAILED;
    }

    if (context->GetComputeNodeOutputNum() == OUTPUT_NUM_DYNAMIC_QUANT_V2) {
        auto offsetDesc = context->GetOutputDesc(OFFSET_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, offsetDesc);
        auto offsetDtype = scaleDesc->GetDataType();
        if (offsetDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context, "offset dtype is only support fp32!");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckInputDtype(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    if (xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16) {
        OP_LOGE(context, "x dtype is only support fp16 or bf16.");
        return ge::GRAPH_FAILED;
    }

    auto smoothDesc = context->GetOptionalInputDesc(SMOOTH_INDEX);
    if (smoothDesc != nullptr) {
        auto smoothDtype = smoothDesc->GetDataType();
        if (smoothDtype != ge::DataType::DT_FLOAT16 && smoothDtype != ge::DataType::DT_BF16) {
            OP_LOGE(context, "smooth dtype is only support fp16 or bf16.");
            return ge::GRAPH_FAILED;
        }
        if (xDtype != smoothDtype) {
            OP_LOGE(context, "x and smooth_scales dtype is not equal.");
            return ge::GRAPH_FAILED;
        }

        auto groupDesc = context->GetOptionalInputDesc(GROUP_INDEX);
        if (groupDesc != nullptr) {
            auto groupDtype = groupDesc->GetDataType();
            if (groupDtype != ge::DataType::DT_INT32) {
                OP_LOGE(context, "group dtype is only support int32.");
                return ge::GRAPH_FAILED;
            }
            groupDtypeSize = g_dTypeLen[groupDtype];
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckAttrs(gert::TilingContext* context)
{
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* dstTypePtr = attrs->GetAttrPointer<int32_t>(DST_TYPE_ATTR_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, dstTypePtr);
        int32_t dstType = *dstTypePtr;
        if (dstType != yDtype) {
            OP_LOGE(context, "dst type:%d not equal output dtype:%d", dstType, yDtype);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::CheckOpParams(gert::TilingContext* context)
{
    OP_CHECK_IF(
        (CheckInputDtype(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "x or smooth_scales dtype is invalid"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOutputDtype(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "op output dtype is invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckAttrs(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "op attrs is invalid."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpShape(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "input or output shape is invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void DynamicQuantTiling::ResetLargeTilingParams()
{
    innerLoopEle = 0U;
    innerLoopTimes = 0U;
    innerLoopTail = 0U;
    groupNum = 0U;
    hasSmooth = false;
}

void DynamicQuantTiling::SetTilingData(gert::TilingContext* context, ge::DataType xDtype)
{
    uint64_t maxUseUbSize = ubSize - RESERVED_LENGTH;
    uint32_t ubAvail = static_cast<uint32_t>(maxUseUbSize / ubPerRowNew);
    bool useDb = true;
    if (ubAvail == 0U) {
        // 这里需要除个单行最大需要的空间
        innerLoopEle = static_cast<uint32_t>(maxUseUbSize / BF16_DB_SMOOTH_UB_SIZE / FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT);
        innerLoopTimes = rowLen / innerLoopEle;
        innerLoopTail = rowLen % innerLoopEle;
        ubAvail = ONE;
    } else if (groupNum == 0U) {
        if (ubPerRow < maxUseUbSize) {
            useDb = true;
            ubAvail = std::max(ubAvail, ONE);
        } else {
            useDb = false;
            ubAvail = ONE;
        }
    } else {
        useDb = false;
        ubAvail = std::max(ubAvail, ONE);
    }
    SetTilingKey(context, xDtype, useDb);
    tilingData.set_coreNum(coreNum);
    tilingData.set_rowLen(rowLen);
    tilingData.set_headCoreNum(headCoreNum);
    tilingData.set_rowPerHeadCore(rowPerHeadCore);
    tilingData.set_rowPerTailCore(rowPerTailCore);
    tilingData.set_multiRowNumHeadCore(std::min({COMPARE_INT, ubAvail, rowPerHeadCore}));
    tilingData.set_multiRowNumTailCore(std::min({COMPARE_INT, ubAvail, rowPerTailCore}));
    tilingData.set_innerLoopEle(innerLoopEle);
    tilingData.set_innerLoopTimes(innerLoopTimes);
    tilingData.set_innerLoopTail(innerLoopTail);
    tilingData.set_groupNum(groupNum);
    tilingData.set_hasSmooth(hasSmooth ? 1 : 0);
    tilingData.set_ubSize(ubSize);
}

/**
 * Calculate the maximum ub space required for each row
 * @param context: ge::TilingContext
 */
void DynamicQuantTiling::CalculateMaxUbSizePerRow(ge::DataType xDtype)
{
    // Align RowLen
    uint32_t alignedRowLen = AlignUp<NUM_SIXTEEN>(rowLen);
    uint32_t alignedGroupNumLen = 0;
    if (groupNum > 0U) {
        alignedGroupNumLen = AlignUp<ALIGEN_EIGHT>(groupNum);
        ubPerRowNew = alignedGroupNumLen * groupDtypeSize + MOE_UB_SIZE * alignedRowLen;
        tilingData.set_alignGroupNum(alignedGroupNumLen);
    } else {
        if (xDtype == ge::DT_BF16) {
            if (hasSmooth) {
                ubPerRow = BF16_DB_SMOOTH_UB_SIZE * alignedRowLen;
                ubPerRowNew = BF16_SMOOTH_UB_SIZE * alignedRowLen;
            } else {
                ubPerRow = BF16_DB_UB_SIZE * alignedRowLen;
                ubPerRowNew = BF16_UB_SIZE * alignedRowLen;
            }
        } else {
            if (hasSmooth) {
                ubPerRow = FP16_DB_SMOOTH_UB_SIZE * alignedRowLen;
                ubPerRowNew = FP16_SMOOTH_UB_SIZE * alignedRowLen;
            } else {
                ubPerRow = FP16_DB_UB_SIZE * alignedRowLen;
                ubPerRowNew = FP16_UB_SIZE * alignedRowLen;
            }
        }
    }
}

ge::graphStatus DynamicQuantTiling::GetCompileInfo(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const DynamicQuantCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    vectorCoreNum = compileInfo->vectorCoreNum;
    ubSize = compileInfo->ubSize;
    OP_CHECK_IF(
        (vectorCoreNum <= 0 || ubSize <= 0),
        OP_LOGE(context, "RunFusionKernelTiling GetCompileInfo Failed, coreNum:%u, ubSize:%lu.",
            vectorCoreNum, ubSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    ResetLargeTilingParams();
    OP_CHECK_IF(
        (GetCompileInfo(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "RunFusionKernelTiling GetCompileInfo failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckOpParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "RunFusionKernelTiling CheckOpParams failed."),
        return ge::GRAPH_FAILED);
    const gert::StorageShape* xShape = context->GetInputShape(X_INDEX);

    rowLen = xShape->GetStorageShape().GetDim(xShape->GetStorageShape().GetDimNum() - 1);
    OP_CHECK_IF(
        (rowLen == 0),
        OP_LOGE(context, "RunFusionKernelTiling CheckOpParams failed."),
        return ge::GRAPH_FAILED);
    size_t dimNum = xShape->GetStorageShape().GetDimNum() - 1;
    uint64_t tempHeadCoreNum = 1;
    for (size_t i = 0; i < dimNum; i++) {
        tempHeadCoreNum *= xShape->GetStorageShape().GetDim(i);
    }
    rowNum = tempHeadCoreNum;
    // For 910B
    rowNumPerMinTask = 1U;
    scaleNumPerMinTask = 1U;
    // For 310P
    if (vectorCoreNum < B_ASCEND_HW_NUMCORES) {
        rowNumPerMinTask = (UB_ALIG_NUM + rowLen - 1U) / rowLen;
        scaleNumPerMinTask = P_ASCEND_HUW_NUMCORE;
    }
    rowNumPerTask = std::max(rowNumPerMinTask, scaleNumPerMinTask);
    wholeTaskNum = rowNum / rowNumPerTask;

    coreNum = std::max(std::min(vectorCoreNum, wholeTaskNum), ONE);
    headCoreNum = rowNum % coreNum;

    rowPerHeadCore = (rowNum + coreNum - 1U) / coreNum;
    rowPerTailCore = rowNum / coreNum;

    auto x = context->GetInputDesc(X_INDEX);
    CalculateMaxUbSizePerRow(x->GetDataType());
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

static ge::graphStatus TilingForDynamicQuant(gert::TilingContext* context)
{
    if (CheckTilingContext(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ge::graphStatus ret = ge::GRAPH_FAILED;
    switch (ascendcPlatform.GetSocVersion()) {
        case platform_ascendc::SocVersion::ASCEND910_95:
            static thread_local DynamicQuantRegbaseTiling tilingRegbase;
            ret = tilingRegbase.RunFusionKernelTiling(context);
            return ret;
        case platform_ascendc::SocVersion::ASCEND910B:
        case platform_ascendc::SocVersion::KIRINX90:
            static thread_local DynamicQuantTiling tiling;
            ret = tiling.RunFusionKernelTiling(context);
            return ret;
        case platform_ascendc::SocVersion::ASCEND910:
        case platform_ascendc::SocVersion::ASCEND310P:
            static thread_local DynamicQuantTiling310P tiling310P;
            ret = tiling310P.RunFusionKernelTiling(context);
            return ret;
        default:
            OP_LOGE(context, "DynamicQuan only supports Atlas 800I A2/A3 or  Atlas 300I Duo.");
            return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus TilingPrepareForDynamicQuant(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DynamicQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF(
        (compileInfo->vectorCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(context, "DynamicQuant GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%lu.",
            compileInfo->vectorCoreNum, compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "GetCoreNum:%d, ubSize:%lu", compileInfo->vectorCoreNum, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicQuant)
    .Tiling(TilingForDynamicQuant)
    .TilingParse<DynamicQuantCompileInfo>(TilingPrepareForDynamicQuant);
IMPL_OP_OPTILING(DynamicQuantV2)
    .Tiling(TilingForDynamicQuant)
    .TilingParse<DynamicQuantCompileInfo>(TilingPrepareForDynamicQuant);
} // namespace optiling
