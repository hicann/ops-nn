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
 * \file anti_mx_quant_tiling_arch35.cpp
 * \brief Main tiling entry for AntiMxQuant operator.
 */

#include "anti_mx_quant_tiling_arch35.h"
#include "quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tilingdata.h"
#include "quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_struct.h"
#include <cmath>
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

using namespace std;
using namespace ge;
using namespace Ops::Base;
using namespace AntiMxQuantOp;

namespace optiling {

constexpr int64_t INDEX_ATTR_AXIS = 0;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 1;
constexpr int64_t DIGIT_ZERO = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_DIM_NUM = 7;
constexpr int64_t MAX_MXSCALE_DIM_NUM = 8;
constexpr int64_t MIN_MXSCALE_DIM_NUM = 2;
constexpr size_t WORKSPACE_SIZE = 0;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E5M2,
                                                        ge::DT_FLOAT8_E4M3FN};
const std::set<ge::DataType> MXSCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};

inline static ge::graphStatus AntiMxQuantSetTilingData(gert::TilingContext* context, AntiMxQuantTilingData& tilingData)
{
    uint64_t tilingDataSize = sizeof(tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    auto rawTilingData = context->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), reinterpret_cast<void*>(&tilingData),
                           tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext* context, AntiMxQuantTilingData& tilingData)
{
    OP_LOGI(context->GetNodeName(),
            "tilingData is ubSize:%ld, dstType:%ld, totalCoreNum:%ld, usedCoreNum:%ld, "
            "rowTileNum:%ld, colTileNum:%ld, rowNum:%ld, colNum:%ld, colNormalBlockNum:%ld, colTailLen:%ld, "
            "rowNormalBlockNum:%ld, rowTailLen:%ld, maxUbBlockNum:%ld",
            tilingData.ubSize, tilingData.dstType, tilingData.totalCoreNum, tilingData.usedCoreNum,
            tilingData.rowTileNum, tilingData.colTileNum, tilingData.rowNum, tilingData.colNum,
            tilingData.colNormalBlockNum, tilingData.colTailLen, tilingData.rowNormalBlockNum, tilingData.rowTailLen,
            tilingData.maxUbBlockNum);
}

static ge::graphStatus GetAttr(const gert::TilingContext* context, AntiMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context->GetNodeName(), "GetAttr begin.");
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrAxis);
    tilingParam.axis = static_cast<int64_t>(*attrAxis);
    OP_LOGD(context->GetNodeName(), "The attr axis is %ld", tilingParam.axis);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    tilingParam.dstType = static_cast<int64_t>(*attrDstType);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    int checkDstType = static_cast<int>(*attrDstType);

    // dst_type: 0=FP32, 1=FP16, 27=BF16
    OP_CHECK_IF((yDtype == ge::DT_FLOAT && checkDstType != 0) || (yDtype == ge::DT_FLOAT16 && checkDstType != 1) ||
                    (yDtype == ge::DT_BF16 && checkDstType != 27),
                OP_LOGE(context->GetNodeName(),
                        "y's data type[%s] and dst_type[%d] is not corresponded, y's data type: "
                        "FLOAT32/FLOAT16/BFLOAT16 correspond to dst_type: 0/1/27, please check.",
                        Ops::Base::ToString(static_cast<ge::DataType>(yDtype)).c_str(), checkDstType),
                return ge::GRAPH_FAILED);

    tilingParam.blockSize = BLOCK_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "CheckDtype begin.");
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(context->GetNodeName(),
                "Input x's data type[%s] only support FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E5M2/FLOAT8_E4M3FN currently, "
                "please check.",
                Ops::Base::ToString(static_cast<ge::DataType>(xDtype)).c_str()),
        return ge::GRAPH_FAILED);

    auto inputMxscalePtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputMxscalePtr);
    auto mxscaleDtype = inputMxscalePtr->GetDataType();
    OP_CHECK_IF(MXSCALE_SUPPORT_DTYPE_SET.count(mxscaleDtype) == 0,
                OP_LOGE(context->GetNodeName(),
                        "Input mxscale's data type[%s] only support FLOAT8_E8M0 currently, please check.",
                        Ops::Base::ToString(static_cast<ge::DataType>(mxscaleDtype)).c_str()),
                return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
                OP_LOGE(context->GetNodeName(),
                        "Output y's data type[%s] only support FLOAT16/BFLOAT16/FLOAT32 currently, please check.",
                        Ops::Base::ToString(static_cast<ge::DataType>(yDtype)).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
static inline uint64_t GetRemainder(uint64_t num, T div)
{
    return div == 0 ? div : num % div;
}

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static ge::graphStatus CheckShape(const gert::TilingContext* context, const AntiMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context->GetNodeName(), "CheckShape begin.");
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape.GetDimNum() < 1 || xShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%zu] should be in [1, 7].", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Check FP4 constraint: last dimension must be even (each byte holds 2 FP4 values)
    auto inputXPtr = context->GetInputDesc(0);
    auto xDtype = inputXPtr->GetDataType();
    if ((xDtype == ge::DT_FLOAT4_E2M1 || xDtype == ge::DT_FLOAT4_E1M2) &&
        (xShape.GetDim(xShape.GetDimNum() - 1) % DIGIT_TWO != 0)) {
        OP_LOGE(context->GetNodeName(), "For FP4 input, the last dimension (colNum=%ld) must be even.",
                xShape.GetDim(xShape.GetDimNum() - 1));
        return ge::GRAPH_FAILED;
    }

    int64_t axis = tilingParam.axis >= 0 ? tilingParam.axis : tilingParam.axis + xShape.GetDimNum();
    int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(axis != dimNum - 1,
                OP_LOGE(context->GetNodeName(),
                        "The attr axis[%ld] is invalid for input rank[%zu], the axis only supports the tail axis "
                        "currently, please check.",
                        tilingParam.axis, xShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE(context->GetNodeName(), "The shape of output y must be same with shape of input x, please check."),
        return ge::GRAPH_FAILED);

    auto mxScaleShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxScaleShapePtr);
    auto mxScaleShape = mxScaleShapePtr->GetStorageShape();

    OP_CHECK_IF(
        mxScaleShape.GetDimNum() < MIN_MXSCALE_DIM_NUM || mxScaleShape.GetDimNum() > MAX_MXSCALE_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Input mxscale rank[%zu] should be in [2, 8].", mxScaleShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    // Construct expected mxscale shape:
    // mxscale.shape[axis] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2
    // mxscale.shape[-1] = 2
    // mxscale rank = x rank + 1
    auto expectedMxScaleShape = xShape;
    int64_t axisScaledSize = (Ops::Base::CeilDiv(xShape.GetDim(axis), BLOCK_SIZE) + DIGIT_ONE) / DIGIT_TWO;
    expectedMxScaleShape.SetDim(axis, axisScaledSize);
    expectedMxScaleShape.AppendDim(DIGIT_TWO);

    OP_CHECK_IF(expectedMxScaleShape != mxScaleShape,
                OP_LOGE(context->GetNodeName(),
                        "The shape of input mxscale %s is incorrect, expected shape is %s, please check.",
                        Shape2String(mxScaleShape).c_str(), Shape2String(expectedMxScaleShape).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatInfo(const gert::TilingContext* context, AntiMxQuantTilingParam& tilingParam)
{
    OP_LOGD(context->GetNodeName(), "GetPlatInfo begin.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParam.totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParam.ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    tilingParam.vfLen = Ops::Base::GetVRegSize(context);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4AntiMxQuant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4AntiMxQuant running begin.");

    AntiMxQuantTilingParam tilingParam;

    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The data type check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The attr get failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The shape check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetPlatInfo(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetPlatInfo failed."), return ge::GRAPH_FAILED);

    // AntiMxQuant only supports tail axis with blockSize=32 currently
    AntiMxQuantTailAxisTiling tailAxisTiling(context, tilingParam);
    return tailAxisTiling.DoTiling();
}

ge::graphStatus TilingPrepare4AntiMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AntiMxQuant entering.");
    return ge::GRAPH_SUCCESS;
}

// Register tiling interface of the AntiMxQuant op.
IMPL_OP_OPTILING(AntiMxQuant).Tiling(Tiling4AntiMxQuant).TilingParse<AntiMxQuantCompileInfo>(TilingPrepare4AntiMxQuant);

} // namespace optiling
