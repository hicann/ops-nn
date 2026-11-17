/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swi_glu_quant_tiling_utils.h
 * \brief
 */

#ifndef SWI_GLU_QUANT_TILING_UTILS_H
#define SWI_GLU_QUANT_TILING_UTILS_H

#include "swi_glu_quant_tiling.h"
#include <string>
#include <vector>
#include "util/shape_util.h"
#include "base/err_msg.h"

namespace optiling {
constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t INPUT_SMOOTH_SCALES_INDEX = 1;
constexpr uint32_t INPUT_OFFSETS_INDEX = 2;
constexpr uint32_t INPUT_GROUPIDX_INDEX = 3;
constexpr uint32_t OUTPUT_Y_INDEX = 0;
constexpr uint32_t OUTPUT_SCALE_INDEX = 1;

constexpr uint32_t EVEN_FACTOR = 2;
constexpr uint32_t CONST_ROUR = 4;
constexpr uint32_t MAX_EXPERT_NUM = 1024;

constexpr uint32_t ALIGEN_EIGHT = 8;

constexpr uint32_t ZERO = 0;
constexpr uint32_t ONE = 1;
constexpr uint32_t TWO = 2;
constexpr uint32_t GROUP_LIST_TYPE_INDEX = 2;
constexpr uint32_t DST_TYPE_INDEX = 3;
constexpr uint32_t COMPARE_INT = 255;

constexpr uint32_t TILING_KEY_VAL_FP16 = 100;
constexpr uint32_t TILING_KEY_VAL_BF16 = 200;
constexpr uint32_t TILING_KEY_VAL_FP = 300;

constexpr uint8_t PER_TENSOR_KEY_VAL = 4;
constexpr uint8_t PER_CHANEL_KEY_VAL = 5;
constexpr uint8_t DYNAMIC_KEY_VAL = 6;

const std::string QUANT_MODE_STATIC = "static";
const std::string QUANT_MODE_DYNAMIC = "dynamic";

template <typename T>
inline auto AlignUp(T num, T div) -> decltype(num)
{
    return (div == 0) ? 0 : (num + div - 1) / div * div;
}

template <typename T>
inline auto AlignDown(T num, T div) -> decltype(num)
{
    return (div == 0) ? 0 : num / div * div;
}

template <typename T>
inline auto CeilDiv(T num, T div) -> decltype(num)
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

template <typename T>
inline auto Min(T num, T div) -> decltype(num)
{
    return num < div ? num : div;
}

inline ge::graphStatus CheckInputDtype(const gert::TilingContext* context)
{
    auto xDtype = context->GetInputDesc(INPUT_X_INDEX)->GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", std::to_string(static_cast<int>(xDtype)),
                                              "input x dtype only supports fp16/bf16/fp32.");
        return ge::GRAPH_FAILED;
    }

    auto smoothScalesDesc = context->GetOptionalInputDesc(INPUT_SMOOTH_SCALES_INDEX);
    if (smoothScalesDesc != nullptr) {
        auto smoothScalesDtype = smoothScalesDesc->GetDataType();
        if (smoothScalesDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "smooth_scales",
                                                  std::to_string(static_cast<int>(smoothScalesDtype)),
                                                  "input smoothScales dtype only supports fp32.");
            return ge::GRAPH_FAILED;
        }
    }

    auto offsetsDesc = context->GetOptionalInputDesc(INPUT_OFFSETS_INDEX);
    if (offsetsDesc != nullptr) {
        auto offsetsDtype = offsetsDesc->GetDataType();
        if (offsetsDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "offsets",
                                                  std::to_string(static_cast<int>(offsetsDtype)),
                                                  "input offsets dtype only supports fp32.");
            return ge::GRAPH_FAILED;
        }
    }

    auto groupDesc = context->GetOptionalInputDesc(INPUT_GROUPIDX_INDEX);
    if (groupDesc != nullptr) {
        auto groupDtype = groupDesc->GetDataType();
        if (groupDtype != ge::DataType::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "group_index",
                                                  std::to_string(static_cast<int>(groupDtype)),
                                                  "group dtype only supports int32.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOutputDtype(const gert::TilingContext* context)
{
    auto yDtype = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    if (yDtype != ge::DataType::DT_INT8 && yDtype != ge::DataType::DT_INT4) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "y", std::to_string(static_cast<int>(yDtype)),
                                              "y dtype only supports int8 or int4.");
        return ge::GRAPH_FAILED;
    }

    auto scaleDtype = context->GetOutputDesc(OUTPUT_SCALE_INDEX)->GetDataType();
    if (scaleDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "scale",
                                              std::to_string(static_cast<int>(scaleDtype)),
                                              "output scale dtype only supports fp32.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckAttrs(const gert::TilingContext* context, SwiGluQuantCompileInfo& compileInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        std::string quantMode = attrs->GetAttrPointer<char>(1);
        if (quantMode != QUANT_MODE_STATIC && quantMode != QUANT_MODE_DYNAMIC) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "quant_mode", quantMode,
                                                  "quantMode must be equal to static or dynamic.");
            return ge::GRAPH_FAILED;
        }
        compileInfo.curQuantMode = quantMode;

        auto isActivateLeftAttr = *(attrs->GetBool(0));
        compileInfo.activateLeft = (isActivateLeftAttr ? 1 : 0);

        auto groupListTypePtr = attrs->GetInt(GROUP_LIST_TYPE_INDEX);
        if ((groupListTypePtr == nullptr)) {
            const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                               "swi_glu_quant" :
                                               context->GetNodeName());
            const std::string msgReason = (std::string(__func__) + ": " +
                                           "The default group_list_type attribute is unavailable; check the registered "
                                           "Tiling attribute context.");
            OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
            return ge::GRAPH_FAILED;
        };
        int64_t groupListType = static_cast<int64_t>(*groupListTypePtr);
        OP_CHECK_IF(
            (groupListType != 0 && groupListType != 1),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "group_list_type",
                                                  std::to_string(groupListType), "groupListType can only be 0 or 1."),
            return ge::GRAPH_FAILED);
        compileInfo.groupListType = groupListType;

        auto dstTypePtr = attrs->GetInt(DST_TYPE_INDEX);
        if ((dstTypePtr == nullptr)) {
            const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                               "swi_glu_quant" :
                                               context->GetNodeName());
            const std::string msgReason = (std::string(__func__) + ": " +
                                           "The default dst_type attribute is unavailable; check the registered Tiling "
                                           "attribute context.");
            OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
            return ge::GRAPH_FAILED;
        };
        int64_t dstType = static_cast<int64_t>(*dstTypePtr);
        OP_CHECK_IF((dstType != ge::DT_INT8 && dstType != ge::DT_INT4),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_type", std::to_string(dstType),
                                                          "dstType can only be 2(DT_INT8) or 29(DT_INT4)."),
                    return ge::GRAPH_FAILED);
        compileInfo.dstType = dstType;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckMoeInputShpae(const gert::StorageShape* shape, uint32_t groupLen)
{
    if (groupLen > ONE) {
        uint32_t shapeDimFirst = shape->GetStorageShape().GetDim(0);
        if (groupLen != shapeDimFirst) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOpInputShape(const gert::TilingContext* context, SwiGluQuantCompileInfo& compileInfo)
{
    auto xShape = context->GetInputShape(INPUT_X_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum < TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(xDimNum),
                                                 "Input x shape dimension is less than 2.");
        return ge::GRAPH_FAILED;
    }
    int64_t xDimLast = xShape->GetStorageShape().GetDim(xDimNum - 1);
    auto yDtype = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    if (yDtype == ge::DT_INT4 && xDimLast % static_cast<int32_t>(CONST_ROUR) != 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "x", std::to_string(xDimLast),
            ("if y datatype is int4, the last dim(" + std::to_string(xDimLast) + ") of x should be divisible by 4."));
        return ge::GRAPH_FAILED;
    }
    auto smoothScalesShape = context->GetOptionalInputShape(INPUT_SMOOTH_SCALES_INDEX);
    auto offsetsShape = context->GetOptionalInputShape(INPUT_OFFSETS_INDEX);
    uint32_t groupLen = 1;
    int64_t hasGroup = 0;
    if (smoothScalesShape != nullptr) {
        auto scaleDimNum = smoothScalesShape->GetStorageShape().GetDimNum();
        compileInfo.isPerTensor = scaleDimNum == 1 ? true : false;
        auto groupShape = context->GetOptionalInputShape(INPUT_GROUPIDX_INDEX);
        if (groupShape != nullptr) {
            groupLen = static_cast<uint32_t>(
                groupShape->GetStorageShape().GetDim(groupShape->GetStorageShape().GetDimNum() - 1));
            hasGroup = 1;
            if (groupLen == ZERO || groupLen > MAX_EXPERT_NUM) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context->GetNodeName(), "group_index", std::to_string(groupLen),
                    std::string("The group_index length must be in [1,") + std::to_string(MAX_EXPERT_NUM) + "].");
                return ge::GRAPH_FAILED;
            }
        }
        // 针对moe场景下的校验
        if (CheckMoeInputShpae(smoothScalesShape, groupLen) != ge::GRAPH_SUCCESS) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "smooth_scales", std::to_string(smoothScalesShape->GetStorageShape().GetDim(0)),
                ("moe expert and smooth_scales first dim is not equal! expert nums is :" + std::to_string(groupLen)));
            return ge::GRAPH_FAILED;
        }
    }
    if (offsetsShape != nullptr) {
        if (CheckMoeInputShpae(offsetsShape, groupLen) != ge::GRAPH_SUCCESS) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "offsets", std::to_string(offsetsShape->GetStorageShape().GetDim(0)),
                ("moe expert and offsets first dim is not equal! expert nums is :" + std::to_string(groupLen)));
            return ge::GRAPH_FAILED;
        }
    }
    compileInfo.groupLength = groupLen;
    compileInfo.hasGroup = hasGroup;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOpDim(const gert::StorageShape* shape1, const gert::StorageShape* shape2,
                                  uint32_t shape1Dim, uint32_t shape2Dim)
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

inline ge::graphStatus CheckOpOutputShape(const gert::TilingContext* context, const SwiGluQuantCompileInfo& compileInfo)
{
    auto xShape = context->GetInputShape(INPUT_X_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    auto yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    if (CheckOpDim(xShape, yShape, xDimNum - 1, yDimNum - 1) != ge::GRAPH_SUCCESS) {
        // 除最后一个维度外，其他维度的shape需要一致
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x,y",
            Ops::Base::ToString(xShape->GetStorageShape()) + "," + Ops::Base::ToString(yShape->GetStorageShape()),
            "x and y must have identical dimensions before the last axis; for int8 y, x.shape[-1] must equal "
            "2*y.shape[-1].");
        return ge::GRAPH_FAILED;
    }
    // DT_INT8量化最后一个维度的shape需要满足2倍关系
    if (compileInfo.dstType == ge::DT_INT8 &&
        (xShape->GetStorageShape().GetDim(xDimNum - 1) != 2 * yShape->GetStorageShape().GetDim(yDimNum - 1))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x,y",
            Ops::Base::ToString(xShape->GetStorageShape()) + "," + Ops::Base::ToString(yShape->GetStorageShape()),
            "x and y must have identical dimensions before the last axis; for int8 y, x.shape[-1] must equal "
            "2*y.shape[-1].");
        return ge::GRAPH_FAILED;
    }
    auto scaleShape = context->GetOutputShape(OUTPUT_SCALE_INDEX);
    size_t scaleDimNum = scaleShape->GetStorageShape().GetDimNum();
    if (CheckOpDim(xShape, scaleShape, xDimNum - 1, scaleDimNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x,scale",
            Ops::Base::ToString(xShape->GetStorageShape()) + "," + Ops::Base::ToString(scaleShape->GetStorageShape()),
            "output_scale shape is wrong, must be same as before x exclude the last dim.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOpShape(gert::TilingContext* context, SwiGluQuantCompileInfo& compileInfo)
{
    if (CheckOpInputShape(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input shape check failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckOpOutputShape(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "output shape check failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOpParams(gert::TilingContext* context, SwiGluQuantCompileInfo& compileInfo)
{
    if (CheckInputDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "x or smooth_scales dtype is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (CheckOutputDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "op output dtype is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (CheckAttrs(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "op attrs is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (CheckOpShape(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input or output shape is invalid.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline bool SetTotalShape(gert::Shape& inShape, const gert::TilingContext* context, SwiGluQuantTilingData& tilingData)
{
    int32_t shapeBefore = 1;
    int32_t shapeAfter = 1;
    int32_t dimNum = inShape.GetDimNum();
    int32_t inDim = -1;

    if (inDim < -dimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(dimNum),
                                                 "The input rank must permit splitting its last axis.");
        return false;
    }
    int32_t splitDim = inDim < 0 ? dimNum + inDim : inDim;
    for (int32_t i = 0; i < splitDim; i++) {
        shapeBefore *= inShape.GetDim(i);
    }
    for (int32_t i = splitDim; i < dimNum; i++) {
        shapeAfter *= inShape.GetDim(i);
    }
    if (shapeAfter % EVEN_FACTOR != 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(shapeAfter),
                                                  "The last dimension of x must be a positive even number.");
        return false;
    }
    if (shapeAfter == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(shapeAfter),
                                                  "The last dimension of x must be a positive even number.");
        return false;
    }
    tilingData.set_rowLen(shapeBefore);
    tilingData.set_colLen(shapeAfter / EVEN_FACTOR);
    return true;
}

inline bool CalculateMaxUbSizePerRow(const gert::TilingContext* context, SwiGluQuantCompileInfo& compileInfo,
                                     SwiGluQuantTilingParam& tilingParam, SwiGluQuantTilingData& tilingData)
{
    // Align ColLen
    uint32_t colLen = tilingData.get_colLen();
    uint32_t alignedColLen = AlignUp<uint32_t>(colLen, compileInfo.block_num);
    if (alignedColLen == 0) {
        const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                           "swi_glu_quant" :
                                           context->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("The UB row-alignment calculation returned zero: colLen=") +
                                       std::to_string(colLen) +
                                       ", blockElements=" + std::to_string(compileInfo.block_num));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    }
    uint32_t ubAvail = compileInfo.dataNumSingleUb / alignedColLen;
    if (ubAvail == 0) {
        // collen超过ub可用空间大小，需要循环处理collen，大shape场景，当前不涉及，预留参数
        tilingParam.optBaseColLen = AlignDown(compileInfo.dataNumSingleUb, compileInfo.block_num);
        if (tilingParam.optBaseColLen > compileInfo.cacheLineLen) {
            tilingParam.optBaseColLen = AlignDown<uint32_t>(tilingParam.optBaseColLen, compileInfo.cacheLineLen);
        }
        ubAvail = ONE;
    } else {
        tilingParam.optBaseColLen = colLen;
        ubAvail = std::max(ubAvail, ONE);
    }

    tilingParam.optBaseRowLenHeadCore = std::min(std::min(ubAvail, tilingParam.rowLenPerHeadCore), COMPARE_INT);
    tilingParam.optBaseRowLenTailCore = std::min(std::min(ubAvail, tilingParam.rowLenPerTailCore), COMPARE_INT);
    return true;
}

inline QuantMode GetQuantMode(const SwiGluQuantCompileInfo& compileInfo)
{
    if (compileInfo.curQuantMode == QUANT_MODE_DYNAMIC) {
        return QuantMode::DYNAMIC;
    } else {
        return compileInfo.isPerTensor ? QuantMode::STATIC_PER_TENSOR : QuantMode::STATIC_PER_CHANNEL;
    }
}

inline uint32_t CalculateTilingKey(ge::DataType xDtype, const SwiGluQuantCompileInfo& compileInfo)
{
    uint32_t tilingKey = 0;
    std::unordered_map<ge::DataType, uint32_t> dTypeTilingKeyMap{
        {ge::DT_FLOAT16, TILING_KEY_VAL_FP16}, {ge::DT_BF16, TILING_KEY_VAL_BF16}, {ge::DT_FLOAT, TILING_KEY_VAL_FP}};
    std::unordered_map<QuantMode, uint8_t> quantModeTilingKeyMap{{QuantMode::STATIC_PER_TENSOR, PER_TENSOR_KEY_VAL},
                                                                 {QuantMode::STATIC_PER_CHANNEL, PER_CHANEL_KEY_VAL},
                                                                 {QuantMode::DYNAMIC, DYNAMIC_KEY_VAL}};
    QuantMode quantMode = GetQuantMode(compileInfo);
    tilingKey += dTypeTilingKeyMap[xDtype];
    tilingKey += quantModeTilingKeyMap[quantMode];
    return tilingKey;
}

// 计算每核处理的总行数和实际使用的核数
inline void CalTilingData(gert::TilingContext* context, SwiGluQuantCompileInfo& compileInfo,
                          SwiGluQuantTilingParam& tilingParam, SwiGluQuantTilingData& tilingData)
{
    uint32_t rowLen = tilingData.get_rowLen();
    tilingParam.coreNumUsed = std::max(std::min(compileInfo.totalCore, rowLen), ONE);
    tilingParam.headCoreNum = rowLen % tilingParam.coreNumUsed;
    tilingParam.rowLenPerHeadCore = (rowLen + tilingParam.coreNumUsed - 1) / tilingParam.coreNumUsed;
    tilingParam.rowLenPerTailCore = rowLen / tilingParam.coreNumUsed;
    CalculateMaxUbSizePerRow(context, compileInfo, tilingParam, tilingData);
}

inline void SetTilingData(SwiGluQuantCompileInfo& compileInfo, SwiGluQuantTilingParam& tilingParam,
                          SwiGluQuantTilingData& tilingData)
{
    tilingData.set_headCoreNum(tilingParam.headCoreNum);
    tilingData.set_basicRowLenHeadCore(tilingParam.optBaseRowLenHeadCore);
    tilingData.set_basicRowLenTailCore(tilingParam.optBaseRowLenTailCore);
    tilingData.set_basicColLen(tilingParam.optBaseColLen);
    tilingData.set_rowLenPerHeadCore(tilingParam.rowLenPerHeadCore);
    tilingData.set_rowLenPerTailCore(tilingParam.rowLenPerTailCore);
    tilingData.set_realCoreNum(tilingParam.coreNumUsed);
    tilingData.set_activateLeft(compileInfo.activateLeft);
    tilingData.set_groupListType(compileInfo.groupListType);
    tilingData.set_dstType(compileInfo.dstType);
    tilingData.set_hasGroup(compileInfo.hasGroup);
}
} // namespace optiling
#endif
