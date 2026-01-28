/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_dynamic_block_quant_tiling.cpp
 * \brief
 */
#include "grouped_dynamic_block_quant_tiling.h"

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr int64_t INDEX_ATTR_MIN_SCALE = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 2;
constexpr int64_t INDEX_ATTR_ROW_BLOCK_SIZE = 3;
constexpr int64_t INDEX_ATTR_COL_BLOCK_SIZE = 4;
constexpr int64_t INDEX_ATTR_GROUP_LIST_TYPE = 5;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t BYTES_OF_OUTPUT_TYPE = 1;
constexpr int64_t BYTES_OF_SCALE_TYPE = 4;
constexpr int64_t SCALE_ALIGN_NUM = 8;
constexpr int64_t RESERVED_SPACE_SCALE = 32;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t DIGIT_HUNDRED = 100;
constexpr int64_t DIGIT_THOUSAND = 1000;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t WORKSPACE_SIZE = 0; // 置0
const std::set<ge::DataType> INPUT_X_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> INPUT_GROUP_LIST_SUPPORT_DTYPE_SET = {ge::DT_INT32};
const std::set<ge::DataType> OUTPUT_Y_SUPPORT_DTYPE_SET = {ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> OUTPUT_SCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT};
constexpr int64_t GROUP_LIST_TYPE_ZERO = 0;
constexpr int64_t GROUP_LIST_TYPE_ONE = 1;
constexpr int64_t BLOCK_SIZE_1 = 1;
constexpr int64_t BLOCK_SIZE_64 = 64;
constexpr int64_t BLOCK_SIZE_128 = 128;
constexpr int64_t BLOCK_SIZE_192 = 192;
constexpr int64_t BLOCK_SIZE_256 = 256;
constexpr int64_t BLOCK_SIZE_512 = 512;
constexpr int64_t DT_HIFLOAT8_TYPE = 34;
constexpr int64_t DT_FLOAT8_E5M2_TYPE = 35;
constexpr int64_t DT_FLOAT8_E4M3FN_TYPE = 36;

inline static ge::graphStatus GroupedDynamicBlockQuantSetTilingData(
    gert::TilingContext* context, GroupedDynamicBlockQuantTilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext* context, GroupedDynamicBlockQuantTilingData& tilingData)
{
    OP_LOGI(
        context,
        "tilingData is tilingKey:%ld, totalCoreNum:%ld, ubSize:%ld, vfLen:%ld, usedCoreNum:%ld, headCoreNum:%ld, "
        "tailCoreNum:%ld, minScale:%f,"
        "roundMode:%ld, dstType:%ld, rowBlockSize:%ld, colBlockSize:%ld, batchNum:%ld, rowNum:%ld, colNum:%ld,"
        "scaleRowNum:%ld, scaleColNum:%ld, uo:%ld, groupNum:%ld, blockFactor:%ld, tailBlockFactor:%ld, "
        "groupBlockNumHeadCore:%ld,"
        "groupBlockNumTailCore:%ld, maxUbRow:%ld",
        tilingData.get_tilingKey(), tilingData.get_totalCoreNum(), tilingData.get_ubSize(), tilingData.get_vfLen(),
        tilingData.get_usedCoreNum(), tilingData.get_headCoreNum(), tilingData.get_tailCoreNum(),
        tilingData.get_minScale(), tilingData.get_roundMode(), tilingData.get_dstType(), tilingData.get_rowBlockSize(),
        tilingData.get_colBlockSize(), tilingData.get_batchNum(), tilingData.get_rowNum(), tilingData.get_colNum(),
        tilingData.get_scaleRowNum(), tilingData.get_scaleColNum(), tilingData.get_uo(), tilingData.get_groupNum(),
        tilingData.get_blockFactor(), tilingData.get_tailBlockFactor(), tilingData.get_groupBlockNumHeadCore(),
        tilingData.get_groupBlockNumTailCore(), tilingData.get_maxUbRow());
}

static RoundModeList GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    }
    if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    }
    if (roundMode == "hybrid") {
        return RoundModeList::MODE_HYBRID;
    }
    return RoundModeList::MODE_UNDEFINED;
}

static ge::graphStatus GetAttr(const gert::TilingContext* context, GroupedDynamicBlockQuantTilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto* attrMinScale = attrs->GetAttrPointer<float>(INDEX_ATTR_MIN_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrMinScale);
    tilingParam.minScale = static_cast<float>(*attrMinScale);
    OP_LOGD(context, "The attr minScale is %f", tilingParam.minScale);
    OP_CHECK_IF(
        (tilingParam.minScale < 0.0),
        OP_LOGE(context, "invalid min_scale:%f. min_scale should be greater or equal than 0", tilingParam.minScale),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    tilingParam.dstType = static_cast<int64_t>(*attrDstType);

    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    tilingParam.roundMode = static_cast<int64_t>(roundMode);

    OP_CHECK_IF(
        (tilingParam.dstType != DT_HIFLOAT8_TYPE && tilingParam.dstType != DT_FLOAT8_E5M2_TYPE &&
         tilingParam.dstType != DT_FLOAT8_E4M3FN_TYPE),
        OP_LOGE(
            context, "invalid dst_type: %ld. only support DT_HIFLOAT8, DT_FLOAT8_E4M3FN or DT_FLOAT8_E5M2",
            tilingParam.dstType),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tilingParam.dstType != static_cast<int64_t>(yDtype),
        OP_LOGE(
            context,
            "invalid attr dst_type is: %ld."
            "output y dtype is %s, correspond to %ld",
            tilingParam.dstType, Ops::Base::ToString(yDtype).c_str(), static_cast<int64_t>(yDtype)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (tilingParam.dstType == DT_HIFLOAT8_TYPE && roundMode != RoundModeList::MODE_ROUND &&
         roundMode != RoundModeList::MODE_HYBRID),
        OP_LOGE(context, "invalid round_mode: %s. dst_type DT_HIFLOAT8 only supported round and hybrid", attrRoundMode),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((tilingParam.dstType == DT_FLOAT8_E5M2_TYPE || tilingParam.dstType == DT_FLOAT8_E4M3FN_TYPE) &&
         roundMode != RoundModeList::MODE_RINT),
        OP_LOGE(
            context,
            "invalid round_mode: %s. dst_type DT_FLOAT8_E4M3FN and DT_FLOAT8_E5M2 only supported rint currently",
            attrRoundMode),
        return ge::GRAPH_FAILED);

    auto* attrRowBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_ROW_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrRowBlockSize);
    tilingParam.rowBlockSize = static_cast<int64_t>(*attrRowBlockSize);
    OP_CHECK_IF(
        tilingParam.rowBlockSize != BLOCK_SIZE_1 && tilingParam.rowBlockSize != BLOCK_SIZE_128 &&
            tilingParam.rowBlockSize != BLOCK_SIZE_256 && tilingParam.rowBlockSize != BLOCK_SIZE_512,
        OP_LOGE(
            context, "The row_block_size is %ld but should be 1/128/256/512, please check.",
            tilingParam.rowBlockSize),
        return ge::GRAPH_FAILED);

    auto* attrColBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_COL_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrColBlockSize);
    tilingParam.colBlockSize = static_cast<int64_t>(*attrColBlockSize);
    OP_CHECK_IF(
        tilingParam.colBlockSize != BLOCK_SIZE_64 && tilingParam.colBlockSize != BLOCK_SIZE_128 &&
            tilingParam.colBlockSize != BLOCK_SIZE_192 && tilingParam.colBlockSize != BLOCK_SIZE_256,
        OP_LOGE(
            context, "The col_block_size is %ld but should be 64/128/192/256, please check.",
            tilingParam.colBlockSize),
        return ge::GRAPH_FAILED);

    auto* attrGroupListType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrGroupListType);
    int64_t groupListType = static_cast<int64_t>(*attrGroupListType);
    OP_CHECK_IF(
        groupListType != GROUP_LIST_TYPE_ZERO && groupListType != GROUP_LIST_TYPE_ONE,
        OP_LOGE(context, "The group_list_type is %ld but should be 0(cumsum) currently, please check.", groupListType),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext* context)
{
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_X_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(context, "Input x dtype only support float16 and bfloat16 currently, please check."),
        return ge::GRAPH_FAILED);

    auto inputGroupedIndexPtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGroupedIndexPtr);
    auto groupedIndexDtype = inputGroupedIndexPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_GROUP_LIST_SUPPORT_DTYPE_SET.count(groupedIndexDtype) == 0,
        OP_LOGE(context, "Input grouped index dtype only support int32 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        OUTPUT_Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(
            context,
            "Output y dtype only support DT_HIFLOAT8/DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputScalePtr);
    auto scaleDtype = outputScalePtr->GetDataType();
    OP_CHECK_IF(
        OUTPUT_SCALE_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(context, "Output scale dtype only support DT_FLOAT32 currently, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext* context, GroupedDynamicBlockQuantTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    auto groupListShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListShapePtr);
    auto groupListShape = groupListShapePtr->GetStorageShape();
    tilingParam.groupNum = static_cast<int64_t>(groupListShape.GetDim(0));

    auto outputYPtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yShape = outputYPtr->GetStorageShape();

    auto scaleShapePtr = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShapePtr);
    auto scaleShape = scaleShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape, OP_LOGE(context, "The shape of output y must be same with shape of input x, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        static_cast<int64_t>(groupListShape.GetDimNum()) != 1,
        OP_LOGE(context, "The shape of group_list dim must be 1, please check."), return ge::GRAPH_FAILED);

    if (static_cast<int64_t>(xShape.GetDimNum()) == 2) {
        OP_CHECK_IF(
            ((static_cast<int64_t>(scaleShape.GetDim(0)) !=
              static_cast<int64_t>(xShape.GetDim(0)) / tilingParam.rowBlockSize + tilingParam.groupNum) ||
             (static_cast<int64_t>(scaleShape.GetDim(1)) !=
              Ops::Base::CeilDiv(xShape.GetDim(1), tilingParam.colBlockSize))),
            OP_LOGE(
                context,
                "When the shape dim of x is 2, the shape of output scale must be same with [x.rows / row_block_size + "
                "groupListSize, ceil(x.cols / col_block_size)], "
                "please check."),
            return ge::GRAPH_FAILED);
    } else if (static_cast<int64_t>(xShape.GetDimNum()) == 3) {
        OP_CHECK_IF(
            ((static_cast<int64_t>(scaleShape.GetDim(0)) != static_cast<int64_t>(xShape.GetDim(0))) ||
             (static_cast<int64_t>(scaleShape.GetDim(1)) !=
              static_cast<int64_t>(xShape.GetDim(1)) / tilingParam.rowBlockSize + tilingParam.groupNum) ||
             (static_cast<int64_t>(scaleShape.GetDim(2)) !=
              Ops::Base::CeilDiv(xShape.GetDim(2), tilingParam.colBlockSize))),
            OP_LOGE(
                context,
                "When the shape dim of x is 3, the shape of output scale must be same with [x.batch, x.rows / "
                "row_block_size + groupListSize, ceil(x.cols / col_block_size)], "
                "please check."),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            true, OP_LOGE(context, "The shape of x dim should be 2 or 3, please check."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

inline static void CalcTilingKey(
    DataType inputType, DataType outputType, bool blockIsSmallThanUB, GroupedDynamicBlockQuantTilingParam& tilingParam)
{
    // 千位数为1、2，分别表示Block放得下UB和Block放不下UB得情况;
    int64_t thousandDigit = blockIsSmallThanUB ? DIGIT_ONE : DIGIT_TWO;
    // 百位数为1、2，分别表示输入类型是float16、bfloat16;
    int64_t hundredDigit = inputType == DT_FLOAT16 ? DIGIT_ONE : DIGIT_TWO;
    // 十位数为1、2、3，分别表示输出类型是float8_e5m2、float8_e4m3fn、hifloat8
    // 前面已做过Dtype校验
    int64_t tenDigit = 0;
    if (outputType == ge::DT_FLOAT8_E4M3FN) {
        tenDigit = DIGIT_ONE;
    } else if (outputType == ge::DT_FLOAT8_E5M2) {
        tenDigit = DIGIT_TWO;
    } else if (outputType == ge::DT_HIFLOAT8) {
        tenDigit = DIGIT_THREE;
    }
    // 个位表示 RoundMode
    int64_t digit = tilingParam.roundMode;
    tilingParam.tilingKey =
        thousandDigit * DIGIT_THOUSAND + hundredDigit * DIGIT_HUNDRED + tenDigit * DIGIT_TEN + digit * DIGIT_ONE;
}

static void CalcAxisSize(GroupedDynamicBlockQuantTilingParam& tilingParam, const gert::Shape& xShape)
{
    if (xShape.GetDimNum() == 2) {
        tilingParam.batchNum = 1;
        tilingParam.rowNum = xShape.GetDim(0);
        tilingParam.colNum = xShape.GetDim(1);
    } else {
        tilingParam.batchNum = xShape.GetDim(0);
        tilingParam.rowNum = xShape.GetDim(1);
        tilingParam.colNum = xShape.GetDim(2);
    }
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, GroupedDynamicBlockQuantTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    CalcAxisSize(tilingParam, xShape);

    // 获取输入/输出数据类型
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto inDtype = inputXPtr->GetDataType();
    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto outDtype = outputYPtr->GetDataType();

    tilingParam.scaleRowNum = tilingParam.rowNum / tilingParam.rowBlockSize + tilingParam.groupNum;
    tilingParam.scaleColNum = Ops::Base::CeilDiv(tilingParam.colNum, tilingParam.colBlockSize);

    tilingParam.blockFactor = tilingParam.colNum <= tilingParam.colBlockSize ? 0 : tilingParam.colBlockSize;
    tilingParam.tailBlockFactor = tilingParam.colNum % tilingParam.colBlockSize == 0 ?
                                      tilingParam.colBlockSize :
                                      tilingParam.colNum % tilingParam.colBlockSize;

    tilingParam.uo = Ops::Base::CeilDiv(tilingParam.colNum, tilingParam.colBlockSize);
    int64_t splitCoreData = tilingParam.uo * tilingParam.batchNum;

    bool isNotNeedCutGroup = splitCoreData >= tilingParam.totalCoreNum;
    if (isNotNeedCutGroup) {
        tilingParam.usedCoreNum = tilingParam.totalCoreNum;
        tilingParam.groupBlockNumHeadCore = Ops::Base::CeilDiv(splitCoreData, tilingParam.usedCoreNum);
        tilingParam.groupBlockNumTailCore = tilingParam.groupBlockNumHeadCore - 1;
        tilingParam.tailCoreNum = tilingParam.groupBlockNumHeadCore * tilingParam.usedCoreNum - splitCoreData;
        tilingParam.headCoreNum = tilingParam.usedCoreNum - tilingParam.tailCoreNum;
        tilingParam.groupBlockNumHeadCore *= tilingParam.groupNum;
        tilingParam.groupBlockNumTailCore *= tilingParam.groupNum;
    } else {
        splitCoreData *= tilingParam.groupNum;
        tilingParam.usedCoreNum = std::min(splitCoreData, tilingParam.totalCoreNum);
        tilingParam.groupBlockNumHeadCore = Ops::Base::CeilDiv(splitCoreData, tilingParam.usedCoreNum);
        tilingParam.groupBlockNumTailCore = tilingParam.groupBlockNumHeadCore - 1;
        tilingParam.tailCoreNum = tilingParam.groupBlockNumHeadCore * tilingParam.usedCoreNum - splitCoreData;
        tilingParam.headCoreNum = tilingParam.usedCoreNum - tilingParam.tailCoreNum;
    }

    // 推导公式
    // (BYTES_OF_INPUT_TYPE+BYTES_OF_OUTPUT_TYPE)*colBlockSize*maxUbAvailableRows +
    // BYTES_OF_SCALE_TYPE*SCALE_ALIGN_NUM*Ceil(maxUbAvailableRows,rowBlockSize) = ubSize/DB_BUFFER
    int64_t totalElementSize = BYTES_OF_SCALE_TYPE * SCALE_ALIGN_NUM + tilingParam.rowBlockSize *
                                                                           tilingParam.colBlockSize *
                                                                           (BYTES_OF_INPUT_TYPE + BYTES_OF_OUTPUT_TYPE);
    int64_t maxUbAvailableRows = tilingParam.rowBlockSize * (tilingParam.ubSize / DB_BUFFER) / totalElementSize;

    bool blockIsSmallThanUB = tilingParam.rowBlockSize <= maxUbAvailableRows;
    tilingParam.maxUbRow =
        blockIsSmallThanUB ?
            maxUbAvailableRows / tilingParam.rowBlockSize * tilingParam.rowBlockSize :
            tilingParam.rowBlockSize * ((tilingParam.ubSize - tilingParam.vfLen) / DB_BUFFER) / totalElementSize;

    CalcTilingKey(inDtype, outDtype, blockIsSmallThanUB, tilingParam);

    return ge::GRAPH_SUCCESS;
}

inline static void SetTilingData(
    GroupedDynamicBlockQuantTilingData& tilingData, const GroupedDynamicBlockQuantTilingParam& tilingParam)
{
    tilingData.set_tilingKey(tilingParam.tilingKey);
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_ubSize(tilingParam.ubSize);
    tilingData.set_vfLen(tilingParam.vfLen);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_headCoreNum(tilingParam.headCoreNum);
    tilingData.set_tailCoreNum(tilingParam.tailCoreNum);
    tilingData.set_minScale(tilingParam.minScale);
    tilingData.set_roundMode(tilingParam.roundMode);
    tilingData.set_dstType(tilingParam.dstType);
    tilingData.set_rowBlockSize(tilingParam.rowBlockSize);
    tilingData.set_colBlockSize(tilingParam.colBlockSize);
    tilingData.set_batchNum(tilingParam.batchNum);
    tilingData.set_rowNum(tilingParam.rowNum);
    tilingData.set_colNum(tilingParam.colNum);
    tilingData.set_scaleRowNum(tilingParam.scaleRowNum);
    tilingData.set_scaleColNum(tilingParam.scaleColNum);
    tilingData.set_uo(tilingParam.uo);
    tilingData.set_groupNum(tilingParam.groupNum);
    tilingData.set_blockFactor(tilingParam.blockFactor);
    tilingData.set_tailBlockFactor(tilingParam.tailBlockFactor);
    tilingData.set_groupBlockNumHeadCore(tilingParam.groupBlockNumHeadCore);
    tilingData.set_groupBlockNumTailCore(tilingParam.groupBlockNumTailCore);
    tilingData.set_maxUbRow(tilingParam.maxUbRow);
}

ge::graphStatus Tiling4GroupedDynamicBlockQuant(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4GroupedDynamicBlockQuant running begin.");
    GroupedDynamicBlockQuantTilingParam tilingParam;

    OP_CHECK_IF(
        CheckDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The dtype check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The attr get failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The shape check failed."),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParam.totalCoreNum <= 0), OP_LOGE(context, "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);
    tilingParam.vfLen = Ops::Base::GetVRegSize(context);

    OP_CHECK_IF((tilingParam.ubSize <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);

    GroupedDynamicBlockQuantTilingData tilingData;
    OP_CHECK_IF(
        DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Dotiling failed."),
        return ge::GRAPH_FAILED);
    SetTilingData(tilingData, tilingParam);

    OP_CHECK_IF(
        GroupedDynamicBlockQuantSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GroupedDynamicBlockQuantSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    PrintTilingData(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GroupedDynamicBlockQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4GroupedDynamicBlockQuant entering.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GroupedDynamicBlockQuant op.
IMPL_OP_OPTILING(GroupedDynamicBlockQuant)
    .Tiling(Tiling4GroupedDynamicBlockQuant)
    .TilingParse<GroupedDynamicBlockQuantCompileInfo>(TilingPrepare4GroupedDynamicBlockQuant);
} // namespace optiling