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
 * \file dynamic_block_quant_i8_tiling.cpp
 * \brief
 */
#include "dynamic_block_quant_tiling.h"
#include "dynamic_block_quant_i8_tiling.h"

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
namespace dynamic_block_quant_i8 {
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_I8_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_I8_SET = {ge::DT_INT8};
const std::set<ge::DataType> SCALE_SUPPORT_I8_DTYPE_SET = {ge::DT_FLOAT};
constexpr int64_t INDEX_ATTR_MIN_SCALE = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 2;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE_ROW = 3;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE_COL = 4;
constexpr int64_t BLOCK_SIZE_1 = 1;
constexpr int64_t BLOCK_SIZE_128 = 128;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t NUM_TEN = 10;

class RowTilingData {
public:
    int64_t coreNum = 0;
    int64_t perCoreBlock = 0;
    int64_t tailBlock = 0;
    int64_t extraCoreBlock = 0;
    RowTilingData(int64_t curCoreNum, int64_t totalBlock)
    {
        this->coreNum = curCoreNum;
        this->perCoreBlock = curCoreNum != 0 ? (totalBlock / curCoreNum) : totalBlock;
        this->tailBlock = totalBlock - curCoreNum * this->perCoreBlock;
        this->extraCoreBlock = this->perCoreBlock + 1;
    }
    int64_t GetCurCoreBlockNum(int64_t coreIdx)
    {
        return coreIdx >= this->tailBlock ? this->perCoreBlock : this->extraCoreBlock;
    }
};

static RoundModeList GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    }
    return RoundModeList::MODE_UNDEFINED;
}

ge::graphStatus DynamicBlockQuantI8::CheckParams(DynamicBlockQuantTilingParam& tilingParam)
{
    OP_CHECK_IF(
        CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context, "The dtype check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttr(tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The attr get failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckShape(tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The shape check failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockQuantI8::CheckDtype()
{
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_I8_SET.count(xDtype) == 0,
        OP_LOGE(context, "Input x dtype only support float16 and bfloat16 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_I8_SET.count(yDtype) == 0,
        OP_LOGE(context, "Output y dtype only support DT_INT8 currently, please check."), return ge::GRAPH_FAILED);

    auto outputScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputScalePtr);
    auto scaleDtype = outputScalePtr->GetDataType();
    OP_CHECK_IF(
        SCALE_SUPPORT_I8_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(context, "Output scale dtype only support DT_FLOAT32 currently, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockQuantI8::GetAttr(DynamicBlockQuantTilingParam& tilingParam)
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

    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    tilingParam.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDstType);
    tilingParam.dstType = static_cast<int64_t>(*attrDstType);

    OP_CHECK_IF(
        tilingParam.dstType != static_cast<int64_t>(yDtype),
        OP_LOGE(
            context,
            "invalid attr dst_type is: %ld."
            "output y dtype is %s, correspond to %ld",
            tilingParam.dstType, Ops::Base::ToString(yDtype).c_str(), static_cast<int64_t>(yDtype)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (tilingParam.dstType != 2),
        OP_LOGE(context, "invalid dst_type: %ld. only support DT_INT8", tilingParam.dstType), return ge::GRAPH_FAILED);

    return GetAttrBlock(tilingParam);
}

ge::graphStatus DynamicBlockQuantI8::GetAttrBlock(DynamicBlockQuantTilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    auto* attrBlockSizeRow = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE_ROW);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSizeRow);
    tilingParam.blockSizeRow = static_cast<int64_t>(*attrBlockSizeRow);
    OP_CHECK_IF(
        tilingParam.blockSizeRow != BLOCK_SIZE_1,
        OP_LOGE(context, "The row_block_size is %ld but should be 1, please check.", tilingParam.blockSizeRow),
        return ge::GRAPH_FAILED);

    auto* attrBlockSizeCol = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE_COL);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSizeCol);
    tilingParam.blockSizeCol = static_cast<int64_t>(*attrBlockSizeCol);
    OP_CHECK_IF(
        tilingParam.blockSizeCol != BLOCK_SIZE_128,
        OP_LOGE(context, "The col_block_size is %ld but should be 128, please check.", tilingParam.blockSizeCol),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockQuantI8::CheckShape(DynamicBlockQuantTilingParam& tilingParam)
{
    auto outputYPtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yShape = outputYPtr->GetStorageShape();

    auto scaleShapePtr = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShapePtr);
    auto scaleShape = scaleShapePtr->GetStorageShape();

    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape, OP_LOGE(context, "The shape of output y must be same with shape of input x, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        static_cast<int64_t>(xShape.GetDimNum() != DIM_TWO && xShape.GetDimNum() != DIM_THREE),
        OP_LOGE(context, "The shape of x dim should be 2 or 3, please check."), return ge::GRAPH_FAILED);

    if (xShape.GetDimNum() == DIM_TWO) {
        OP_CHECK_IF(
            (static_cast<int64_t>(scaleShape.GetDim(0)) !=
             Ops::Base::CeilDiv(xShape.GetDim(0), tilingParam.blockSizeRow)) ||
                (static_cast<int64_t>(scaleShape.GetDim(1)) !=
                 Ops::Base::CeilDiv(xShape.GetDim(1), tilingParam.blockSizeCol)),
            OP_LOGE(
                context,
                "The shape of output scale must be same with [ceil(x_shape[0]/row_block_size), "
                "ceil(x_shape[1]/col_block_size)], please check."),
            return ge::GRAPH_FAILED);
    } else if (xShape.GetDimNum() == DIM_THREE) {
        OP_CHECK_IF(
            (static_cast<int64_t>(scaleShape.GetDim(0)) != xShape.GetDim(0)) ||
                (static_cast<int64_t>(scaleShape.GetDim(1)) !=
                 Ops::Base::CeilDiv(xShape.GetDim(1), tilingParam.blockSizeRow)) ||
                (static_cast<int64_t>(scaleShape.GetDim(DIM_TWO)) !=
                 Ops::Base::CeilDiv(xShape.GetDim(DIM_TWO), tilingParam.blockSizeCol)),
            OP_LOGE(
                context,
                "The shape of output scale must be same with [x_shape[0], ceil(x_shape[1]/row_block_size), "
                "ceil(x_shape[2]/col_block_size)], please check."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockQuantI8::DoTiling(DynamicBlockQuantTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    CalcTilingData(tilingParam, xShape);

    // 获取输入/输出数据类型
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto inDtype = inputXPtr->GetDataType();
    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto outDtype = outputYPtr->GetDataType();
    CalcTilingKey(inDtype, tilingParam);
    return ge::GRAPH_SUCCESS;
}

void DynamicBlockQuantI8::CalcTilingData(DynamicBlockQuantTilingParam& tilingParam, const gert::Shape& xShape)
{
    if (xShape.GetDimNum() == DIM_TWO) {
        tilingParam.rowNum = xShape.GetDim(0);
        tilingParam.colNum = xShape.GetDim(1);
    } else {
        tilingParam.rowNum = xShape.GetDim(0) * xShape.GetDim(1);
        tilingParam.colNum = xShape.GetDim(DIM_TWO);
    }

    // 每一列的block数量
    int64_t singleRowBlockNum = Ops::Base::CeilDiv(tilingParam.colNum, tilingParam.blockSizeCol);
    // 先按行切分
    perCoreRowNum = tilingParam.rowNum / tilingParam.totalCoreNum;
    int64_t tailStartRowIdx = perCoreRowNum * tilingParam.totalCoreNum;

    int64_t tailRowNum = tilingParam.rowNum - tailStartRowIdx;

    // 计算剩余的block数量
    int64_t tailBlockNum = singleRowBlockNum * tailRowNum;

    // 尾块开始的行idx
    int64_t curRowIdx = tailStartRowIdx;
    int64_t curColBlockIdx = 0;
    // 若剩余block数量小于总核数
    if (tailBlockNum <= tilingParam.totalCoreNum) {
        for (int64_t i = 0; i < tailBlockNum; i++) {
            if (curColBlockIdx >= singleRowBlockNum) {
                curRowIdx++;
                curColBlockIdx = 0;
            }
            tailRowList[i] = static_cast<uint32_t>(curRowIdx);
            tailColBlockStartList[i] = static_cast<uint32_t>(curColBlockIdx);
            tailColBlockEndList[i] = static_cast<uint32_t>(++curColBlockIdx);
        }
        tilingParam.usedCoreNum = perCoreRowNum > 0 ? tilingParam.totalCoreNum : tailBlockNum;

        return;
    }

    // 尾块中每一行至少使用的核数
    int64_t tailRowCoreNum = tilingParam.totalCoreNum / tailRowNum;
    // 须额外多使用一个核的行数
    int64_t tailExtraCoreRowNum = tilingParam.totalCoreNum % tailRowNum;

    // 当前行开始位置使用的核总idx
    int64_t curRowCoreBaseIdx = 0;

    RowTilingData extraRowTilingData(tailRowCoreNum + 1, singleRowBlockNum);
    RowTilingData rowTilingData(tailRowCoreNum, singleRowBlockNum);
    RowTilingData* curRowTilingData = &extraRowTilingData;

    int64_t curCoreBlockNum = 0;
    for (int64_t i = 0; i < tailRowNum; i++) {
        curColBlockIdx = 0;
        curRowTilingData = i >= tailExtraCoreRowNum ? &rowTilingData : &extraRowTilingData;
        for (int64_t j = 0; j < curRowTilingData->coreNum; j++) {
            tailRowList[curRowCoreBaseIdx + j] = static_cast<uint32_t>(tailStartRowIdx + i);
            curCoreBlockNum = curRowTilingData->GetCurCoreBlockNum(j);
            tailColBlockStartList[curRowCoreBaseIdx + j] = static_cast<uint32_t>(curColBlockIdx);
            curColBlockIdx += curCoreBlockNum;
            tailColBlockEndList[curRowCoreBaseIdx + j] = static_cast<uint32_t>(curColBlockIdx);
        }
        curRowCoreBaseIdx += curRowTilingData->coreNum;
    }

    tilingParam.usedCoreNum = tilingParam.totalCoreNum;
}

void DynamicBlockQuantI8::CalcTilingKey(DataType inputType, DynamicBlockQuantTilingParam& tilingParam)
{
    // 十分位为0、1，分别表示输入为float16，bfloat16
    int64_t tenDigit = inputType == DT_FLOAT16 ? 0 : 1;
    int64_t digit = 0;
    tilingParam.tilingKey = tenDigit * NUM_TEN + digit;
}

void DynamicBlockQuantI8::SetTilingData(
    DynamicBlockQuantTilingData& tilingData, const DynamicBlockQuantTilingParam& tilingParam)
{
    tilingData.set_tilingKey(tilingParam.tilingKey);
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_ubSize(tilingParam.ubSize);
    tilingData.set_minScale(tilingParam.minScale);
    tilingData.set_roundMode(tilingParam.roundMode);
    tilingData.set_dstType(tilingParam.dstType);
    tilingData.set_blockSizeRow(tilingParam.blockSizeRow);
    tilingData.set_blockSizeCol(tilingParam.blockSizeCol);
    tilingData.set_rowNum(tilingParam.rowNum);
    tilingData.set_colNum(tilingParam.colNum);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_tailRowList(tailRowList);
    tilingData.set_tailColBlockStartList(tailColBlockStartList);
    tilingData.set_tailColBlockEndList(tailColBlockEndList);
    tilingData.set_perCoreRowNum(perCoreRowNum);
}
} // namespace dynamic_block_quant_i8
} // namespace optiling