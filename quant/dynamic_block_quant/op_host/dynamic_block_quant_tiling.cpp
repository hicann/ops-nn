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
 * \file dynamic_block_quant_tiling.cpp
 * \brief
 */
#include "dynamic_block_quant_tiling.h"
#include "dynamic_block_quant_i8_tiling.h"

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"

using namespace ge;
using namespace AscendC;
using namespace optiling::dynamic_block_quant_i8;

namespace optiling {
constexpr int64_t INDEX_ATTR_MIN_SCALE = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 2;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE_ROW = 3;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE_COL = 4;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t BYTES_OF_FLOAT_TYPE = 4;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THOUSAND = 1000;
constexpr int64_t DIGIT_HUNDRED = 100;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t EXIST_NODE_NUM = 3;
constexpr int64_t AXIS_NUM_AFTER_MERGE = 3;
constexpr int64_t NEW_SHAPE_INDEX_TWO = 2;
constexpr int64_t WORKSPACE_SIZE = 32;
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<int64_t> DST_SUPPORT_DTYPE_SET = {34, 35, 36};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT};
constexpr int64_t DIM1_BLOCK_COUNT = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t TAIL_TILING_KEY_DIGIT = 4;
constexpr int64_t SINGLE_LOOP_MIN_COLS = 128;
constexpr int64_t BLOCK_SIZE_1 = 1;
constexpr int64_t BLOCK_SIZE_128 = 128;

inline static ge::graphStatus DynamicBlockQuantSetTilingData(
    gert::TilingContext* context, DynamicBlockQuantTilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

inline static void PrintTilingData(const gert::TilingContext* context, DynamicBlockQuantTilingData& tilingData)
{
    OP_LOGI(
        context,
        "tilingData is totalCoreNum:%ld, ubSize:%ld, vfLen:%ld, minScale:%f, \
roundMode:%ld, dstType:%ld, blockSizeRow:%ld, blockSizeCol:%ld, rowNum:%ld, colNum:%ld, rowBlockLoopNum:%ld, \
colBlockLoopNum:%ld, rowUbBlockLoopNum:%ld, colUbBlockLoopNum:%ld, rowUbFactor:%ld, colUbFactor:%ld, \
usedCoreNum:%ld, rowTileNum:%ld, colTileNum:%ld, normalCoreRowTileNum:%ld, \
normalCoreColTileNum:%ld, tailCoreRowTileNum:%ld, tailCoreColTileNum:%ld",
        tilingData.get_totalCoreNum(), tilingData.get_ubSize(), tilingData.get_vfLen(), tilingData.get_minScale(),
        tilingData.get_roundMode(), tilingData.get_dstType(), tilingData.get_blockSizeRow(),
        tilingData.get_blockSizeCol(), tilingData.get_rowNum(), tilingData.get_colNum(),
        tilingData.get_rowBlockLoopNum(), tilingData.get_colBlockLoopNum(), tilingData.get_rowUbBlockLoopNum(),
        tilingData.get_colUbBlockLoopNum(), tilingData.get_rowUbFactor(), tilingData.get_colUbFactor(),
        tilingData.get_usedCoreNum(), tilingData.get_normalCoreRowTileNum(), tilingData.get_rowTileNum(),
        tilingData.get_colTileNum(), tilingData.get_normalCoreColTileNum(), tilingData.get_tailCoreRowTileNum(),
        tilingData.get_tailCoreColTileNum());
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

static ge::graphStatus GetAttr(const gert::TilingContext* context, DynamicBlockQuantTilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* attrMinScale = attrs->GetAttrPointer<float>(INDEX_ATTR_MIN_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrMinScale);
    tilingParam.minScale = static_cast<float>(*attrMinScale);
    OP_LOGD(context, "The attr minScale is %f", tilingParam.minScale);
    OP_CHECK_IF((tilingParam.minScale < 0.0),
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

    OP_CHECK_IF((tilingParam.dstType != 34 && tilingParam.dstType != 35 && tilingParam.dstType != 36),
        OP_LOGE(context, "invalid dst_type: %ld. only support DT_HIFLOAT8, DT_FLOAT8_E4M3FN or DT_FLOAT8_E5M2",
            tilingParam.dstType),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingParam.dstType != static_cast<int64_t>(yDtype),
        OP_LOGE(context, "invalid attr dst_type is: %ld." "output y dtype is %s, correspond to %ld",
            tilingParam.dstType, Ops::Base::ToString(yDtype).c_str(), static_cast<int64_t>(yDtype)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((tilingParam.dstType == 34 && roundMode != RoundModeList::MODE_ROUND),
        OP_LOGE(context, "invalid round_mode: %s. dst_type DT_HIFLOAT8 supported round", attrRoundMode),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(((tilingParam.dstType == 35 || tilingParam.dstType == 36) && roundMode != RoundModeList::MODE_RINT),
        OP_LOGE(context, "invalid round_mode: %s. dst_type DT_FLOAT8_E4M3FN or DT_FLOAT8_E5M2 only supported rint",
            attrRoundMode),
        return ge::GRAPH_FAILED);

    auto* attrBlockSizeRow = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE_ROW);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSizeRow);
    tilingParam.blockSizeRow = static_cast<int64_t>(*attrBlockSizeRow);
    OP_CHECK_IF(tilingParam.blockSizeRow != BLOCK_SIZE_1 && tilingParam.blockSizeRow != BLOCK_SIZE_128,
        OP_LOGE(context, "The row_block_size is %ld but should be 1 or 128, please check.", tilingParam.blockSizeRow),
        return ge::GRAPH_FAILED);

    auto* attrBlockSizeCol = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE_COL);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBlockSizeCol);
    tilingParam.blockSizeCol = static_cast<int64_t>(*attrBlockSizeCol);
    OP_CHECK_IF(tilingParam.blockSizeCol != BLOCK_SIZE_128,
        OP_LOGE(context, "The col_block_size is %ld but should be 128, please check.", tilingParam.blockSizeCol),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(const gert::TilingContext* context)
{
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(context, "Input x dtype only support float16 and bfloat16 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        DST_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(
            context,
            "Output y dtype only support DT_HIFLOAT8/DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputScalePtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputScalePtr);
    auto scaleDtype = outputScalePtr->GetDataType();
    OP_CHECK_IF(
        OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(context, "Output scale dtype only support DT_FLOAT32 currently, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(const gert::TilingContext* context, const DynamicBlockQuantTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

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
        static_cast<int64_t>(xShape.GetDimNum()) != 2,
        OP_LOGE(context, "The shape of x dim should be 2, please check."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (static_cast<int64_t>(scaleShape.GetDim(0)) !=
         Ops::Base::CeilDiv(xShape.GetDim(0), tilingParam.blockSizeRow)) ||
            (static_cast<int64_t>(scaleShape.GetDim(1)) !=
             Ops::Base::CeilDiv(xShape.GetDim(1), tilingParam.blockSizeCol)),
        OP_LOGE(
            context,
            "The shape of output scale must be same with [ceil(x.rows/row_block_size), ceil(x.cols/col_block_size)], "
            "please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

inline static void CalcTilingKey(DataType inputType, DataType outputType, DynamicBlockQuantTilingParam& tilingParam)
{
    // 千分位表示 RoundMode
    int64_t thousandDigit = tilingParam.roundMode;
    // 百位数为1、2，分别表示输入类型是float16、bfloat16;
    int64_t hundredDigit = inputType == DT_FLOAT16 ? 1 : DIGIT_TWO;
    // 十位数为0、1、2、3，分别表示输出类型是float8_e5m2、float8_e4m3fn、hifloat8
    // 前面已做过Dtype校验
    int64_t tenDigit = 0;
    if (outputType == ge::DT_FLOAT8_E4M3FN) {
        tenDigit = 1;
    } else if (outputType == ge::DT_HIFLOAT8) {
        tenDigit = DIGIT_TWO;
    }
    int64_t digit = 0;
    if (tilingParam.blockSizeCol <= SINGLE_LOOP_MIN_COLS && tilingParam.blockSizeRow == 1) {
        digit = 1;
    }
    tilingParam.tilingKey =
        thousandDigit * DIGIT_THOUSAND + hundredDigit * DIGIT_HUNDRED + tenDigit * DIGIT_TEN + digit;
}

static void CalcAxisSize(DynamicBlockQuantTilingParam& tilingParam, const gert::Shape& xShape)
{
    tilingParam.rowNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        tilingParam.rowNum *= xShape.GetDim(i);
    }
    tilingParam.colNum = xShape.GetDim(xShape.GetDimNum() - 1);
    tilingParam.rowBlockLoopNum = Ops::Base::CeilDiv(tilingParam.rowNum, tilingParam.blockSizeRow);
    tilingParam.colBlockLoopNum = Ops::Base::CeilDiv(tilingParam.colNum, tilingParam.blockSizeCol);
}

static void SpliteUB(DynamicBlockQuantTilingParam& tilingParam, int64_t maxUbAvailable)
{
    tilingParam.colUbBlockLoopNum = maxUbAvailable < tilingParam.normalCoreColTileNum ?
                                        maxUbAvailable :
                                        maxUbAvailable / tilingParam.normalCoreColTileNum;
    maxUbAvailable = maxUbAvailable / tilingParam.colUbBlockLoopNum;
    tilingParam.rowUbBlockLoopNum =
        maxUbAvailable > tilingParam.normalCoreRowTileNum ? tilingParam.normalCoreRowTileNum : maxUbAvailable;
    tilingParam.rowUbFactor = tilingParam.rowUbBlockLoopNum * tilingParam.blockSizeRow;
    tilingParam.colUbFactor = tilingParam.colUbBlockLoopNum * tilingParam.blockSizeCol;
}

std::set<int64_t> FindUniqueCut(int64_t usedCoreNum)
{
    std::set<int64_t> result;
    int64_t upbound = std::ceil(std::sqrt(usedCoreNum) + 1);

    for (int64_t m = 1; m < upbound; m++) {
        int64_t y = usedCoreNum / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

static void AutoTiling(DynamicBlockQuantTilingParam& tilingParam)
{
    OP_LOGD("AutoTiling", "DynamicBlockQuant AutoTiling Enter.");

    // 计算可用核数
    tilingParam.usedCoreNum =
        std::min(tilingParam.totalCoreNum, tilingParam.rowBlockLoopNum * tilingParam.colBlockLoopNum);
    tilingParam.usedCoreNum = tilingParam.usedCoreNum == 0 ? 1 : tilingParam.usedCoreNum;

    // 查找切分的组合
    std::set<int64_t> cutSet = FindUniqueCut(tilingParam.usedCoreNum);

    std::vector<std::vector<int64_t>> allTiling;

    // 行方向切分，枚举 m 的取值
    for (int64_t m : cutSet) {
        if (m > tilingParam.rowBlockLoopNum) {
            continue;
        }

        int64_t n = tilingParam.usedCoreNum / m;
        n = n < 1 ? 1 : n;
        if (n > tilingParam.colBlockLoopNum) {
            continue;
        }

        int64_t rowNormalBlock = Ops::Base::CeilDiv(tilingParam.rowBlockLoopNum, m);
        int64_t colNormalBlock = Ops::Base::CeilDiv(tilingParam.colBlockLoopNum, n);

        int64_t delta = rowNormalBlock * colNormalBlock;
        if (m * n == static_cast<int64_t>(tilingParam.usedCoreNum)) {
            if (tilingParam.rowBlockLoopNum % m == 0 && tilingParam.colBlockLoopNum % n == 0) {
                tilingParam.rowTileNum = m;
                tilingParam.colTileNum = n;
                return;
            } else if (tilingParam.rowBlockLoopNum % m == 0) {
                delta = delta - rowNormalBlock * (tilingParam.colBlockLoopNum % colNormalBlock);
            } else if (tilingParam.colBlockLoopNum % n == 0) {
                delta = delta - (tilingParam.rowBlockLoopNum % rowNormalBlock) * n;
            } else {
                delta = delta -
                        (tilingParam.rowBlockLoopNum % rowNormalBlock) * (tilingParam.colBlockLoopNum % colNormalBlock);
            }
        }

        allTiling.push_back({m, n, m * n, delta});
    }

    // 排序以选择最合适的切分
    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int NIndex = 1;
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[DeltaIndex], a[NIndex]) < std::make_pair(b[DeltaIndex], b[NIndex]);
    });

    tilingParam.rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    tilingParam.colTileNum = static_cast<uint16_t>(allTiling[0][1]);
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, DynamicBlockQuantTilingParam& tilingParam)
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
    CalcTilingKey(inDtype, outDtype, tilingParam);

    // 先切多核
    AutoTiling(tilingParam);

    tilingParam.usedCoreNum = tilingParam.rowTileNum * tilingParam.colTileNum;

    tilingParam.normalCoreRowTileNum = Ops::Base::CeilDiv(tilingParam.rowBlockLoopNum, tilingParam.rowTileNum);
    tilingParam.normalCoreColTileNum = Ops::Base::CeilDiv(tilingParam.colBlockLoopNum, tilingParam.colTileNum);

    tilingParam.tailCoreRowTileNum = Ops::Base::FloorDiv(tilingParam.rowBlockLoopNum, tilingParam.rowTileNum);
    tilingParam.tailCoreColTileNum = Ops::Base::FloorDiv(tilingParam.colBlockLoopNum, tilingParam.colTileNum);

    tilingParam.rowNormalCoreNum =
        tilingParam.rowBlockLoopNum - tilingParam.rowTileNum * tilingParam.tailCoreRowTileNum;
    tilingParam.colNormalCoreNum =
        tilingParam.colBlockLoopNum - tilingParam.colTileNum * tilingParam.tailCoreColTileNum;

    tilingParam.rowNormalCoreNum =
        tilingParam.rowNormalCoreNum == 0 ? tilingParam.rowTileNum : tilingParam.rowNormalCoreNum;
    tilingParam.colNormalCoreNum =
        tilingParam.colNormalCoreNum == 0 ? tilingParam.colTileNum : tilingParam.colNormalCoreNum;

    tilingParam.rowTailCoreNum = tilingParam.rowTileNum - tilingParam.rowNormalCoreNum;
    tilingParam.colTailCoreNum = tilingParam.colTileNum - tilingParam.colNormalCoreNum;

    int64_t blockSizeColAlign = Ops::Base::CeilAlign(tilingParam.blockSizeCol, SINGLE_LOOP_MIN_COLS);

    // 每个block需要的临时ub大小
    int64_t perBlockTmpUbSize = tilingParam.blockSizeRow * blockSizeColAlign * BYTES_OF_FLOAT_TYPE;

    // 每个block需要的ub大小
    int64_t perBlockUbSize =
        tilingParam.blockSizeRow * blockSizeColAlign * BYTES_OF_INPUT_TYPE * EXIST_NODE_NUM + perBlockTmpUbSize;
    // 计算ub可以放下的block块数量
    int64_t maxUbAvailable = tilingParam.ubSize / perBlockUbSize;
    // 计算ubFactor
    SpliteUB(tilingParam, maxUbAvailable);

    return ge::GRAPH_SUCCESS;
}

inline static void SetTilingData(
    DynamicBlockQuantTilingData& tilingData, const DynamicBlockQuantTilingParam& tilingParam)
{
    tilingData.set_tilingKey(tilingParam.tilingKey);
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_ubSize(tilingParam.ubSize);
    tilingData.set_vfLen(tilingParam.vfLen);
    tilingData.set_minScale(tilingParam.minScale);
    tilingData.set_roundMode(tilingParam.roundMode);
    tilingData.set_dstType(tilingParam.dstType);
    tilingData.set_blockSizeRow(tilingParam.blockSizeRow);
    tilingData.set_blockSizeCol(tilingParam.blockSizeCol);
    tilingData.set_rowNum(tilingParam.rowNum);
    tilingData.set_colNum(tilingParam.colNum);
    tilingData.set_rowBlockLoopNum(tilingParam.rowBlockLoopNum);
    tilingData.set_colBlockLoopNum(tilingParam.colBlockLoopNum);
    tilingData.set_rowUbBlockLoopNum(tilingParam.rowUbBlockLoopNum);
    tilingData.set_colUbBlockLoopNum(tilingParam.colUbBlockLoopNum);
    tilingData.set_rowUbFactor(tilingParam.rowUbFactor);
    tilingData.set_colUbFactor(tilingParam.colUbFactor);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_rowTileNum(tilingParam.rowTileNum);
    tilingData.set_colTileNum(tilingParam.colTileNum);
    tilingData.set_normalCoreRowTileNum(tilingParam.normalCoreRowTileNum);
    tilingData.set_normalCoreColTileNum(tilingParam.normalCoreColTileNum);
    tilingData.set_tailCoreRowTileNum(tilingParam.tailCoreRowTileNum);
    tilingData.set_tailCoreColTileNum(tilingParam.tailCoreColTileNum);
    tilingData.set_rowNormalCoreNum(tilingParam.rowNormalCoreNum);
    tilingData.set_colNormalCoreNum(tilingParam.colNormalCoreNum);
    tilingData.set_rowTailCoreNum(tilingParam.rowTailCoreNum);
    tilingData.set_colTailCoreNum(tilingParam.colTailCoreNum);
}

ge::graphStatus Tiling4DynamicBlockQuant(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4DynamicBlockQuant running begin.");

    DynamicBlockQuantTilingParam tilingParam;
    DynamicBlockQuantI8 DynamicBlockQuantI8(context);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    bool isSoc910b = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910B ||
                     ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_93;
    if (isSoc910b) {
        OP_CHECK_IF(DynamicBlockQuantI8.CheckParams(tilingParam) != ge::GRAPH_SUCCESS,
            OP_LOGE(context, "The params check failed."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The dtype check failed."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(GetAttr(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The attr get failed."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(CheckShape(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "The shape check failed."),
            return ge::GRAPH_FAILED);
    }

    tilingParam.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParam.totalCoreNum <= 0), OP_LOGE(context, "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParam.ubSize = static_cast<int64_t>(ubSize);

    OP_CHECK_IF((tilingParam.ubSize <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);

    DynamicBlockQuantTilingData tilingData;

    if (isSoc910b) {
        OP_CHECK_IF(DynamicBlockQuantI8.DoTiling(tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Dotiling failed."),
            return ge::GRAPH_FAILED);

        DynamicBlockQuantI8.SetTilingData(tilingData, tilingParam);
    } else {
        tilingParam.vfLen = Ops::Base::GetVRegSize(context);

        OP_CHECK_IF(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Dotiling failed."),
            return ge::GRAPH_FAILED);
        SetTilingData(tilingData, tilingParam);
    }

    OP_CHECK_IF(DynamicBlockQuantSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "DynamicBlockQuantSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    PrintTilingData(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4DynamicBlockQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4DynamicBlockQuant entering.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the DynamicBlockQuant op.
IMPL_OP_OPTILING(DynamicBlockQuant)
    .Tiling(Tiling4DynamicBlockQuant)
    .TilingParse<DynamicBlockQuantCompileInfo>(TilingPrepare4DynamicBlockQuant);
} // namespace optiling