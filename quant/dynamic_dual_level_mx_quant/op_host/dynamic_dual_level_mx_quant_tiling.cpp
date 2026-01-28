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
 * \file dynamic_dual_level_mx_quant_tiling.cpp
 * \brief
 */

#include "dynamic_dual_level_mx_quant_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "util/math_util.h"
#include "quant/dynamic_dual_level_mx_quant/op_kernel/arch35/dynamic_dual_level_mx_quant_struct.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace DynamicDualLevelMxQuantOp;

namespace optiling {
constexpr int64_t INDEX_ATTR_ROUND_MODE = 0;
constexpr int64_t INDEX_ATTR_LEVEL0_BLOCK_SIZE = 1;
constexpr int64_t INDEX_ATTR_LEVEL1_BLOCK_SIZE = 2;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_32 = 32;
constexpr int64_t DIGIT_64 = 64;
constexpr int64_t DIGIT_128 = 128;
constexpr int64_t BLOCK_SIZE = 512;
constexpr int64_t DEFAULT_LEVEL0_BLOCK_SIZE = 512;
constexpr int64_t DEFAULT_LEVEL1_BLOCK_SIZE = 32;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t RESERVED_UB_SIZE = 1024;
constexpr int64_t COPY_SINGLE_ROW = 1;
constexpr int64_t COPY_MULTIPLE_ROW = 0;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1};
const std::set<ge::DataType> LEVEL0_SCALE_OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT};
const std::set<ge::DataType> LEVEL1_SCALE_OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};

RoundModeList DynamicDualLevelMxQuantTiling::GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    } else if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    } else if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}

ge::graphStatus DynamicDualLevelMxQuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode == RoundModeList::MODE_UNDEFINED),
        OP_LOGE(
            context_->GetNodeName(), "invalid round_mode:%s; round_mode should be one of {rint, floor, round}",
            attrRoundMode),
        return ge::GRAPH_FAILED);
    tilingParams.roundMode = static_cast<int64_t>(roundMode);

    auto* attrLevel0BlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_LEVEL0_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrLevel0BlockSize);
    tilingParams.level0BlockSize = static_cast<int64_t>(*attrLevel0BlockSize);
    OP_CHECK_IF(
        tilingParams.level0BlockSize != DEFAULT_LEVEL0_BLOCK_SIZE,
        OP_LOGE(
            context_->GetNodeName(), "The level0BlockSize[%ld] should be 512, please check.",
            tilingParams.level0BlockSize),
        return ge::GRAPH_FAILED);

    auto* attrLevel1BlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_LEVEL1_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrLevel1BlockSize);
    tilingParams.level1BlockSize = static_cast<int64_t>(*attrLevel1BlockSize);
    OP_CHECK_IF(
        tilingParams.level1BlockSize != DEFAULT_LEVEL1_BLOCK_SIZE,
        OP_LOGE(
            context_->GetNodeName(), "The level1BlockSize[%ld] should be 32, please check.",
            tilingParams.level1BlockSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicDualLevelMxQuantTiling::CheckDtype() const
{
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Input x's data type only support FLOAT16 and BFLOAT16 currently, but x is [%s], please check.",
            Ops::Base::ToString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output y's data type only support FLOAT4_E2M1 currently, but y is [%s], please check.",
            Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputLevel0ScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputLevel0ScalePtr);
    auto Level0ScaleDtype = outputLevel0ScalePtr->GetDataType();
    OP_CHECK_IF(
        LEVEL0_SCALE_OUTPUT_SUPPORT_DTYPE_SET.count(Level0ScaleDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output level0ScaleDtype's data type only support DT_FLOAT currently, but scale is [%s], please check.",
            Ops::Base::ToString(Level0ScaleDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputLevel1ScalePtr = context_->GetOutputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputLevel1ScalePtr);
    auto Level1ScaleDtype = outputLevel1ScalePtr->GetDataType();
    OP_CHECK_IF(
        LEVEL1_SCALE_OUTPUT_SUPPORT_DTYPE_SET.count(Level1ScaleDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output Level1ScaleDtype's data type only support FLOAT8_E8M0 currently, but scale is [%s], please check.",
            Ops::Base::ToString(Level1ScaleDtype).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicDualLevelMxQuantTiling::CheckShape() const
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape.GetDim(xShape.GetDimNum() - 1) % DIGIT_TWO != 0,
        OP_LOGE(
            context_->GetNodeName(),
            "When output y's data type is FLOAT4_E2M1, the x last axis should be even, please check."),
        return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE(
            context_->GetNodeName(), 
            "The shape of output y:%s must be same with shape of input x:%s, please check.",
            Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
        return ge::GRAPH_FAILED);

    auto level0ScaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, level0ScaleShapePtr);
    auto level0ScaleShape = level0ScaleShapePtr->GetStorageShape();

    auto level1ScaleShapePtr = context_->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, level1ScaleShapePtr);
    auto level1ScaleShape = level1ScaleShapePtr->GetStorageShape();

    auto newScale0Shape = xShape;
    newScale0Shape.SetDim(
        xShape.GetDimNum() - 1, Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - 1), BLOCK_SIZE));
    auto newScale1Shape = xShape;
    newScale1Shape.SetDim(xShape.GetDimNum() - 1, Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_64));
    newScale1Shape.AppendDim(DIGIT_TWO);

    OP_CHECK_IF(
        newScale0Shape != level0ScaleShape,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of output level0_scale %s is incorrect, correct shape is %s, please check.",
            Shape2String(level0ScaleShape).c_str(), Shape2String(newScale0Shape).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        newScale1Shape != level1ScaleShape,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of output level1_scale %s is incorrect, correct shape is %s, please check.",
            Shape2String(level1ScaleShape).c_str(), Shape2String(newScale1Shape).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// totalCoreNum，ubSize，workspaceSize
ge::graphStatus DynamicDualLevelMxQuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicDualLevelMxQuantTiling GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (tilingParams.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (tilingParams.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    tilingParams.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

// 合轴
ge::graphStatus DynamicDualLevelMxQuantTiling::MergeAxis()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        tilingParams.colSize *= xShape.GetDim(i);
    }
    tilingParams.rowSize = xShape.GetDim(xShape.GetDimNum() - 1);
    tilingParams.rowBlockNum = Ops::Base::CeilDiv(tilingParams.rowSize, tilingParams.blockSizeRow);
    tilingParams.colBlockNum = Ops::Base::CeilDiv(tilingParams.colSize, tilingParams.blockSizeCol);

    return ge::GRAPH_SUCCESS;
}

std::set<int64_t> FindShapeCut(int64_t usedCoreNum)
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

ge::graphStatus DynamicDualLevelMxQuantTiling::AutoTiling()
{
    OP_LOGD("AutoTiling", "DynamicDualLevelMxQuant AutoTiling Enter.");

    // 计算可用核数
    tilingParams.usedCoreNum = std::min(tilingParams.totalCoreNum, tilingParams.rowBlockNum * tilingParams.colBlockNum);
    tilingParams.usedCoreNum = tilingParams.usedCoreNum == 0 ? 1 : tilingParams.usedCoreNum;

    // 查找切分的组合
    std::set<int64_t> cutSet = FindShapeCut(tilingParams.usedCoreNum);

    std::vector<std::vector<int64_t>> allTiling;

    // 行方向切分，枚举 m 的取值
    for (int64_t m : cutSet) {
        if (m > tilingParams.colBlockNum) {
            continue;
        }

        int64_t n = tilingParams.usedCoreNum / m;
        n = n < 1 ? 1 : n;
        if (n > tilingParams.rowBlockNum) {
            continue;
        }

        int64_t colNormalBlock = Ops::Base::CeilDiv(tilingParams.colBlockNum, m);
        int64_t rowNormalBlock = Ops::Base::CeilDiv(tilingParams.rowBlockNum, n);
        OP_CHECK_IF(
            (colNormalBlock == 0 || rowNormalBlock == 0), OP_LOGE(context_->GetNodeName(), "x is empty tensor."),
            return ge::GRAPH_FAILED);

        int64_t delta = rowNormalBlock * colNormalBlock;
        if (m * n == static_cast<int64_t>(tilingParams.usedCoreNum)) {
            if (tilingParams.colBlockNum % m == 0 && tilingParams.rowBlockNum % n == 0) {
                delta = 0;
            } else if (tilingParams.colBlockNum % m == 0) {
                delta = delta - colNormalBlock * (tilingParams.rowBlockNum % rowNormalBlock);
            } else if (tilingParams.rowBlockNum % n == 0) {
                delta = delta - (tilingParams.colBlockNum % colNormalBlock) * n;
            } else {
                delta =
                    delta - (tilingParams.colBlockNum % colNormalBlock) * (tilingParams.rowBlockNum % rowNormalBlock);
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

    tilingParams.colTileNum = static_cast<uint16_t>(allTiling[0][0]);
    tilingParams.rowTileNum = static_cast<uint16_t>(allTiling[0][1]);
    return ge::GRAPH_SUCCESS;
}

// 切块
void DynamicDualLevelMxQuantTiling::SplitCore()
{
    // 计算ub能处理多少个基本块
    uint64_t xUbSize = BLOCK_SIZE * BYTES_OF_INPUT_TYPE * N_BUFFER;
    uint64_t yUbSize = BLOCK_SIZE / DIGIT_TWO * N_BUFFER;
    uint64_t tempXUbSize = BLOCK_SIZE * BYTES_OF_INPUT_TYPE;
    // ceilAlign(512/512,32)
    uint64_t level0ScaleUbSize = DIGIT_32 * N_BUFFER;
    // ceilAlign(512/32,32)
    uint64_t level1ScaleUbSize = DIGIT_32 * N_BUFFER;
    // ceilAlign(2*512/32,32)
    uint64_t level1ScaleReciprocalUbSize = DIGIT_32;
    tilingParams.ubFactor =
        (tilingParams.ubSize - RESERVED_UB_SIZE) /
        (xUbSize + yUbSize + tempXUbSize + level0ScaleUbSize + level1ScaleUbSize + level1ScaleReciprocalUbSize);
}

ge::graphStatus DynamicDualLevelMxQuantTiling::SetTilingParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicDualLevelMxQuantTiling SetTilingParams.");

    tilingParams.level0BlockSize = DEFAULT_LEVEL0_BLOCK_SIZE;
    tilingParams.level1BlockSize = DEFAULT_LEVEL1_BLOCK_SIZE;
    tilingParams.blockSizeRow = BLOCK_SIZE;
    tilingParams.blockSizeCol = DIGIT_ONE;
    MergeAxis();
    AutoTiling();
    SplitCore();

    // 分核分块后，存在部分核处理的块大小为0，减少核数，再均分
    tilingParams.normalTileRowBlockNum = Ops::Base::CeilDiv(tilingParams.rowBlockNum, tilingParams.rowTileNum);
    tilingParams.normalTileColBlockNum = Ops::Base::CeilDiv(tilingParams.colBlockNum, tilingParams.colTileNum);

    tilingParams.rowTileNum = Ops::Base::CeilDiv(tilingParams.rowBlockNum, tilingParams.normalTileRowBlockNum);
    tilingParams.colTileNum = Ops::Base::CeilDiv(tilingParams.colBlockNum, tilingParams.normalTileColBlockNum);

    tilingParams.tailTileRowBlockNum =
        tilingParams.rowBlockNum - (tilingParams.rowTileNum - 1) * tilingParams.normalTileRowBlockNum;
    tilingParams.tailTileColBlockNum =
        tilingParams.colBlockNum - (tilingParams.colTileNum - 1) * tilingParams.normalTileColBlockNum;

    tilingParams.usedCoreNum = tilingParams.rowTileNum * tilingParams.colTileNum;

    tilingParams.normalTileRowSize = tilingParams.normalTileRowBlockNum * BLOCK_SIZE;
    tilingParams.tailTileRowSize =
        tilingParams.rowSize - (tilingParams.rowTileNum - 1) * tilingParams.normalTileRowSize;

    // 计算核内循环数
    if (tilingParams.normalTileRowBlockNum < tilingParams.ubFactor) {
        tilingParams.normalTileRowLoopNum = DIGIT_ONE;
        tilingParams.normalTileColLoopNum = Ops::Base::CeilDiv(
            tilingParams.normalTileColBlockNum, tilingParams.ubFactor / tilingParams.normalTileRowBlockNum);
        tilingParams.copyMethod = COPY_MULTIPLE_ROW;
    } else {
        tilingParams.normalTileRowLoopNum =
            Ops::Base::CeilDiv(tilingParams.normalTileRowBlockNum, tilingParams.ubFactor);
        tilingParams.normalTileColLoopNum = tilingParams.normalTileColBlockNum;
        tilingParams.copyMethod = COPY_SINGLE_ROW;
    }

    if (tilingParams.tailTileRowBlockNum < tilingParams.ubFactor) {
        tilingParams.tailTileRowLoopNum = DIGIT_ONE;
        tilingParams.tailTileColLoopNum = Ops::Base::CeilDiv(
            tilingParams.tailTileColBlockNum, tilingParams.ubFactor / tilingParams.normalTileRowBlockNum);
    } else {
        tilingParams.tailTileRowLoopNum = Ops::Base::CeilDiv(tilingParams.tailTileRowBlockNum, tilingParams.ubFactor);
        tilingParams.tailTileColLoopNum = tilingParams.tailTileColBlockNum;
    }
    tilingParams.tailAlignNum = Ops::Base::CeilDiv(tilingParams.tailTileRowSize % BLOCK_SIZE, DIGIT_128);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicDualLevelMxQuantTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicDualLevelMxQuantTiling DoTiling.");

    OP_CHECK_IF(
        GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The platforminfo get failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The data type check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The attr get failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetTilingParams() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "DynamicDualLevelMxQuantTiling SetTilingParams failed"),
        return ge::GRAPH_FAILED);

    SetTilingKey();
    SetTilingData();
    PrintTilingData();

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    if (tilingData.GetDataSize() > rawTilingData->GetCapacity()) {
        OP_LOGE(context_->GetNodeName(), "Tiling DataSize Greater than capacity, please check.");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %ld.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams.workspaceSize;

    return ge::GRAPH_SUCCESS;
}

void DynamicDualLevelMxQuantTiling::SetTilingKey()
{
    if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_ROUND)) {
        roundMode_ = TPL_ROUND;
    } else if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_FLOOR)) {
        roundMode_ = TPL_FLOOR;
    } else if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_RINT)) {
        roundMode_ = TPL_RINT;
    }

    int64_t tilingKey = GET_TPL_TILING_KEY(roundMode_);
    OP_LOGD(context_->GetNodeName(), "roundMode is %ld", roundMode_);
    context_->SetTilingKey(tilingKey);
}

void DynamicDualLevelMxQuantTiling::SetTilingData()
{
    tilingData.set_tilingKey(tilingParams.tilingKey);
    tilingData.set_totalCoreNum(tilingParams.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParams.usedCoreNum);
    tilingData.set_roundMode(tilingParams.roundMode);
    tilingData.set_level0BlockSize(tilingParams.level0BlockSize);
    tilingData.set_level1BlockSize(tilingParams.level1BlockSize);
    tilingData.set_rowSize(tilingParams.rowSize);
    tilingData.set_colSize(tilingParams.colSize);
    tilingData.set_blockSizeRow(tilingParams.blockSizeRow);
    tilingData.set_blockSizeCol(tilingParams.blockSizeCol);
    tilingData.set_rowBlockNum(tilingParams.rowBlockNum);
    tilingData.set_colBlockNum(tilingParams.colBlockNum);
    tilingData.set_rowTileNum(tilingParams.rowTileNum);
    tilingData.set_colTileNum(tilingParams.colTileNum);
    tilingData.set_normalTileRowBlockNum(tilingParams.normalTileRowBlockNum);
    tilingData.set_normalTileColBlockNum(tilingParams.normalTileColBlockNum);
    tilingData.set_tailTileRowBlockNum(tilingParams.tailTileRowBlockNum);
    tilingData.set_tailTileColBlockNum(tilingParams.tailTileColBlockNum);
    tilingData.set_normalTileRowSize(tilingParams.normalTileRowSize);
    tilingData.set_tailTileRowSize(tilingParams.tailTileRowSize);
    tilingData.set_normalTileRowLoopNum(tilingParams.normalTileRowLoopNum);
    tilingData.set_normalTileColLoopNum(tilingParams.normalTileColLoopNum);
    tilingData.set_tailTileRowLoopNum(tilingParams.tailTileRowLoopNum);
    tilingData.set_tailTileColLoopNum(tilingParams.tailTileColLoopNum);
    tilingData.set_ubFactor(tilingParams.ubFactor);
    tilingData.set_tailAlignNum(tilingParams.tailAlignNum);
    tilingData.set_copyMethod(tilingParams.copyMethod);
}

void DynamicDualLevelMxQuantTiling::PrintTilingData()
{
    OP_LOGD(
        context_->GetNodeName(),
        "TilingData tilingKey: %ld, totalCoreNum: %ld, usedCoreNum: %ld, roundMode: %ld, "
        "level0BlockSize: %ld, level1BlockSize: %ld, rowSize: %ld, colSize: %ld, blockSizeRow: %ld, blockSizeCol: %ld, "
        "rowBlockNum: %ld, colBlockNum: %ld, rowTileNum: %ld, "
        "colTileNum: %ld, normalTileRowBlockNum: %ld, normalTileColBlockNum: %ld, "
        "tailTileRowBlockNum: %ld, tailTileColBlockNum: %ld, normalTileRowSize: %ld, "
        "tailTileRowSize: %ld, normalTileRowLoopNum: %ld, normalTileColLoopNum: %ld, tailTileRowLoopNum: %ld, "
        "tailTileColLoopNum: %ld, ubFactor: %ld, tailAlignNum: %ld ,copyMethod: %ld ",
        tilingData.get_tilingKey(), tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(),
        tilingData.get_roundMode(), tilingData.get_level0BlockSize(), tilingData.get_level1BlockSize(),
        tilingData.get_rowSize(), tilingData.get_colSize(), tilingData.get_blockSizeRow(),
        tilingData.get_blockSizeCol(), tilingData.get_rowBlockNum(), tilingData.get_colBlockNum(),
        tilingData.get_rowTileNum(), tilingData.get_colTileNum(), tilingData.get_normalTileRowBlockNum(),
        tilingData.get_normalTileColBlockNum(), tilingData.get_tailTileRowBlockNum(),
        tilingData.get_tailTileColBlockNum(), tilingData.get_normalTileRowSize(), tilingData.get_tailTileRowSize(),
        tilingData.get_normalTileRowLoopNum(), tilingData.get_normalTileColLoopNum(),
        tilingData.get_tailTileRowLoopNum(), tilingData.get_tailTileColLoopNum(), tilingData.get_ubFactor(),
        tilingData.get_tailAlignNum(), tilingData.get_copyMethod());
}

static ge::graphStatus TilingForDynamicDualLevelMxQuant(gert::TilingContext* context)
{
    OP_LOGD("DynamicDualLevelMxQuantTiling", "Enter TilingForDynamicDualLevelMxQuantTiling");

    OP_CHECK_IF(
        context == nullptr, OP_LOGE("DynamicDualLevelMxQuantTiling", "Tiling context is null."),
        return ge::GRAPH_FAILED);

    DynamicDualLevelMxQuantTiling dualLevelMxQunatTiling(context);
    return dualLevelMxQunatTiling.DoTiling();
}

static ge::graphStatus TilingPrepareForDynamicDualLevelMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD("DynamicDualLevelMxQuantTiling", "Enter TilingPrepareForDynamicDualLevelMxQuantTiling");

    OP_CHECK_IF(
        context == nullptr, OP_LOGE("DynamicDualLevelMxQuantTiling", "TilingParse context is null."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicDualLevelMxQuant)
    .Tiling(TilingForDynamicDualLevelMxQuant)
    .TilingParse<DynamicDualLevelMxQuantCompileInfo>(TilingPrepareForDynamicDualLevelMxQuant);

} // namespace optiling