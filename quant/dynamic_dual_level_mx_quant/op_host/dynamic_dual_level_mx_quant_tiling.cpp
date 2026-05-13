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
#include "op_host/tiling_templates_registry.h"

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
constexpr int64_t UB_ALIGN_BYTES = 32;
constexpr int64_t LEVEL1_SCALE_DIVISOR = 64;
constexpr int64_t TAIL_ALIGN_BYTES = 128;
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

template <typename T>
ge::graphStatus DynamicDualLevelMxQuantTiling::GetAndValidateAttr(
    const gert::RuntimeAttrs* attrs, int64_t index, T& target, T expectedValue, const std::string& attrName,
    const std::string& expectedMsg) const
{
    auto* attrPtr = attrs->GetAttrPointer<T>(index);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
    target = static_cast<T>(*attrPtr);
    OP_CHECK_IF(
        target != expectedValue,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), attrName.c_str(), std::to_string(target).c_str(), expectedMsg.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "round_mode",
            attrRoundMode, "The value of round_mode must be in {rint, floor, round}"),
        return ge::GRAPH_FAILED);
    tilingParams.roundMode = static_cast<int64_t>(roundMode);

    OP_CHECK_IF(
        GetAndValidateAttr(
            attrs, INDEX_ATTR_LEVEL0_BLOCK_SIZE, tilingParams.level0BlockSize, DEFAULT_LEVEL0_BLOCK_SIZE,
            "level0_block_size", "level0_block_size should be 512") != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "level0_block_size is invalid, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAndValidateAttr(
            attrs, INDEX_ATTR_LEVEL1_BLOCK_SIZE, tilingParams.level1BlockSize, DEFAULT_LEVEL1_BLOCK_SIZE,
            "level1_block_size", "level1_block_size should be 32") != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "level1_block_size is invalid, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicDualLevelMxQuantTiling::CheckRequiredDtype() const
{
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), "FLOAT16 or BFLOAT16"),
        return ge::GRAPH_FAILED);
    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(), "FLOAT4_E2M1"),
        return ge::GRAPH_FAILED);

    auto outputLevel0ScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputLevel0ScalePtr);
    auto Level0ScaleDtype = outputLevel0ScalePtr->GetDataType();
    OP_CHECK_IF(
        LEVEL0_SCALE_OUTPUT_SUPPORT_DTYPE_SET.count(Level0ScaleDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "level0_scale", ge::TypeUtils::DataTypeToSerialString(Level0ScaleDtype).c_str(),
            "DT_FLOAT"),
        return ge::GRAPH_FAILED);

    auto outputLevel1ScalePtr = context_->GetOutputDesc(DIGIT_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputLevel1ScalePtr);
    auto Level1ScaleDtype = outputLevel1ScalePtr->GetDataType();
    OP_CHECK_IF(
        LEVEL1_SCALE_OUTPUT_SUPPORT_DTYPE_SET.count(Level1ScaleDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "level1_scale", ge::TypeUtils::DataTypeToSerialString(Level1ScaleDtype).c_str(),
            "FLOAT8_E8M0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicDualLevelMxQuantTiling::CheckRequiredShape() const
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    dimNum_ = xShape.GetDimNum();
    lastDim_ = xShape.GetDim(dimNum_ - 1);

    OP_CHECK_IF(
        lastDim_ % DIGIT_TWO != 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(lastDim_).c_str(),
            "The last axis of x should be even when the dtype of y is FLOAT4_E2M1"),
        return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "x and y",
            (Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(yShape)).c_str(),
            "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);

    auto level0ScaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, level0ScaleShapePtr);
    auto level0ScaleShape = level0ScaleShapePtr->GetStorageShape();

    auto level1ScaleShapePtr = context_->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, level1ScaleShapePtr);
    auto level1ScaleShape = level1ScaleShapePtr->GetStorageShape();

    auto newScale0Shape = xShape;
    newScale0Shape.SetDim(dimNum_ - 1, Ops::Base::CeilDiv(lastDim_, BLOCK_SIZE));
    OP_CHECK_IF(
        newScale0Shape != level0ScaleShape,
        OP_LOGE_FOR_INVALID_SHAPE(
            context_->GetNodeName(), "level0_scale", Ops::Base::ToString(level0ScaleShape).c_str(),
            Ops::Base::ToString(newScale0Shape).c_str()),
        return ge::GRAPH_FAILED);

    auto newScale1Shape = xShape;
    newScale1Shape.SetDim(dimNum_ - 1, Ops::Base::CeilDiv(lastDim_, LEVEL1_SCALE_DIVISOR));
    newScale1Shape.AppendDim(DIGIT_TWO);
    OP_CHECK_IF(
        newScale1Shape != level1ScaleShape,
        OP_LOGE_FOR_INVALID_SHAPE(
            context_->GetNodeName(), "level1_scale", Ops::Base::ToString(level1ScaleShape).c_str(),
            Ops::Base::ToString(newScale1Shape).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicDualLevelMxQuantTiling::CheckSmoothScaleDtypeShape()
{
    auto x = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    auto xDtype = x->GetDataType();
    auto xShape = x->GetStorageShape();
    auto smoothScale = context_->GetOptionalInputTensor(1);
    if (smoothScale != nullptr) {
        auto smoothScaleDtype = smoothScale->GetDataType();
        OP_CHECK_IF(
            smoothScaleDtype != xDtype,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "smooth_scale and x",
                (ge::TypeUtils::DataTypeToSerialString(smoothScaleDtype) + " and " +
                 ge::TypeUtils::DataTypeToSerialString(xDtype)).c_str(),
                "The dtypes of smooth_scale and x must be the same"),
            return ge::GRAPH_FAILED);
        auto smoothScaleShape = smoothScale->GetStorageShape();
        OP_CHECK_IF(
            smoothScaleShape.IsScalar() || smoothScaleShape.GetDimNum() != 1 || smoothScaleShape.GetDim(0) != lastDim_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "smooth_scale", Ops::Base::ToString(smoothScaleShape).c_str(),
                ("smooth_scale can NOT be scalar and the shape size of smooth_scale should be same as the last axis of x " +
                 std::to_string(lastDim_)).c_str()),
            return ge::GRAPH_FAILED);
        needSmoothScale = true;
    }
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

    for (size_t i = 0; i < dimNum_ - 1; i++) {
        tilingParams.colSize *= xShape.GetDim(i);
    }
    tilingParams.rowSize = lastDim_;
    tilingParams.rowBlockNum = Ops::Base::CeilDiv(tilingParams.rowSize, tilingParams.blockSizeRow);
    tilingParams.colBlockNum = Ops::Base::CeilDiv(tilingParams.colSize, tilingParams.blockSizeCol);

    return ge::GRAPH_SUCCESS;
}

static std::set<int64_t> FindShapeCut(int64_t usedCoreNum)
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
    OP_LOGD(context_->GetNodeName(), "DynamicDualLevelMxQuant AutoTiling Enter.");

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
ge::graphStatus DynamicDualLevelMxQuantTiling::SplitCore()
{
    // 计算ub能处理多少个基本块
    uint64_t xUbSize = BLOCK_SIZE * BYTES_OF_INPUT_TYPE * N_BUFFER;
    uint64_t yUbSize = BLOCK_SIZE / DIGIT_TWO * N_BUFFER;
    uint64_t tempXUbSize = BLOCK_SIZE * BYTES_OF_INPUT_TYPE;
    uint64_t smoothScaleUbSize = needSmoothScale ? BLOCK_SIZE * BYTES_OF_INPUT_TYPE * N_BUFFER : 0;
    // ceilAlign(512/512,32)
    uint64_t level0ScaleUbSize = UB_ALIGN_BYTES * N_BUFFER;
    // ceilAlign(512/32,32)
    uint64_t level1ScaleUbSize = UB_ALIGN_BYTES * N_BUFFER;
    // ceilAlign(2*512/32,32)
    uint64_t level1ScaleReciprocalUbSize = UB_ALIGN_BYTES;
    tilingParams.ubFactor = (tilingParams.ubSize - RESERVED_UB_SIZE) /
                            (xUbSize + yUbSize + tempXUbSize + smoothScaleUbSize + level0ScaleUbSize +
                             level1ScaleUbSize + level1ScaleReciprocalUbSize);
    OP_CHECK_IF(
        tilingParams.ubSize <= RESERVED_UB_SIZE || tilingParams.ubFactor <= 0,
        OP_LOGE(
            context_->GetNodeName(), "Invalid ubSize/ubFactor for SplitCore: ubSize: %ld, ubFactor: %ld",
            tilingParams.ubSize, tilingParams.ubFactor),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
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
    if (SplitCore() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

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
    tilingParams.tailAlignNum = Ops::Base::CeilDiv(tilingParams.tailTileRowSize % BLOCK_SIZE, TAIL_ALIGN_BYTES);
    tilingParams.needSmoothScale = needSmoothScale;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicDualLevelMxQuantTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicDualLevelMxQuantTiling DoTiling.");

    OP_CHECK_IF(
        GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The platforminfo get failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckRequiredDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The data type check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The attr get failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckRequiredShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckSmoothScaleDtypeShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetTilingParams() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "DynamicDualLevelMxQuantTiling SetTilingParams failed"),
        return ge::GRAPH_FAILED);

    tilingData = context_->GetTilingData<DynamicDualLevelMxQuantTilingData>();
    OP_CHECK_IF(
        (tilingData == nullptr),
        OP_LOGE(context_->GetNodeName(), "Get DynamicDualLevelMxQuantTilingData from context failed"),
        return ge::GRAPH_FAILED);

    SetTilingKey();
    SetTilingData();
    PrintTilingData();

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
    tilingData->tilingKey = tilingParams.tilingKey;
    tilingData->totalCoreNum = tilingParams.totalCoreNum;
    tilingData->usedCoreNum = tilingParams.usedCoreNum;
    tilingData->roundMode = tilingParams.roundMode;
    tilingData->level0BlockSize = tilingParams.level0BlockSize;
    tilingData->level1BlockSize = tilingParams.level1BlockSize;
    tilingData->rowSize = tilingParams.rowSize;
    tilingData->colSize = tilingParams.colSize;
    tilingData->blockSizeRow = tilingParams.blockSizeRow;
    tilingData->blockSizeCol = tilingParams.blockSizeCol;
    tilingData->rowBlockNum = tilingParams.rowBlockNum;
    tilingData->colBlockNum = tilingParams.colBlockNum;
    tilingData->rowTileNum = tilingParams.rowTileNum;
    tilingData->colTileNum = tilingParams.colTileNum;
    tilingData->normalTileRowBlockNum = tilingParams.normalTileRowBlockNum;
    tilingData->normalTileColBlockNum = tilingParams.normalTileColBlockNum;
    tilingData->tailTileRowBlockNum = tilingParams.tailTileRowBlockNum;
    tilingData->tailTileColBlockNum = tilingParams.tailTileColBlockNum;
    tilingData->normalTileRowSize = tilingParams.normalTileRowSize;
    tilingData->tailTileRowSize = tilingParams.tailTileRowSize;
    tilingData->normalTileRowLoopNum = tilingParams.normalTileRowLoopNum;
    tilingData->normalTileColLoopNum = tilingParams.normalTileColLoopNum;
    tilingData->tailTileRowLoopNum = tilingParams.tailTileRowLoopNum;
    tilingData->tailTileColLoopNum = tilingParams.tailTileColLoopNum;
    tilingData->ubFactor = tilingParams.ubFactor;
    tilingData->tailAlignNum = tilingParams.tailAlignNum;
    tilingData->copyMethod = tilingParams.copyMethod;
    tilingData->needSmoothScale = tilingParams.needSmoothScale;
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
        "tailTileColLoopNum: %ld, ubFactor: %ld, tailAlignNum: %ld, copyMethod: %ld, needSmoothScale: %ld",
        tilingData->tilingKey, tilingData->totalCoreNum, tilingData->usedCoreNum, tilingData->roundMode,
        tilingData->level0BlockSize, tilingData->level1BlockSize, tilingData->rowSize, tilingData->colSize,
        tilingData->blockSizeRow, tilingData->blockSizeCol, tilingData->rowBlockNum, tilingData->colBlockNum,
        tilingData->rowTileNum, tilingData->colTileNum, tilingData->normalTileRowBlockNum,
        tilingData->normalTileColBlockNum, tilingData->tailTileRowBlockNum, tilingData->tailTileColBlockNum,
        tilingData->normalTileRowSize, tilingData->tailTileRowSize, tilingData->normalTileRowLoopNum,
        tilingData->normalTileColLoopNum, tilingData->tailTileRowLoopNum, tilingData->tailTileColLoopNum,
        tilingData->ubFactor, tilingData->tailAlignNum, tilingData->copyMethod, tilingData->needSmoothScale);
}

static ge::graphStatus TilingForDynamicDualLevelMxQuant(gert::TilingContext* context)
{
    OP_LOGD("DynamicDualLevelMxQuantTiling", "Enter TilingForDynamicDualLevelMxQuantTiling");

    OP_CHECK_IF(
        context == nullptr, OP_LOGE("DynamicDualLevelMxQuantTiling", "Tiling context is null."), return ge::GRAPH_FAILED);

    DynamicDualLevelMxQuantTiling dualLevelMxQunatTiling(context);
    return dualLevelMxQunatTiling.DoTiling();
}

static ge::graphStatus TilingPrepareForDynamicDualLevelMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD("DynamicDualLevelMxQuantTiling", "Enter TilingPrepareForDynamicDualLevelMxQuantTiling");

    OP_CHECK_IF(
        context == nullptr, OP_LOGE("DynamicDualLevelMxQuantTiling", "TilingParse context is null."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicDualLevelMxQuant)
    .Tiling(TilingForDynamicDualLevelMxQuant)
    .TilingParse<DynamicDualLevelMxQuantCompileInfo>(TilingPrepareForDynamicDualLevelMxQuant);

} // namespace optiling