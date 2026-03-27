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
 * \file dynamic_block_mx_quant_tiling.cpp
 * \brief
 */

#include "dynamic_block_mx_quant_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "util/math_util.h"
#include "quant/dynamic_block_mx_quant/op_kernel/arch35/dynamic_block_mx_quant_struct.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace DynamicBlockMxQuantOp;

namespace optiling {
constexpr int64_t INDEX_ATTR_ROUND_MODE = 0;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 1;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 2;
constexpr int64_t INDEX_ATTR_DST_TYPE_MAX = 3;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t BYTES_OF_RECIPROCAL_SCALE_TYPE = 2;
constexpr int64_t BYTES_OF_OUTPUT_Y_TYPE = 1;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THOUSAND = 1000;
constexpr int64_t DIGIT_HUNDRED = 100;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t EXIST_NODE_NUM = 3;
constexpr int64_t AXIS_NUM_AFTER_MERGE = 3;
constexpr int64_t NEW_SHAPE_INDEX_TWO = 2;
constexpr int64_t WORKSPACE_SIZE = 0;
constexpr int64_t DIGIT_ZERO = 0;
constexpr float DIGIT_ZERO_FLOAT = 0.0;
constexpr float DIGIT_SIX_FLOAT = 6.0;
constexpr float DIGIT_SEVEN_FLOAT = 7.0;
constexpr int64_t DIGIT_32 = 32;
constexpr int64_t DIGIT_64 = 64;
constexpr int64_t SCALE1_COL = 32;
constexpr int64_t SCALE2_ROW = 2;
constexpr int64_t gatherCommandRequire = 256;
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> INPUT_FP16_DTYPE_SET = {ge::DT_FLOAT16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> MX_SCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BASE_BLOCK_SIZE_ROW = 64;
constexpr int64_t BASE_BLOCK_SIZE_COL = 256;
constexpr int64_t BLOCK_SIZE_ROW = 32;
constexpr int64_t BLOCK_SIZE_COL = 32;
constexpr int64_t TAIL_TILING_KEY_DIGIT = 4;
constexpr int64_t SINGLE_LOOP_MIN_COLS = 128;
constexpr int64_t RESERVED_UB_SIZE = 1024; // 预留空间

RoundModeList DynamicBlockMxQuantTiling::GetRoundMode(const std::string& roundMode)
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

// roundMode, dstType
ge::graphStatus DynamicBlockMxQuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
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
    OP_CHECK_IF(
        (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype) != 0 && roundMode != RoundModeList::MODE_RINT),
        OP_LOGE(
            context_->GetNodeName(),
            "When output y's data type is FLOAT8_E4M3FN/FLOAT8_E5M2, round_mode:[%s] only support rint, please check.",
            attrRoundMode),
        return ge::GRAPH_FAILED);
    tilingParams.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrDstType);
    tilingParams.dstType = static_cast<int64_t>(*attrDstType);
    int checkDstType = static_cast<int>(*attrDstType);
    OP_CHECK_IF(
        (yDtype == ge::DT_FLOAT4_E2M1 && checkDstType != 40) || (yDtype == ge::DT_FLOAT4_E1M2 && checkDstType != 41) ||
            (yDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
            (yDtype == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
        OP_LOGE(
            context_->GetNodeName(),
            "y's data type:[%s] and dst_type:[%ld] is not corresponded, y's data type: "
            "FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 correspond to dst_type: 40/41/36/35, please check.",
            Ops::Base::ToString(yDtype).c_str(), tilingParams.dstType),
        return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrScaleAlg);
    tilingParams.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    OP_CHECK_IF(
        tilingParams.scaleAlg != DIGIT_ZERO && tilingParams.scaleAlg != DIGIT_TWO,
        OP_LOGE(context_->GetNodeName(), "The scaleAlg:[%ld] should be 0 or 2, please check.", tilingParams.scaleAlg),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingParams.scaleAlg != DIGIT_ZERO) &&
            (yDtype == ge::DT_FLOAT8_E4M3FN || yDtype == ge::DT_FLOAT8_E5M2 || yDtype == ge::DT_FLOAT4_E1M2),
        OP_LOGE(
            context_->GetNodeName(),
            "When y's data type:[%s] is FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 , scale_alg:[%ld] only support 0 "
            "currently.",
            Ops::Base::ToString(yDtype).c_str(), tilingParams.scaleAlg),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (yDtype == ge::DT_FLOAT4_E2M1 && tilingParams.scaleAlg != DIGIT_ZERO && tilingParams.scaleAlg != DIGIT_TWO),
        OP_LOGE(
            context_->GetNodeName(), "When y's data type is FLOAT4_E2M1, scale_alg:[%ld] only support 0 or 2 currently",
            tilingParams.scaleAlg),
        return ge::GRAPH_FAILED);

    auto* attrDstTypeMax = attrs->GetAttrPointer<float>(INDEX_ATTR_DST_TYPE_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrDstTypeMax);
    tilingParams.dstTypeMax = static_cast<float>(*attrDstTypeMax);
    OP_CHECK_IF(
        (tilingParams.dstTypeMax != DIGIT_ZERO_FLOAT &&
         (tilingParams.scaleAlg != DIGIT_TWO || yDtype != ge::DT_FLOAT4_E2M1)),
        OP_LOGE(
            context_->GetNodeName(),
            "The dstTypeMax:[%f] must be non-zero value only when scaleAlg:[%ld] equals 2 and "
            "dstType:[%s] is FLOAT4_E2M1 ",
            tilingParams.dstTypeMax, tilingParams.scaleAlg, Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingParams.scaleAlg == DIGIT_TWO && yDtype == ge::DT_FLOAT4_E2M1 &&
         (tilingParams.dstTypeMax != DIGIT_ZERO_FLOAT && tilingParams.dstTypeMax != DIGIT_SIX_FLOAT &&
          tilingParams.dstTypeMax != DIGIT_SEVEN_FLOAT)),
        OP_LOGE(
            context_->GetNodeName(),
            "When scaleAlg is 2 and dtype is FLOAT4_E2M1, dstTypeMax:[%f] only support 0.0/6.0/7.0.",
            tilingParams.dstTypeMax),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockMxQuantTiling::CheckDtype() const
{
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Input x's data type only support FLOAT16 and BFLOAT16 currently, but x is:[%s], please check.",
            Ops::Base::ToString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output y's data type only support FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 currently, but y is "
            "[%s], please check.",
            Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputScale1Ptr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScale1Ptr);
    auto scale1Dtype = outputScale1Ptr->GetDataType();
    OP_CHECK_IF(
        MX_SCALE_SUPPORT_DTYPE_SET.count(scale1Dtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output scale1's data type only support FLOAT8_E8M0 currently, but scale is [%s], please check.",
            Ops::Base::ToString(scale1Dtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputScale2Ptr = context_->GetOutputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScale2Ptr);
    auto scale2Dtype = outputScale2Ptr->GetDataType();
    OP_CHECK_IF(
        MX_SCALE_SUPPORT_DTYPE_SET.count(scale2Dtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output scale2's data type only support FLOAT8_E8M0 currently, but scale is [%s], please check.",
            Ops::Base::ToString(scale2Dtype).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockMxQuantTiling::CheckShape() const
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    size_t xSizeNum = xShape.GetShapeSize();
    if (xSizeNum == 0ULL) {
        OP_LOGE(context_->GetNodeName(), "dynamic_block_mx_quant does not support empty tensor.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        xShape.GetDimNum() != 2 && xShape.GetDimNum() != 3,
        OP_LOGE(
            context_->GetNodeName(), "The shape is invalid, axis num:[%ld] should be 2 or 3, please check.",
            static_cast<int64_t>(xShape.GetDimNum())),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    OP_CHECK_IF(
        xShape.GetDim(xShape.GetDimNum() - 1) % DIGIT_TWO != 0 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0,
        OP_LOGE(
            context_->GetNodeName(),
            "When output y's data type is FLOAT4_E2M1/FLOAT4_E1M2, the last axis should be even, please check."),
        return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of output y:[%s] must be same with shape of input x:[%s], please check.",
            Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
        return ge::GRAPH_FAILED);

    auto scale1ShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale1ShapePtr);
    auto scale1Shape = scale1ShapePtr->GetStorageShape();

    auto scale2ShapePtr = context_->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale2ShapePtr);
    auto scale2Shape = scale2ShapePtr->GetStorageShape();

    auto newScale1Shape = xShape;
    newScale1Shape.SetDim(
        xShape.GetDimNum() - DIGIT_ONE, Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - DIGIT_ONE), DIGIT_64));
    newScale1Shape.AppendDim(DIGIT_TWO);

    auto newScale2Shape = xShape;
    newScale2Shape.SetDim(
        xShape.GetDimNum() - DIGIT_TWO, Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO), DIGIT_64));
    newScale2Shape.AppendDim(DIGIT_TWO);

    OP_CHECK_IF(
        newScale1Shape != scale1Shape,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of output scale1:[%s] is incorrect, correct shape is:[%s], please check.",
            Shape2String(scale1Shape).c_str(), Shape2String(newScale1Shape).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        newScale2Shape != scale2Shape,
        OP_LOGE(
            context_->GetNodeName(),
            "The shape of output scale2:[%s] is incorrect, correct shape is:[%s], please check.",
            Shape2String(scale2Shape).c_str(), Shape2String(newScale2Shape).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// totalCoreNum，ubSize，workspaceSize
ge::graphStatus DynamicBlockMxQuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicBlockMxQuantTiling GetPlatformInfo.");

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

ge::graphStatus DynamicBlockMxQuantTiling::CalcAxisSize()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    if (xShape.GetDimNum() == 2) {
        tilingParams.batchNum = 1;
        tilingParams.rowNum = xShape.GetDim(0);
        tilingParams.colNum = xShape.GetDim(xShape.GetDimNum() - 1);
    } else {
        tilingParams.batchNum = xShape.GetDim(0);
        tilingParams.rowNum = xShape.GetDim(1);
        tilingParams.colNum = xShape.GetDim(2);
    }

    tilingParams.singleBatchRowBlockLoopNum = Ops::Base::CeilDiv(tilingParams.rowNum, tilingParams.blockH);
    tilingParams.rowBlockLoopNum = tilingParams.singleBatchRowBlockLoopNum * tilingParams.batchNum;
    tilingParams.colBlockLoopNum = Ops::Base::CeilDiv(tilingParams.colNum, tilingParams.blockW);

    return ge::GRAPH_SUCCESS;
}

// ub能处理的基本块
void DynamicBlockMxQuantTiling::SplitCore()
{
    // 每个block需要的临时ub大小
    int64_t perBlockTmpUbSize = 0;
    // 每个block需要的ub大小
    int64_t perBlockUbSize = 0;
    // input and output size
    perBlockTmpUbSize += tilingParams.blockH * tilingParams.blockW * (BYTES_OF_INPUT_TYPE + BYTES_OF_OUTPUT_Y_TYPE);
    // scale size
    perBlockTmpUbSize += tilingParams.blockH * BLOCK_SIZE + tilingParams.blockH / BLOCK_SIZE * tilingParams.blockW;
    // double buffer
    perBlockTmpUbSize *= N_BUFFER;
    // 1/scale && gather comand require
    int64_t scaleAlignedBlockW = Ops::Base::CeilAlign((tilingParams.blockW / BLOCK_SIZE), BLOCK_SIZE);
    perBlockUbSize =
        perBlockTmpUbSize + (tilingParams.blockH / BLOCK_SIZE) * scaleAlignedBlockW * BYTES_OF_RECIPROCAL_SCALE_TYPE;

    perBlockUbSize = perBlockUbSize != 0 ? perBlockUbSize : 1;

    int64_t maxUbAvailable = (tilingParams.ubSize - gatherCommandRequire - RESERVED_UB_SIZE) / perBlockUbSize;
    // normalCoreColTileNum:一个核列方向的基本块数
    // maxUbAvailable: Ub能放入的基本块数量
    // ub内一行可以搬多少个block
    tilingParams.colUbBlockLoopNum =
        maxUbAvailable < tilingParams.normalCoreColTileNum ? maxUbAvailable : tilingParams.normalCoreColTileNum;
    maxUbAvailable = maxUbAvailable / tilingParams.colUbBlockLoopNum;
    tilingParams.rowUbBlockLoopNum =
        maxUbAvailable > tilingParams.normalCoreRowTileNum ? tilingParams.normalCoreRowTileNum : maxUbAvailable;
    tilingParams.rowUbFactor = tilingParams.rowUbBlockLoopNum * tilingParams.blockH;
    tilingParams.colUbFactor = tilingParams.colUbBlockLoopNum * tilingParams.blockW;
}

inline std::set<int64_t> FindUniqueCut(int64_t usedCoreNum)
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

ge::graphStatus DynamicBlockMxQuantTiling::AutoTiling()
{
    OP_LOGD("AutoTiling", "DynamicBlockMxQuant AutoTiling Enter.");

    // 计算可用核数
    tilingParams.usedCoreNum =
        std::min(tilingParams.totalCoreNum, tilingParams.rowBlockLoopNum * tilingParams.colBlockLoopNum);
    tilingParams.usedCoreNum = tilingParams.usedCoreNum == 0 ? 1 : tilingParams.usedCoreNum;

    // 查找切分的组合
    std::set<int64_t> cutSet = FindUniqueCut(tilingParams.usedCoreNum);
    std::vector<std::vector<int64_t>> allTiling;

    // 行方向切分，枚举 m 的取值
    for (int64_t m : cutSet) {
        if (m > tilingParams.rowBlockLoopNum) {
            continue;
        }

        int64_t n = tilingParams.usedCoreNum / m;
        n = n < 1 ? 1 : n;
        if (n > tilingParams.colBlockLoopNum) {
            continue;
        }

        int64_t rowNormalBlock = Ops::Base::CeilDiv(tilingParams.rowBlockLoopNum, m);
        int64_t colNormalBlock = Ops::Base::CeilDiv(tilingParams.colBlockLoopNum, n);
        if (rowNormalBlock == 0 || colNormalBlock == 0) {
            continue;
        }

        int64_t delta = rowNormalBlock * colNormalBlock;
        if (m * n == static_cast<int64_t>(tilingParams.usedCoreNum)) {
            if (tilingParams.rowBlockLoopNum % m == 0 && tilingParams.colBlockLoopNum % n == 0) {
                tilingParams.rowTileNum = m;
                tilingParams.colTileNum = n;
                delta = 0;
            } else if (tilingParams.rowBlockLoopNum % m == 0) {
                delta = delta - rowNormalBlock * (tilingParams.colBlockLoopNum % colNormalBlock);
            } else if (tilingParams.colBlockLoopNum % n == 0) {
                delta = delta - (tilingParams.rowBlockLoopNum % rowNormalBlock) * n;
            } else {
                delta = delta - (tilingParams.rowBlockLoopNum % rowNormalBlock) *
                                    (tilingParams.colBlockLoopNum % colNormalBlock);
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

    tilingParams.rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    tilingParams.colTileNum = static_cast<uint16_t>(allTiling[0][1]);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockMxQuantTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicBlockMxQuantTiling DoTiling.");

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
        OP_LOGE(context_->GetNodeName(), "DynamicBlockMxQuantTiling SetTilingParams failed"), return ge::GRAPH_FAILED);

    SetTilingKey();
    SetTilingData();
    PrintTilingData();

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    uint64_t tilingDataSize = sizeof(tilingData);
    errno_t ret = memcpy_s(
        rawTilingData->GetData(), rawTilingData->GetCapacity(), reinterpret_cast<void*>(&tilingData), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Tiling DataSize Greater than capacity, please check.");
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams.workspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockMxQuantTiling::SetTilingParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicBlockMxQuantTiling SetTilingParams.");

    tilingParams.blockH = BASE_BLOCK_SIZE_ROW;
    tilingParams.blockW = BASE_BLOCK_SIZE_COL;

    CalcAxisSize();
    AutoTiling();

    tilingParams.usedCoreNum = tilingParams.rowTileNum * tilingParams.colTileNum;

    // 头核行方向基本块的数量
    tilingParams.normalCoreRowTileNum = Ops::Base::CeilDiv(tilingParams.rowBlockLoopNum, tilingParams.rowTileNum);
    // 头核列方向基本块的数量
    tilingParams.normalCoreColTileNum = Ops::Base::CeilDiv(tilingParams.colBlockLoopNum, tilingParams.colTileNum);

    // 尾核行方向基本块的数量
    tilingParams.tailCoreRowTileNum = Ops::Base::FloorDiv(tilingParams.rowBlockLoopNum, tilingParams.rowTileNum);
    // 尾核行方向基本块的数量
    tilingParams.tailCoreColTileNum = Ops::Base::FloorDiv(tilingParams.colBlockLoopNum, tilingParams.colTileNum);

    // rowTileNum：100，rowTileNum：9；normalCoreColTileNum：12，tailCoreRowTileNum：11
    tilingParams.rowNormalCoreNum =
        tilingParams.rowBlockLoopNum - tilingParams.rowTileNum * tilingParams.tailCoreRowTileNum;
    tilingParams.colNormalCoreNum =
        tilingParams.colBlockLoopNum - tilingParams.colTileNum * tilingParams.tailCoreColTileNum;

    tilingParams.rowNormalCoreNum =
        tilingParams.rowNormalCoreNum == 0 ? tilingParams.rowTileNum : tilingParams.rowNormalCoreNum;
    tilingParams.colNormalCoreNum =
        tilingParams.colNormalCoreNum == 0 ? tilingParams.colTileNum : tilingParams.colNormalCoreNum;

    tilingParams.rowTailCoreNum = tilingParams.rowTileNum - tilingParams.rowNormalCoreNum;
    tilingParams.colTailCoreNum = tilingParams.colTileNum - tilingParams.colNormalCoreNum;
    tilingParams.blockSizeRow = BLOCK_SIZE_ROW;
    tilingParams.blockSizeCol = BLOCK_SIZE_COL;
    tilingParams.rowScaleNum = Ops::Base::CeilDiv(tilingParams.rowNum, DIGIT_TWO * BLOCK_SIZE_ROW) * DIGIT_TWO;
    tilingParams.colScaleNum = Ops::Base::CeilDiv(tilingParams.colNum, DIGIT_TWO * BLOCK_SIZE_COL) * DIGIT_TWO;

    SplitCore();
    return ge::GRAPH_SUCCESS;
}

void DynamicBlockMxQuantTiling::SetTilingKey()
{
    if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_ROUND)) {
        roundMode_ = TPL_ROUND;
    } else if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_FLOOR)) {
        roundMode_ = TPL_FLOOR;
    } else if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_RINT)) {
        roundMode_ = TPL_RINT;
    }

    scaleAlg_ = tilingParams.scaleAlg;
    int64_t tilingKey = GET_TPL_TILING_KEY(roundMode_, scaleAlg_);
    OP_LOGD(context_->GetNodeName(), "roundMode is %ld, scaleAlg is %ld", roundMode_, scaleAlg_);
    context_->SetTilingKey(tilingKey);
}

void DynamicBlockMxQuantTiling::SetTilingData()
{
    tilingData.tilingKey = tilingParams.tilingKey;
    tilingData.totalCoreNum = tilingParams.totalCoreNum;
    tilingData.usedCoreNum = tilingParams.usedCoreNum;
    tilingData.ubSize = tilingParams.ubSize;
    tilingData.roundMode = tilingParams.roundMode;
    tilingData.dstType = tilingParams.dstType;
    tilingData.scaleAlg = tilingParams.scaleAlg;
    tilingData.dstTypeMax = tilingParams.dstTypeMax;
    tilingData.blockSizeRow = tilingParams.blockSizeRow;
    tilingData.blockSizeCol = tilingParams.blockSizeCol;
    tilingData.batchNum = tilingParams.batchNum;
    tilingData.rowNum = tilingParams.rowNum;
    tilingData.colNum = tilingParams.colNum;
    tilingData.singleBatchRowBlockLoopNum = tilingParams.singleBatchRowBlockLoopNum;
    tilingData.rowBlockLoopNum = tilingParams.rowBlockLoopNum;
    tilingData.colBlockLoopNum = tilingParams.colBlockLoopNum;
    tilingData.rowUbBlockLoopNum = tilingParams.rowUbBlockLoopNum;
    tilingData.colUbBlockLoopNum = tilingParams.colUbBlockLoopNum;
    tilingData.rowUbFactor = tilingParams.rowUbFactor;
    tilingData.colUbFactor = tilingParams.colUbFactor;
    tilingData.rowTileNum = tilingParams.rowTileNum;
    tilingData.colTileNum = tilingParams.colTileNum;
    tilingData.normalCoreRowTileNum = tilingParams.normalCoreRowTileNum;
    tilingData.normalCoreColTileNum = tilingParams.normalCoreColTileNum;
    tilingData.tailCoreRowTileNum = tilingParams.tailCoreRowTileNum;
    tilingData.tailCoreColTileNum = tilingParams.tailCoreColTileNum;
    tilingData.rowNormalCoreNum = tilingParams.rowNormalCoreNum;
    tilingData.colNormalCoreNum = tilingParams.colNormalCoreNum;
    tilingData.rowTailCoreNum = tilingParams.rowTailCoreNum;
    tilingData.colTailCoreNum = tilingParams.colTailCoreNum;
    tilingData.blockH = tilingParams.blockH;
    tilingData.blockW = tilingParams.blockW;
    tilingData.rowScaleNum = tilingParams.rowScaleNum;
    tilingData.colScaleNum = tilingParams.colScaleNum;
}

void DynamicBlockMxQuantTiling::PrintTilingData()
{
    OP_LOGD(
        context_->GetNodeName(),
        "TilingData tilingKey: %ld, totalCoreNum: %ld, usedCoreNum: %ld, ubSize: %ld, roundMode: %ld, "
        "dstType: %ld, scaleAlg: %ld, batchNum: %ld, rowNum: %ld, colNum: %ld, singleBatchRowBlockLoopNum: %ld, "
        "rowBlockLoopNum: %ld, colBlockLoopNum: %ld, rowUbBlockLoopNum: %ld, colUbBlockLoopNum: %ld, rowUbFactor: %ld, "
        "colUbFactor: %ld, rowTileNum: %ld, colTileNum: %ld, normalCoreRowTileNum: %ld, normalCoreColTileNum: %ld, "
        "tailCoreRowTileNum: %ld, tailCoreColTileNum: %ld, rowNormalCoreNum: %ld, colNormalCoreNum: %ld, "
        "rowTailCoreNum: %ld, colTailCoreNum: %ld, blockH: %ld, blockW: %ld, rowScaleNum: %ld, colScaleNum: %ld, "
        "dstTypeMax: %f ",
        tilingData.tilingKey, tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.ubSize, tilingData.roundMode,
        tilingData.dstType, tilingData.scaleAlg, tilingData.batchNum, tilingData.rowNum, tilingData.colNum,
        tilingData.singleBatchRowBlockLoopNum, tilingData.rowBlockLoopNum, tilingData.colBlockLoopNum,
        tilingData.rowUbBlockLoopNum, tilingData.colUbBlockLoopNum, tilingData.rowUbFactor, tilingData.colUbFactor,
        tilingData.rowTileNum, tilingData.colTileNum, tilingData.normalCoreRowTileNum, tilingData.normalCoreColTileNum,
        tilingData.tailCoreRowTileNum, tilingData.tailCoreColTileNum, tilingData.rowNormalCoreNum,
        tilingData.colNormalCoreNum, tilingData.rowTailCoreNum, tilingData.colTailCoreNum, tilingData.blockH,
        tilingData.blockW, tilingData.rowScaleNum, tilingData.colScaleNum, tilingData.dstTypeMax);
}

static ge::graphStatus TilingForDynamicBlockMxQuant(gert::TilingContext* context)
{
    OP_LOGD("DynamicBlockMxQuantTiling", "Enter TilingForDynamicBlockMxQuantTiling");

    OP_CHECK_IF(
        context == nullptr, OP_LOGE("DynamicBlockMxQuantTiling", "Tiling context is null."), return ge::GRAPH_FAILED);

    DynamicBlockMxQuantTiling blockMxQuantTiling(context);
    return blockMxQuantTiling.DoTiling();
}

static ge::graphStatus TilingPrepareForDynamicBlockMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD("DynamicBlockMxQuantTiling", "Enter TilingPrepareForDynamicBlockMxQuantTiling");

    OP_CHECK_IF(
        context == nullptr, OP_LOGE("DynamicBlockMxQuantTiling", "TilingParse context is null."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicBlockMxQuant)
    .Tiling(TilingForDynamicBlockMxQuant)
    .TilingParse<DynamicBlockMxQuantCompileInfo>(TilingPrepareForDynamicBlockMxQuant);

} // namespace optiling