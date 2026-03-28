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
 * \file dynamic_mx_quant_tail_axis_tiling_arch35.cpp
 * \brief This tiling template only supports scenarios where the quantization axis is the last axis and the blockSize is 32.
 */

#include "dynamic_mx_quant_tiling_arch35.h"
#include "../op_kernel/arch35/dynamic_mx_quant_tilingdata.h"
#include <cmath>
#include "platform/platform_info.h"

namespace optiling {
using namespace ge;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t DIGIT_TWOHUNDREDFIFTYSIX = 256;
constexpr int64_t DIGIT_ONEHUNDREDTWENTYEIGHT = 128;
constexpr int64_t DIGIT_EIGHT = 8;

constexpr int64_t BYTES_OF_INPUT_UINT16_DB_BUFFER_TYPE = 4;
constexpr int64_t BYTES_OF_OUTPUT_FP8_DB_BUFFER_TYPE = 2;
constexpr int64_t BYTES_OF_OUTPUT_FP4_DB_BUFFER_TYPE = 1;
constexpr int64_t BYTES_OF_MAX_VALUE_TYPE = 2;
constexpr int64_t BYTES_OF_SCALE_TYPE = 1;
constexpr int64_t BYTES_OF_INVERSE_SCALE_TYPE = 2;
constexpr int64_t DB_BUFFER = 2;
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};

ge::graphStatus DynamicMxQuantTailAxisTiling::TailAxisSetTilingData(gert::TilingContext* context, DynamicMxQuantTailAxisTilingData& tilingData)
{
    uint64_t tilingDataSize = sizeof(tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    auto rawTilingData = context->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
        reinterpret_cast<void*>(&tilingData), tilingDataSize);
        if (ret != EOK) {
            OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret = %d", ret);
            return ge::GRAPH_FAILED;
        }
        context->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

void DynamicMxQuantTailAxisTiling::SetTilingDataForTailAxis(DynamicMxQuantTailAxisTilingData& tilingData, const DynamicMxQuantTilingParam& tilingParam)
{
    tilingData.tilingKey = tilingParam.tilingKey;
    tilingData.ubSize = tilingParam.ubSize;
    tilingData.roundMode = tilingParam.roundMode;
    tilingData.blockSize = tilingParam.blockSize;
    tilingData.totalCoreNum = tilingParam.totalCoreNum;
    tilingData.usedCoreNum = tilingParam.usedCoreNum;
    tilingData.rowTileNum = tilingParam.rowTileNum;
    tilingData.colTileNum = tilingParam.colTileNum;
    tilingData.rowNum = tilingParam.rowNum;
    tilingData.colNum = tilingParam.colNum;
    tilingData.colNormalBlockNum = tilingParam.colNormalBlockNum;
    tilingData.colTailLen = tilingParam.colTailLen;
    tilingData.rowNormalBlockNum = tilingParam.rowNormalBlockNum;
    tilingData.rowTailLen = tilingParam.rowTailLen;
    tilingData.maxUbBlockNum = tilingParam.maxUbBlockNum;
    tilingData.dstTypeMax = tilingParam.dstTypeMax;
    tilingData.invDstTypeMax = tilingParam.invDstTypeMax;
}

void DynamicMxQuantTailAxisTiling::PrintTilingDataForTailAxis(gert::TilingContext* context, const DynamicMxQuantTailAxisTilingData& tilingData)
{
    OP_LOGI(
    context->GetNodeName(),
        "tilingData is tilingKey:%ld, ubSize:%ld, roundMode:%ld, blockSize:%ld, \
        totalCoreNum:%ld, usedCoreNum:%ld, rowTileNum:%ld, colTileNum:%ld, \
        rowNum:%ld, colNum:%ld, colNormalBlockNum:%ld, colTailLen:%ld, \
        rowNormalBlockNum:%ld, rowTailLen:%ld, maxUbBlockNum:%ld, dstTypeMax:%f, invDstTypeMax:%f.",
        tilingData.tilingKey, tilingData.ubSize, tilingData.roundMode, tilingData.blockSize,
        tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.rowTileNum, tilingData.colTileNum,
        tilingData.rowNum, tilingData.colNum, tilingData.colNormalBlockNum, tilingData.colTailLen, 
        tilingData.rowNormalBlockNum, tilingData.rowTailLen, tilingData.maxUbBlockNum, tilingData.dstTypeMax, tilingData.invDstTypeMax);
}

std::set<int64_t> DynamicMxQuantTailAxisTiling::FindSplitCombo(int64_t usedCoreNum)
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

void DynamicMxQuantTailAxisTiling::AutoTiling(DynamicMxQuantTilingParam& tilingParam)
{
    // 计算可用核数
    tilingParam.usedCoreNum =
        std::min(tilingParam.totalCoreNum, tilingParam.rowBlockLoopNum * tilingParam.colBlockLoopNum);
    tilingParam.usedCoreNum = tilingParam.usedCoreNum == 0 ? 1 : tilingParam.usedCoreNum;

    // 查找切分的组合
    std::set<int64_t> cutSet = FindSplitCombo(tilingParam.usedCoreNum);
    std::vector<std::vector<int64_t>> allTiling;

    // 行方向切分，枚举m的取值
    for (int64_t m : cutSet) {
        if (m > tilingParam.rowBlockLoopNum) {
            continue;
        }
        int64_t n = tilingParam.usedCoreNum / m;
        n = n < 1 ? 1 : n;
        if (n > tilingParam.colBlockLoopNum) {
            continue;
        }

        int64_t rowNormalBlockNum = Ops::Base::CeilDiv(tilingParam.rowBlockLoopNum, m);
        int64_t colNormalBlockNum = Ops::Base::CeilDiv(tilingParam.colBlockLoopNum, n);
        int64_t delta = rowNormalBlockNum * colNormalBlockNum;
        if (m * n == static_cast<int64_t>(tilingParam.usedCoreNum)) {
            if (tilingParam.rowBlockLoopNum % m == 0 && tilingParam.colBlockLoopNum % n == 0) {
                delta = 0;
            } else if (tilingParam.rowBlockLoopNum % m == 0) {
                delta = delta - rowNormalBlockNum * (tilingParam.colBlockLoopNum % colNormalBlockNum);
            } else if (tilingParam.colBlockLoopNum % n == 0) {
                delta = delta - (tilingParam.rowBlockLoopNum % rowNormalBlockNum) * colNormalBlockNum;
            } else {
                delta = delta - (tilingParam.rowBlockLoopNum % rowNormalBlockNum) *
                    (tilingParam.colBlockLoopNum % colNormalBlockNum);
            }
        }

        allTiling.push_back({m, n, m * n, delta});
    }

    // 排序以选择最合适的切分
    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int FirstIndex = 1;   // 优先切 M
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[DeltaIndex], a[FirstIndex]) < std::make_pair(b[DeltaIndex], b[FirstIndex]);
    });

    tilingParam.rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    tilingParam.colTileNum = static_cast<uint16_t>(allTiling[0][1]);
}

void DynamicMxQuantTailAxisTiling::CalcTilingKeyForTail(DynamicMxQuantTilingParam& tilingParam, const ge::DataType& yDtype)
{
    int64_t tenDigit = 0;
    int64_t digit = 0;
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0) {
        tenDigit = DIGIT_ONE;       // 量化输出FP4 十位为1
    } else {
        tenDigit = DIGIT_TWO;       // 量化输出FP8 十位为2
    }
    digit = tilingParam.scaleAlg;   // 个位为SCALE_ALG
    tilingParam.tilingKey = tenDigit * DIGIT_TEN + digit * DIGIT_ONE;
}

void DynamicMxQuantTailAxisTiling::CalcAxisSize(DynamicMxQuantTilingParam& tilingParam, const gert::Shape& xShape)
{
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        tilingParam.rowNum *= xShape.GetDim(i);
    }
    tilingParam.colNum = xShape.GetDim(xShape.GetDimNum() - 1);

    tilingParam.rowBlockLoopNum = Ops::Base::CeilDiv(tilingParam.rowNum, DIGIT_ONE);
    tilingParam.colBlockLoopNum = Ops::Base::CeilDiv(tilingParam.colNum, DIGIT_TWOHUNDREDFIFTYSIX);
}

ge::graphStatus DynamicMxQuantTailAxisTiling::DoTiling() {
    OP_LOGD(context->GetNodeName(), "DynamicMxQuantTailAxisTiling DoTiling entering.");

    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    CalcAxisSize(tilingParam, xShape);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    AutoTiling(tilingParam);
    tilingParam.rowNormalBlockNum = Ops::Base::CeilDiv(tilingParam.rowBlockLoopNum, tilingParam.rowTileNum);
    tilingParam.colNormalBlockNum = Ops::Base::CeilDiv(tilingParam.colBlockLoopNum, tilingParam.colTileNum);
    tilingParam.rowTileNum = Ops::Base::CeilDiv(tilingParam.rowBlockLoopNum, tilingParam.rowNormalBlockNum);
    tilingParam.colTileNum = Ops::Base::CeilDiv(tilingParam.colBlockLoopNum, tilingParam.colNormalBlockNum);
    tilingParam.usedCoreNum = tilingParam.rowTileNum * tilingParam.colTileNum;
    tilingParam.rowTailLen = tilingParam.rowNum - (tilingParam.rowNormalBlockNum * DIGIT_ONE *
        (tilingParam.rowTileNum - DIGIT_ONE));
    tilingParam.colTailLen = tilingParam.colNum - (tilingParam.colNormalBlockNum * DIGIT_TWOHUNDREDFIFTYSIX *
        (tilingParam.colTileNum - DIGIT_ONE));

    // 计算UB可以放下的block（1×256）数量
    // maxUbBlockNum * DIGIT_TWOHUNDREDFIFTYSIX * (BYTES_OF_INPUT_TYPE * DB_BUFFER + BYTES_OF_OUTPUT_TYPE * DB_BUFFER) +
    // maxUbBlockNum * DIGIT_EIGHT * (BYTES_OF_MAX_VALUE_TYPE + BYTES_OF_SCALE_TYPE * DB_BUFFER + BYTES_OF_INVERSE_SCALE_TYPE)
    // <= ubSize
    if (tilingParam.dstType == 40 || tilingParam.dstType == 41) {   // Y FP4
        tilingParam.maxUbBlockNum = tilingParam.ubSize / (DIGIT_TWOHUNDREDFIFTYSIX * (BYTES_OF_INPUT_UINT16_DB_BUFFER_TYPE + BYTES_OF_OUTPUT_FP4_DB_BUFFER_TYPE) 
            + DIGIT_EIGHT * (BYTES_OF_MAX_VALUE_TYPE + BYTES_OF_SCALE_TYPE * DB_BUFFER + BYTES_OF_INVERSE_SCALE_TYPE));
    } else if (tilingParam.dstType == 35 || tilingParam.dstType == 36) {    // Y FP8
        tilingParam.maxUbBlockNum = tilingParam.ubSize / (DIGIT_TWOHUNDREDFIFTYSIX * (BYTES_OF_INPUT_UINT16_DB_BUFFER_TYPE + BYTES_OF_OUTPUT_FP8_DB_BUFFER_TYPE) 
            + DIGIT_EIGHT * (BYTES_OF_MAX_VALUE_TYPE + BYTES_OF_SCALE_TYPE * DB_BUFFER + BYTES_OF_INVERSE_SCALE_TYPE));
    }
    tilingParam.maxUbBlockNum *= DIGIT_EIGHT;       // 转换成UB可以放下的block（1×32）数量

    CalcTilingKeyForTail(tilingParam, yDtype);
    DynamicMxQuantTailAxisTilingData tilingData;
    SetTilingDataForTailAxis(tilingData, tilingParam);
    OP_CHECK_IF(
        TailAxisSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "DynamicMxQuantTailAxisSetTilingData set tiling data failed."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.usedCoreNum);
    context->SetTilingKey(tilingData.tilingKey);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = tilingParam.workspaceSize;

    PrintTilingDataForTailAxis(context, tilingData);
    return ge::GRAPH_SUCCESS;
}
}