/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deep_norm_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include "deep_norm_tiling.h"

namespace optiling {

constexpr uint32_t BLOCK_BASE_NUM = 16;
constexpr size_t MAX_DIM_X = 8;
constexpr size_t MIN_DIM_X = 2;
constexpr size_t MAX_DIM_GAMMA = 7;
constexpr size_t MIN_DIM_GAMMA = 1;

constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_GX_INDEX = 1;
constexpr int32_t INPUT_BETA_INDEX = 2;
constexpr int32_t INPUT_GAMMA_INDEX = 3;
constexpr int32_t OUTPUT_MEAN_INDEX = 0;
constexpr int32_t OUTPUT_RSTD_INDEX = 1;
constexpr int32_t OUTPUT_Y_INDEX = 2;
constexpr uint32_t TILING_ISSHORT_OFFSET = 16;
constexpr uint32_t TILING_UPPER_LIMIT_OFFSET = 8;
constexpr uint32_t TILING_BEYOND_LIMIT_OFFSET = 4;
constexpr uint32_t TILING_ISFP32_OFFSET = 2;
constexpr uint32_t TILING_ISFP16_OFFSET = 1;

static uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    return y == 0U ? x : (x + y - 1U) / y;
}

static uint32_t ROUND_UP(uint32_t x, uint32_t blockNumEl)
{
    return CEIL_DIV(x, blockNumEl) * blockNumEl;
}

static bool CheckInputOutputShapeDim(const gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape* gxShape = context->GetInputShape(INPUT_GX_INDEX);
    const gert::StorageShape* betaShape = context->GetInputShape(INPUT_BETA_INDEX);
    const gert::StorageShape* gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape* meanShape = context->GetOutputShape(OUTPUT_MEAN_INDEX);
    const gert::StorageShape* rstdShape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gxDimNum = gxShape->GetStorageShape().GetDimNum();
    size_t betaDimNum = betaShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t meanDimNum = meanShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();

    // Check shape dim range
    OP_CHECK_IF(
        (xDimNum > MAX_DIM_X) || (xDimNum < MIN_DIM_X),
        OP_LOGE(
            context, "Input x shape invaild, dim num should in range[%lu, %lu].", MIN_DIM_X, MAX_DIM_X),
        return false);
    OP_CHECK_IF(
        (gammaDimNum > MAX_DIM_GAMMA) || (gammaDimNum < MIN_DIM_GAMMA),
        OP_LOGE(
            context, "Input gamma shape invaild, dim num should in range[%lu, %lu].", MIN_DIM_GAMMA,
            MAX_DIM_GAMMA),
        return false);
    // Check shape dim relationship
    OP_CHECK_IF(
        gxDimNum != xDimNum, OP_LOGE(context, "Input gx shape invaild, dim num is not equal x dim."),
        return false);
    OP_CHECK_IF(
        (yDimNum != xDimNum) || (meanDimNum != xDimNum) || (rstdDimNum != xDimNum),
        OP_LOGE(context, "Output y/mean/rstd shape invaild, dim num is not equal x dim."), return false);
    OP_CHECK_IF(
        betaDimNum != gammaDimNum,
        OP_LOGE(context, "Input beta shape invaild, dim num is not equal gamma dim."), return false);
    OP_CHECK_IF(
        xDimNum <= gammaDimNum, OP_LOGE(context, "x dim num should not be smaller than gamma dim num."),
        return false);
    return true;
}

static bool CheckInputOutputShapeValue(const gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape* gxShape = context->GetInputShape(INPUT_GX_INDEX);
    const gert::StorageShape* betaShape = context->GetInputShape(INPUT_BETA_INDEX);
    const gert::StorageShape* gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape* meanShape = context->GetOutputShape(OUTPUT_MEAN_INDEX);
    const gert::StorageShape* rstdShape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();

    // Check shape value
    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_CHECK_IF(
            xShape->GetStorageShape().GetDim(i) == 0, OP_LOGE(context, "Input x shape can not be 0."),
            return false);
        OP_CHECK_IF(
            gxShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i),
            OP_LOGE(context, "Input gx shape invaild, shape is not equal x shape."), return false);
        OP_CHECK_IF(
            (yShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)),
            OP_LOGE(context, "Input y shape invaild, shape is not equal x shape."), return false);
    }
    for (uint32_t i = 0; i < xDimNum - gammaDimNum; i++) {
        OP_CHECK_IF(
            (rstdShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)) ||
                (meanShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)),
            OP_LOGE(context, "Output rstd/mean shape invaild, shape is not equal x first few dim."),
            return false);
    }
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_CHECK_IF(
            (gammaShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(xDimNum - gammaDimNum + i)) ||
                (betaShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(xDimNum - gammaDimNum + i)),
            OP_LOGE(context, "Input gamma shape invaild, gamma shape is not equal x last few dim."),
            return false);
    }
    return true;
}

static void SetTilingKey4DeepNorm(
    gert::TilingContext* context, uint32_t& numCol, uint32_t& shortLimit, uint32_t& limitLastDim,
    uint32_t& limitLastDim2, ge::DataType& dataType)
{
    uint32_t isShort = numCol <= shortLimit ? 1U : 0U;
    uint32_t upperLimit = limitLastDim2 < numCol ? 1U : 0U;
    uint32_t beyondLimit = limitLastDim < numCol ? 1U : 0U;
    uint32_t isFP32 = dataType == ge::DT_FLOAT ? 1U : 0U;
    uint32_t isFP16 = dataType == ge::DT_FLOAT16 ? 1U : 0U;
    //       D > 15360/8192     D > 4096      D <= 4096   D <= 100
    // fp32:    1110:14         0110:6         0010:2      10010:18
    // fp16:    1101:13         0101:5         0001:1      10001:17
    // bf16:    1100:12         0100:4         0000:0      10000:16
    uint32_t dtypeKey = isShort * TILING_ISSHORT_OFFSET + upperLimit * TILING_UPPER_LIMIT_OFFSET +
                        beyondLimit * TILING_BEYOND_LIMIT_OFFSET + isFP32 * TILING_ISFP32_OFFSET +
                        isFP16 * TILING_ISFP16_OFFSET;
    context->SetTilingKey(dtypeKey);
}

static ge::graphStatus Tiling4DeepNorm(gert::TilingContext* context)
{
    DeepNormTilingData tiling;
    OP_CHECK_IF(
        !CheckInputOutputShapeDim(context), OP_LOGE(context, "Input shape dim invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputOutputShapeValue(context), OP_LOGE(context, "Input shape value invalid."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t maxUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    // Get basic info
    const gert::Shape xShape = context->GetInputShape(0)->GetStorageShape();
    size_t gammaIndex = 3;
    const gert::Shape gammaShape = context->GetInputShape(gammaIndex)->GetStorageShape();
    uint32_t numCol = gammaShape.GetShapeSize();

    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    int32_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }

    auto dataType = context->GetInputDesc(0)->GetDataType();

    uint32_t bufferNum = 1;
    // buffer num = 1
    uint32_t limitLastDim = 4096;
    uint32_t fp16Limit = 15360;
    uint32_t fp32Limit = 8192;
    // when D > 500  Loop N is better than loop D
    uint32_t shortLimit = 500;
    uint32_t limitLastDim2 = dataType == ge::DT_FLOAT ? fp32Limit : fp16Limit;
    uint32_t doubleBuffer = 2;

    if (bufferNum == doubleBuffer) {
        // buffer num = 2
        uint32_t baseLimit = 4096;
        uint32_t fp16LimitDb = 13312;
        uint32_t fp32LimitDb = 3072;
        limitLastDim = dataType == ge::DT_FLOAT ? fp32LimitDb : baseLimit;
        limitLastDim2 = dataType == ge::DT_FLOAT ? fp32LimitDb : fp16LimitDb;
    }
    uint32_t numCore = CEIL_DIV(numRow, CEIL_DIV(numRow, maxCoreNum));
    uint32_t rowWork = CEIL_DIV(static_cast<uint32_t>(numRow), numCore);
    uint32_t lFirstdimPerCoreNum = static_cast<uint32_t>(numRow) - rowWork * (numCore - 1U);

    float tempAlpha = *context->GetAttrs()->GetFloat(0);
    float tempAve = numCol == 0U ? 1 : float(1.0 / numCol);
    float eps = *context->GetAttrs()->GetFloat(1);

    // About tiling
    uint32_t usedLastDim = numCol;
    if (limitLastDim < numCol) {
        uint32_t blockNum = CEIL_DIV(numCol, limitLastDim);
        usedLastDim = CEIL_DIV(numCol, blockNum * BLOCK_BASE_NUM) * BLOCK_BASE_NUM;
        tiling.set_updated_last_dim(usedLastDim);
        tiling.set_updated_last_times(blockNum);
    } else {
        tiling.set_updated_last_dim(0);
        tiling.set_updated_last_times(0);
    }

    int32_t byteNum = dataType == ge::DT_FLOAT ? 4 : 2;
    int32_t blockNum = 32 / byteNum;
    int32_t maxEleNum = static_cast<int32_t>(maxUbSize / static_cast<uint64_t>(byteNum));
    int32_t dynDataUsed = 2;                                    // tensor from gm
    int32_t staticDataUsed = 3;                                 // local tensor
    int32_t queDataUsed = static_cast<int32_t>(2U * bufferNum); // local tensor
    uint32_t dynLastDimUint32 = limitLastDim2 < numCol ? usedLastDim : numCol;
    int32_t dynLastDim = static_cast<int32_t>(dynLastDimUint32);
    int32_t scalarUsed = 50;
    int32_t numTempBuf = 64;
    int32_t isSmallType = dataType != ge::DT_FLOAT && limitLastDim >= numCol ? 1 : 0;
    int32_t isShortCase = numCol <= shortLimit ? 1 : 0;
    int32_t totalMemNeed = static_cast<int32_t>(
        (static_cast<uint32_t>(dynDataUsed) * ROUND_UP(usedLastDim, static_cast<uint32_t>(blockNum)) +
         ROUND_UP(static_cast<uint32_t>(dynLastDim), static_cast<uint32_t>(blockNum)) +
         64U / static_cast<uint32_t>(byteNum)) *
            rowWork * bufferNum +
        static_cast<uint32_t>(isSmallType) * static_cast<uint32_t>(1 - isShortCase) * 4U *
            ROUND_UP(usedLastDim, static_cast<uint32_t>(blockNum)) * rowWork +
        static_cast<uint32_t>(isShortCase) * static_cast<uint32_t>(isSmallType) *
            static_cast<uint32_t>(staticDataUsed) * ROUND_UP(usedLastDim, static_cast<uint32_t>(blockNum)) * 4U /
            static_cast<uint32_t>(byteNum) * rowWork);
    int32_t sumData = static_cast<int32_t>(
        static_cast<uint32_t>(maxEleNum) - static_cast<uint32_t>(numTempBuf) - static_cast<uint32_t>(scalarUsed) -
        static_cast<uint32_t>(queDataUsed) * ROUND_UP(usedLastDim, static_cast<uint32_t>(blockNum)) -
        static_cast<uint32_t>(1 - isShortCase) * static_cast<uint32_t>(staticDataUsed) *
            ROUND_UP(usedLastDim, static_cast<uint32_t>(blockNum)) * 4U / static_cast<uint32_t>(byteNum) -
        ROUND_UP(static_cast<uint32_t>(dynLastDim), static_cast<uint32_t>(blockNum)) * 4U /
            static_cast<uint32_t>(byteNum));
    uint32_t firstDimPerTime = 0;
    if (limitLastDim < numCol) {
        firstDimPerTime = 1U;
    } else if (totalMemNeed > sumData) {
        uint32_t timeCopyIn = CEIL_DIV(static_cast<uint32_t>(totalMemNeed), static_cast<uint32_t>(sumData));
        if (timeCopyIn > 0U) {
            firstDimPerTime = rowWork / timeCopyIn;
        }
    } else {
        firstDimPerTime = rowWork;
    }
    uint32_t maxRepeat = 255;
    if (numCol <= shortLimit) {
        firstDimPerTime = firstDimPerTime > maxRepeat ? maxRepeat : firstDimPerTime;
    }
    tiling.set_num_core(numCore);
    tiling.set_num_last_dim(numCol);
    tiling.set_num_first_dim(numRow);
    tiling.set_nl_firstdim_per_core(rowWork);
    tiling.set_l_firstdim_per_core(lFirstdimPerCoreNum);
    tiling.set_first_dim_per_times(firstDimPerTime);
    tiling.set_eps_str(*reinterpret_cast<uint32_t*>(&eps));
    tiling.set_ave_str(*reinterpret_cast<uint32_t*>(&tempAve));
    tiling.set_alpha_str(*reinterpret_cast<uint32_t*>(&tempAlpha));
    context->SetBlockDim(numCore);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    SetTilingKey4DeepNorm(context, numCol, shortLimit, limitLastDim, limitLastDim2, dataType);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DeepNorm(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepare4DeepNorm");
    return ge::GRAPH_SUCCESS;
}

struct DeepNormCompileInfo {
};
IMPL_OP_OPTILING(DeepNorm).Tiling(Tiling4DeepNorm).TilingParse<DeepNormCompileInfo>(TilingPrepare4DeepNorm);
} // namespace optiling
