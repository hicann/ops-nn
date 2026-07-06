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
 * \file fused_add_rms_norm_tiling.cpp
 * \brief
 */
#include "fused_add_rms_norm_tiling.h"

namespace optiling {
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t UB_FACTOR_B16 = 12288;
constexpr uint32_t UB_FACTOR_B32 = 10240;
constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t MODE_NORMAL = 0;
constexpr int32_t INPUT_X1_INDEX = 0;
constexpr int32_t INPUT_X2_INDEX = 1;
constexpr int32_t INPUT_GAMMA_INDEX = 2;
constexpr int32_t OUTPUT_Y_INDEX = 0;
constexpr int32_t OUTPUT_RSTD_INDEX = 1;
constexpr int32_t OUTPUT_X_INDEX = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;

struct FusedAddRmsNormShapeInfo {
    const gert::StorageShape* x1 = nullptr;
    const gert::StorageShape* x2 = nullptr;
    const gert::StorageShape* gamma = nullptr;
    const gert::StorageShape* y = nullptr;
    const gert::StorageShape* rstd = nullptr;
    const gert::StorageShape* x = nullptr;
};

static bool GetShapeInfo(const gert::TilingContext* context, FusedAddRmsNormShapeInfo& shapeInfo)
{
    shapeInfo.x1 = context->GetInputShape(INPUT_X1_INDEX);
    shapeInfo.x2 = context->GetInputShape(INPUT_X2_INDEX);
    shapeInfo.gamma = context->GetInputShape(INPUT_GAMMA_INDEX);
    shapeInfo.y = context->GetOutputShape(OUTPUT_Y_INDEX);
    shapeInfo.rstd = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    shapeInfo.x = context->GetOutputShape(OUTPUT_X_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInfo.x1);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInfo.x2);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInfo.gamma);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInfo.y);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInfo.rstd);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInfo.x);
    return true;
}

static uint32_t GetDataPerBlock(ge::DataType dataType)
{
    return (dataType == ge::DT_FLOAT) ? FLOAT_BLOCK_ALIGN_NUM : BLOCK_ALIGN_NUM;
}

static bool CheckInputOutputDim(const gert::TilingContext* context)
{
    FusedAddRmsNormShapeInfo shapeInfo;
    OP_CHECK_IF(!GetShapeInfo(context, shapeInfo), OP_LOGE(context, "Get shape info failed."), return false);

    size_t x1DimNum = shapeInfo.x1->GetStorageShape().GetDimNum();
    size_t x2DimNum = shapeInfo.x2->GetStorageShape().GetDimNum();
    size_t gammaDimNum = shapeInfo.gamma->GetStorageShape().GetDimNum();
    size_t yDimNum = shapeInfo.y->GetStorageShape().GetDimNum();
    size_t rstdDimNum = shapeInfo.rstd->GetStorageShape().GetDimNum();
    size_t xDimNum = shapeInfo.x->GetStorageShape().GetDimNum();

    OP_CHECK_IF(
        x1DimNum > MAX_DIM_NUM || x1DimNum < MIN_DIM_X,
        OP_LOGE(context, "Input x1's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_CHECK_IF(
        gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE(context, "Input gamma's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_CHECK_IF(
        x1DimNum != yDimNum, OP_LOGE(context, "Input x's dim num must equal to output y's dim num."),
        return false);

    OP_CHECK_IF(
        x1DimNum != x2DimNum,
        OP_LOGE(context, "Input x2/x1 shape invaild, dim num is not equal x1 dim."), return false);
    OP_CHECK_IF(
        (yDimNum != xDimNum) || (xDimNum != x1DimNum) || (rstdDimNum != x1DimNum),
        OP_LOGE(context, "Output y/x/rstd shape invaild, dim num is not equal x1 dim."), return false);
    OP_CHECK_IF(
        x1DimNum < gammaDimNum, OP_LOGE(context, "X1 dim num should not be smaller than gamma dim num."),
        return false);
    return true;
}

static bool CheckInputOutputShape(const gert::TilingContext* context)
{
    OP_CHECK_IF(!CheckInputOutputDim(context), OP_LOGE(context, "Input Dim invalid."), return false);
    FusedAddRmsNormShapeInfo shapeInfo;
    OP_CHECK_IF(!GetShapeInfo(context, shapeInfo), OP_LOGE(context, "Get shape info failed."), return false);

    size_t x1DimNum = shapeInfo.x1->GetStorageShape().GetDimNum();
    size_t gammaDimNum = shapeInfo.gamma->GetStorageShape().GetDimNum();

    for (uint32_t i = 0; i < x1DimNum; i++) {
        OP_CHECK_IF(
            shapeInfo.x1->GetStorageShape().GetDim(i) == 0, OP_LOGE(context, "Input x1 shape can not be 0."),
            return false);
        OP_CHECK_IF(
            shapeInfo.x2->GetStorageShape().GetDim(i) != shapeInfo.x1->GetStorageShape().GetDim(i),
            OP_LOGE(context, "Input x2/x1 shape invaild, shape is not equal x1 shape."), return false);
        OP_CHECK_IF(
            (shapeInfo.y->GetStorageShape().GetDim(i) != shapeInfo.x1->GetStorageShape().GetDim(i)) ||
                (shapeInfo.x->GetStorageShape().GetDim(i) != shapeInfo.x1->GetStorageShape().GetDim(i)),
            OP_LOGE(context, "Input y/x shape invaild, shape is not equal x1 shape."), return false);
    }
    for (uint32_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        OP_CHECK_IF(
            shapeInfo.rstd->GetStorageShape().GetDim(i) != shapeInfo.x2->GetStorageShape().GetDim(i),
            OP_LOGE(context, "Output rstd shape invaild, shape is not equal x1 first few dim."),
            return false);
    }
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_CHECK_IF(
            shapeInfo.gamma->GetStorageShape().GetDim(i) !=
                shapeInfo.x1->GetStorageShape().GetDim(x1DimNum - gammaDimNum + i),
            OP_LOGE(context, "Input gamma shape invaild, gamma shape is not equal x1 last few dim."),
            return false);
        OP_CHECK_IF(
            shapeInfo.rstd->GetStorageShape().GetDim(x1DimNum - 1 - i) != 1,
            OP_LOGE(context, "Output rstd shape invaild, last few dim is not equal to 1."),
            return false);
    }
    return true;
}

static void GetCompileParameters(
    gert::TilingContext* context, uint32_t& numCore, uint64_t& ubSize)
{
    auto ptrCompileInfo = reinterpret_cast<const FusedAddRmsNormCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
    }
    ubSize -= UB_USED;
}

static void CalculateRowAndColParameters(gert::TilingContext* context, uint32_t& numRow, uint32_t& numCol)
{
    const gert::Shape x1_shape = context->GetInputShape(0)->GetStorageShape();
    const size_t gammaIndex = 2;
    const gert::Shape gamma_shape = context->GetInputShape(gammaIndex)->GetStorageShape();
    numCol = gamma_shape.GetShapeSize();

    const size_t x1DimNum = x1_shape.GetDimNum();
    const size_t gammaDimNum = gamma_shape.GetDimNum();
    numRow = 1U;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; ++i) {
        numRow *= x1_shape.GetDim(i);
    }
}

static ge::graphStatus GetEpsilonParameter(gert::TilingContext* context, float& epsilon)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    epsilon = *attrs->GetFloat(0);
    OP_CHECK_IF(
        epsilon < 0, OP_LOGE(context, "Epsilon less than zero, please check."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetScaleParameter(gert::TilingContext* context, float& scale)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scale_ptr = attrs->GetFloat(1);
    if (scale_ptr != nullptr) {
        scale = *scale_ptr;
    }
    return ge::GRAPH_SUCCESS;
}

static void CalculateBlockParameters(
    uint32_t numRow, uint32_t numCore, uint32_t& blockFactor, uint32_t& latsBlockFactor, uint32_t& useCoreNum)
{
    blockFactor = 1U;
    uint32_t tileNum = Ops::Base::CeilDiv(numRow, numCore * blockFactor);
    blockFactor *= tileNum;
    useCoreNum = Ops::Base::CeilDiv(numRow, blockFactor);
    latsBlockFactor = numRow - blockFactor * (useCoreNum - 1);
}

static ge::DataType SetDataTypeParameters(gert::TilingContext* context, uint32_t& dataPerBlock)
{
    auto dataType = context->GetInputDesc(0)->GetDataType();
    dataPerBlock = GetDataPerBlock(dataType);
    return dataType;
}

static void SetTilingParameters(
    FusedAddRMSNormTilingData* tiling, uint32_t num_row, uint32_t num_col, uint32_t numColAlign,
    uint32_t block_factor, uint32_t latsBlockFactor, uint32_t row_factor,
    uint32_t ub_factor, float epsilon, float scale)
{
    const float avg_factor = (num_col == 0) ? 0 : 1.0f / num_col;
    tiling->set_num_row(num_row);
    tiling->set_num_col(num_col);
    tiling->set_num_col_align(numColAlign);
    tiling->set_block_factor(block_factor);
    tiling->set_last_block_factor(latsBlockFactor);
    tiling->set_row_factor(row_factor);
    tiling->set_ub_factor(ub_factor);
    tiling->set_epsilon(epsilon);
    tiling->set_scale(scale);
    tiling->set_avg_factor(avg_factor);
}

static void SaveTilingData(
    gert::TilingContext* context, FusedAddRMSNormTilingData* tiling, uint32_t modeKey)
{
    context->SetTilingKey(modeKey);
    tiling->SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
}

static void SetWorkspaceSize(gert::TilingContext* context)
{
    constexpr size_t sysWorkspaceSize = 16 * 1024 * 1024;
    constexpr size_t usrSize = 256;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
}

static void LogTilingResults(
    gert::TilingContext* context, FusedAddRMSNormTilingData* tiling, uint32_t modeKey, uint32_t useCoreNum,
    float epsilon, float scale)
{
    OP_LOGI(context, "Tiling Key: %u", modeKey);
    OP_LOGI(context, "Block Dim: %u", useCoreNum);
    OP_LOGI(context, "usr Workspace: 256");
    OP_LOGI(
        context,
        "num_row: %d, num_col: %d, block_factor: %d, row_factor: %d, ub_factor: %d, "
        "epsilon: %f, scale: %f, avg_factor: %f",
        tiling->get_num_row(), tiling->get_num_col(), tiling->get_block_factor(), tiling->get_row_factor(),
        tiling->get_ub_factor(), epsilon, scale, tiling->get_avg_factor());
}

static ge::graphStatus Tiling4FusedAddRmsNorm(gert::TilingContext* context)
{
    OP_LOGI("Tiling4FusedAddRmsNorm", "Enter Tiling4FusedAddRmsNorm");
    OP_CHECK_IF(
        !CheckInputOutputShape(context), OP_LOGE(context, "Input shape invalid."),
        return ge::GRAPH_FAILED);

    FusedAddRMSNormTilingData tiling;
    uint32_t num_core;
    uint64_t ub_size;

    GetCompileParameters(context, num_core, ub_size);

    uint32_t num_row;
    uint32_t num_col;
    CalculateRowAndColParameters(context, num_row, num_col);

    float epsilon = 0;
    GetEpsilonParameter(context, epsilon);
    if (epsilon < 0) {
        return ge::GRAPH_FAILED;
    }

    float scale = 1.0f;
    OP_CHECK_IF(GetScaleParameter(context, scale) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Get scale parameter failed."), return ge::GRAPH_FAILED);

    uint32_t block_factor;
    uint32_t latsBlockFactor;
    uint32_t use_core_num;
    CalculateBlockParameters(num_row, num_core, block_factor, latsBlockFactor, use_core_num);
    context->SetBlockDim(use_core_num);

    uint32_t data_per_block;
    ge::DataType data_type = SetDataTypeParameters(context, data_per_block);

    uint32_t mode_key = MODE_NORMAL;
    uint32_t row_factor = 64;
    uint32_t ub_factor = (data_type == ge::DT_FLOAT) ? UB_FACTOR_B32 : UB_FACTOR_B16;
    uint32_t numColAlign = Ops::Base::CeilDiv(num_col, data_per_block) * data_per_block;
    OP_CHECK_IF(num_col > ub_factor,
        OP_LOGE(context, "MiniCPM scaled add rms norm first version requires num_col <= ub_factor."),
        return ge::GRAPH_FAILED);

    SetTilingParameters(
        &tiling, num_row, num_col, numColAlign, block_factor, latsBlockFactor, row_factor, ub_factor, epsilon, scale);
    SaveTilingData(context, &tiling, mode_key);

    SetWorkspaceSize(context);

    LogTilingResults(context, &tiling, mode_key, use_core_num, epsilon, scale);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4FusedAddRmsNorm(gert::TilingParseContext* context)
{
    OP_LOGI(context, "TilingPrepare4FusedAddRmsNorm running.");
    auto compileInfo = context->GetCompiledInfo<FusedAddRmsNormCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedAddRmsNorm)
    .Tiling(Tiling4FusedAddRmsNorm)
    .TilingParse<FusedAddRmsNormCompileInfo>(TilingPrepare4FusedAddRmsNorm);

}  // namespace optiling
