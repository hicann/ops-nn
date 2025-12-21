/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file modulate_tiling.cc
 * \brief
 */

#include "modulate_tiling.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t MIN_TILE_SIZE = 32 * 1024;
constexpr int64_t ALIGNED_SIZE = 64;
constexpr int64_t SCALE_AND_SHIFT = 0;
constexpr int64_t NO_SCALE = 1;
constexpr int64_t NO_SHIFT = 2;
constexpr int64_t INPUT_X = 0;
constexpr int64_t INPUT_SCALE = 1;
constexpr int64_t INPUT_SHIFT = 2;
constexpr int64_t DIM_B = 0;
constexpr int64_t DIM_L = 1;
constexpr int64_t DIM_D = 2;
constexpr int64_t TILING_DIM_B = 0;
constexpr int64_t TILING_DIM_L = 1;
constexpr int64_t TILING_DIM_D = 2;
constexpr int64_t UB_TENSOR_NUM = 5;
constexpr int64_t UB_TENSOR_NUM_ALL = 6;
constexpr int64_t HALF_CORE_NUM = 2;

struct TilingDataStructModulate {
    int64_t inputB = 0;
    int64_t inputL = 0;
    int64_t inputD = 0;
    int64_t ubLength = 0;
    int64_t frontNum = 0;
    int64_t frontLength = 0;
    int64_t tailNum = 0;
    int64_t tailLength = 0;
    int64_t useDTiling = 0;
    int64_t parameterStatus = 0;
};

enum class TilingStrategy
{
    TilingB,
    TilingL,
    TilingD
};

class ModulateTilingOp {
public:
    explicit ModulateTilingOp(gert::TilingContext* context_) : context(context_){};
    void Init();

private:
    TilingStrategy SelectStrategy();
    void CalcTiling();
    void CalcTilingParam(int64_t TilingDim, int64_t totalElements, bool useDtiling = false);
    void SetTiling();
    void PrintTilingData();
    gert::TilingContext* context = nullptr;
    TilingDataStructModulate tilingData;
    int64_t dataTypeSize = 0;
    int64_t coreNum = 0;
};

void ModulateTilingOp::PrintTilingData()
{
    OP_LOGD(context, "inputB: %ld.", tilingData.inputB);
    OP_LOGD(context, "inputL: %ld.", tilingData.inputL);
    OP_LOGD(context, "inputD: %ld.", tilingData.inputD);
    OP_LOGD(context, "ubLength: %ld.", tilingData.ubLength);
    OP_LOGD(context, "frontNum: %ld.", tilingData.frontNum);
    OP_LOGD(context, "frontLength: %ld.", tilingData.frontLength);
    OP_LOGD(context, "tailNum: %ld.", tilingData.tailNum);
    OP_LOGD(context, "tailLength: %ld.", tilingData.tailLength);
    OP_LOGD(context, "useDTiling: %ld.", tilingData.useDTiling);
    OP_LOGD(context, "parameterStatus: %ld.", tilingData.parameterStatus);
}

void ModulateTilingOp::SetTiling()
{
    ModulateTilingData tiling;
    tiling.set_inputB(tilingData.inputB);
    tiling.set_inputL(tilingData.inputL);
    tiling.set_inputD(tilingData.inputD);
    tiling.set_ubLength(tilingData.ubLength);
    tiling.set_frontNum(tilingData.frontNum);
    tiling.set_frontLength(tilingData.frontLength);
    tiling.set_tailNum(tilingData.tailNum);
    tiling.set_tailLength(tilingData.tailLength);
    tiling.set_useDTiling(tilingData.useDTiling);
    tiling.set_parameterStatus(tilingData.parameterStatus);
    context->SetBlockDim(tilingData.frontNum + tilingData.tailNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
}

void ModulateTilingOp::Init()
{
    const gert::StorageShape* xShape = context->GetInputShape(INPUT_X);
    const gert::StorageShape* scaleShape = context->GetOptionalInputShape(INPUT_SCALE);
    const gert::StorageShape* shiftShape = context->GetOptionalInputShape(INPUT_SHIFT);
    const ge::DataType dataType = context->GetInputDesc(0)->GetDataType();

    switch (dataType) {
        case ge::DT_FLOAT:
            this->dataTypeSize = 4; // 4 bytes for DT_FLOAT
            break;
        case ge::DT_FLOAT16:
            this->dataTypeSize = 2; // 2 bytes for DT_FLOAT16
            break;
        case ge::DT_BF16:
            this->dataTypeSize = 2; // 2 bytes for DT_BF16
            break;
        default:
            break;
    }

    if (scaleShape && shiftShape) {
        this->tilingData.parameterStatus = SCALE_AND_SHIFT;
    } else if (!scaleShape) {
        this->tilingData.parameterStatus = NO_SCALE;
    } else if (!shiftShape) {
        this->tilingData.parameterStatus = NO_SHIFT;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    this->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "ascendcPlatform CoreNum: %ld.", this->coreNum);

    int64_t inputB = xShape->GetStorageShape().GetDim(DIM_B);
    int64_t inputL = xShape->GetStorageShape().GetDim(DIM_L);
    int64_t inputD = xShape->GetStorageShape().GetDim(DIM_D);
    this->tilingData.inputB = inputB;
    this->tilingData.inputL = inputL;
    this->tilingData.inputD = inputD;

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    int64_t ubTensorNum = (this->tilingData.parameterStatus == SCALE_AND_SHIFT) ? UB_TENSOR_NUM_ALL : UB_TENSOR_NUM;
    this->tilingData.ubLength =
        static_cast<int64_t>(ubSizePlatForm) / ubTensorNum / this->dataTypeSize / ALIGNED_SIZE * ALIGNED_SIZE;

    size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
    currentWorkSpace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    CalcTiling();
    PrintTilingData();
}

TilingStrategy ModulateTilingOp::SelectStrategy()
{
    if (this->tilingData.inputB * this->tilingData.inputL < this->coreNum) {
        int64_t tileD = this->tilingData.inputD / this->coreNum;
        int64_t tileDSize = this->tilingData.inputB * this->tilingData.inputL * tileD * this->dataTypeSize;
        return tileDSize < MIN_TILE_SIZE ? TilingStrategy::TilingB : TilingStrategy::TilingD;
    }

    if (this->tilingData.inputB >= this->coreNum) {
        return TilingStrategy::TilingB;
    }

    if (this->tilingData.inputB >= this->coreNum / HALF_CORE_NUM) {
        int64_t tileL = this->tilingData.inputB * this->tilingData.inputL / this->coreNum;
        int64_t tileLSize = this->tilingData.inputD * tileL * this->dataTypeSize;
        return tileLSize < MIN_TILE_SIZE ? TilingStrategy::TilingB : TilingStrategy::TilingL;
    }
    return TilingStrategy::TilingL;
}

void ModulateTilingOp::CalcTiling()
{
    TilingStrategy strategy = SelectStrategy();
    switch(strategy) {
        case TilingStrategy::TilingB:
            CalcTilingParam(TILING_DIM_B, this->tilingData.inputB);
            break;
        case TilingStrategy::TilingL:
            CalcTilingParam(TILING_DIM_L, this->tilingData.inputB * this->tilingData.inputL);
            break;
        case TilingStrategy::TilingD:
            CalcTilingParam(TILING_DIM_D, this->tilingData.inputD, true);
            break;
    }
    SetTiling();
}

void ModulateTilingOp::CalcTilingParam(int64_t TilingDim, int64_t totalElements, bool useDtiling)
{
    context->SetTilingKey(TilingDim);
    this->tilingData.tailLength = totalElements / this->coreNum;
    this->tilingData.frontLength = this->tilingData.tailLength + 1;
    this->tilingData.frontNum = totalElements % this->coreNum;
    this->tilingData.tailNum = this->tilingData.tailLength == 0 ? 0 : this->coreNum - this->tilingData.frontNum;
    if(useDtiling) {
        this->tilingData.useDTiling = 1;
    }
}

static ge::graphStatus ModulateTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling for Modulate start.");
    ModulateTilingOp tilingOp(context);
    tilingOp.Init();
    OP_LOGD(context, "Tiling for Modulate end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForModulate(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For Modulate start.");
    auto compileInfo = context->GetCompiledInfo<ModulateCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum == 0) {
        OP_LOGE(context, "coreNum %lu", compileInfo->totalCoreNum);
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For Modulate end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Modulate).Tiling(ModulateTilingFunc).TilingParse<ModulateCompileInfo>(TilingPrepareForModulate);
} // namespace optiling
