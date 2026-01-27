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
 * \file modulate_regbase_tiling.cpp
 * \brief
 */

#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_key.h"
#include "modulate_tiling.h"
#include "modulate_regbase_tiling.h"
#include "../op_kernel/arch35/modulate_struct.h"

namespace optiling {
static constexpr uint64_t SECTOR_LINE = 128;
static constexpr uint64_t BLOCK_SIZE = 32;
static constexpr uint64_t MIN_MULTI_ROWS = 2;
static constexpr uint64_t X_Y_BUFFER_NUM = 2;
static constexpr uint64_t DOUBLE_BUFFER = 2;
static constexpr uint64_t INPUT_X = 0;
static constexpr uint64_t INPUT_SCALE = 1;
static constexpr uint64_t INPUT_SHIFT = 2;
static constexpr uint64_t DIM_B = 0;
static constexpr uint64_t DIM_L = 1;
static constexpr uint64_t DIM_D = 2;
static constexpr uint64_t MIN_TILE_SIZE = 32 * 1024;

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus ModulateTilingForRegbase::GetPlatformInfo()
{
    OP_LOGD(opName_, "ModulateTilingForRegbase GetPlatformInfo.");
    auto compileInfo = static_cast<const ModulateCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<uint64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus ModulateTilingForRegbase::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "ModulateTilingForRegbase GetShapeAttrsInfo.");
    // 获取输入shape和dtype
    const gert::StorageShape* xShape = context_->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    const gert::StorageShape* scaleShape = context_->GetOptionalInputShape(INPUT_SCALE);
    const gert::StorageShape* shiftShape = context_->GetOptionalInputShape(INPUT_SHIFT);

    auto xDesc = context_->GetInputDesc(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDType_ = xDesc->GetDataType();
    xDtypeSize_ = ge::GetSizeByDataType(xDType_);
    OP_CHECK_IF(
        xDtypeSize_ <= 0,
        OP_LOGE(
            opName_, "Get invalid dtype size. x dtype [%s], size: %u.", Ops::Base::ToString(xDType_).c_str(),
            xDtypeSize_),
        return ge::GRAPH_FAILED);
    xAlign_ = BLOCK_SIZE / xDtypeSize_;
    isScale_ = (scaleShape != nullptr);
    isShift_ = (shiftShape != nullptr);
    tilingData_.inputB = xShape->GetStorageShape().GetDim(DIM_B);
    tilingData_.inputL = xShape->GetStorageShape().GetDim(DIM_L);
    tilingData_.inputD = xShape->GetStorageShape().GetDim(DIM_D);

    return ge::GRAPH_SUCCESS;
}

// 3、计算数据切分TilingData
ge::graphStatus ModulateTilingForRegbase::DoOpTiling()
{
    OP_LOGD(opName_, "ModulateTilingForRegbase DoOpTiling.");

    tilingStrategy_ = SelectStrategy();
    switch(tilingStrategy_) {
        case TilingRegbaseStrategy::TilingL:
            CalcTilingParamL(tilingData_.inputB * tilingData_.inputL);
            break;
        case TilingRegbaseStrategy::TilingD:
            CalcTilingParamD(tilingData_.inputD);
            break;
    }

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus ModulateTilingForRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t ModulateTilingForRegbase::GetTilingKey() const
{
    OP_LOGD(opName_, "ModulateTilingForRegbase GetTilingKey.");

    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint8_t>(tilingStrategy_), isScale_, isShift_);
    OP_LOGD(opName_, "tilingKey is: [%lu]", tilingKey);
    OP_LOGD(opName_, "tilingStrategy_, isScale_, isShift_ is: [%u, %u, %u]", tilingStrategy_, isScale_, isShift_);

    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus ModulateTilingForRegbase::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus ModulateTilingForRegbase::PostTiling()
{
    OP_LOGD(opName_, "ModulateTilingForRegbase PostTiling.");

    // 设置workspace大小
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           &tilingData_, sizeof(ModulateRegbaseTilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(ModulateRegbaseTilingData));

    return ge::GRAPH_SUCCESS;
}

TilingRegbaseStrategy ModulateTilingForRegbase::SelectStrategy()
{
    if (tilingData_.inputB * tilingData_.inputL < totalCoreNum_) {
        uint64_t tileD = tilingData_.inputD / totalCoreNum_;
        uint64_t tileDSize = tilingData_.inputB * tilingData_.inputL * tileD * xDtypeSize_;
        return tileDSize < MIN_TILE_SIZE ? TilingRegbaseStrategy::TilingL : TilingRegbaseStrategy::TilingD;
    }

    return TilingRegbaseStrategy::TilingL;
}

void ModulateTilingForRegbase::CalcTilingParamL(const uint64_t &dataNum)
{
    blockDim_ = std::max(static_cast<uint64_t>(1), std::min(dataNum, totalCoreNum_));
    tilingData_.formerCoreNum = dataNum % blockDim_;
    tilingData_.tailCoreNum = blockDim_ - tilingData_.formerCoreNum;
    tilingData_.formerDataNum = Ops::Base::CeilDiv(dataNum, static_cast<uint64_t>(blockDim_));
    tilingData_.tailDataNum = dataNum / blockDim_;
    uint64_t inputDAlign = Ops::Base::CeilAlign(tilingData_.inputD, xAlign_);
    uint64_t hasScaleShift = static_cast<uint64_t>(isScale_ && isShift_);
    uint64_t minMultiRowSize = (X_Y_BUFFER_NUM * DOUBLE_BUFFER * MIN_MULTI_ROWS + 1 + hasScaleShift) * inputDAlign * xDtypeSize_;
    if (minMultiRowSize < ubSize_) {
        tilingData_.maxCopyRows = (ubSize_ / (inputDAlign * xDtypeSize_) - hasScaleShift - 1) / DOUBLE_BUFFER / X_Y_BUFFER_NUM;
        tilingData_.maxDInUB = tilingData_.inputD;
        tilingData_.maxCalcNum = tilingData_.inputD;
    } else {
        // 仅处理1行
        tilingData_.maxCopyRows = 1;
        tilingData_.maxDInUB = ubSize_ / (X_Y_BUFFER_NUM * DOUBLE_BUFFER + 1 + hasScaleShift);
        tilingData_.maxDInUB = Ops::Base::FloorAlign(tilingData_.maxDInUB, BLOCK_SIZE) / xDtypeSize_;
        tilingData_.maxCalcNum = tilingData_.inputD < tilingData_.maxDInUB ? tilingData_.inputD : tilingData_.maxDInUB;
    }
    tilingData_.maxCopyRows = std::min(tilingData_.maxCopyRows, tilingData_.formerDataNum);
}

void ModulateTilingForRegbase::CalcTilingParamD(const uint64_t &dataNum)
{
    blockDim_ = std::max(static_cast<uint64_t>(1), std::min(dataNum, totalCoreNum_));
    tilingData_.formerCoreNum = dataNum % blockDim_;
    tilingData_.tailCoreNum = blockDim_ - tilingData_.formerCoreNum;
    tilingData_.formerDataNum = Ops::Base::CeilDiv(dataNum, static_cast<uint64_t>(blockDim_));
    tilingData_.tailDataNum = dataNum / blockDim_;
    uint64_t hasScaleShift = static_cast<uint64_t>(isScale_ && isShift_);
    // 仅处理1行
    tilingData_.maxCopyRows = 1;
    tilingData_.maxDInUB = ubSize_ / (X_Y_BUFFER_NUM * DOUBLE_BUFFER + 1 + hasScaleShift);
    tilingData_.maxDInUB = Ops::Base::FloorAlign(tilingData_.maxDInUB, BLOCK_SIZE) / xDtypeSize_;
    tilingData_.maxCalcNum = tilingData_.formerDataNum < tilingData_.maxDInUB ? tilingData_.formerDataNum : tilingData_.maxDInUB;
}

bool ModulateTilingForRegbase::IsCapable()
{
    return true;
}

void ModulateTilingForRegbase::PrintTilingData()
{
    OP_LOGD(opName_, "inputB:           %lu.", tilingData_.inputB);
    OP_LOGD(opName_, "inputL:           %lu.", tilingData_.inputL);
    OP_LOGD(opName_, "inputD:           %lu.", tilingData_.inputD);
    OP_LOGD(opName_, "formerCoreNum:    %lu.", tilingData_.formerCoreNum);
    OP_LOGD(opName_, "tailCoreNum:      %lu.", tilingData_.tailCoreNum);
    OP_LOGD(opName_, "formerDataNum:    %lu.", tilingData_.formerDataNum);
    OP_LOGD(opName_, "tailDataNum:      %lu.", tilingData_.tailDataNum);
    OP_LOGD(opName_, "maxDInUB:         %lu.", tilingData_.maxDInUB);
    OP_LOGD(opName_, "maxCalcNum:       %lu.", tilingData_.maxCalcNum);
    OP_LOGD(opName_, "maxCopyRows:      %lu.", tilingData_.maxCopyRows);
}


void ModulateTilingForRegbase::SetTilingData()
{
    OP_LOGD(opName_, "ModulateTilingForRegbase SetTilingData.");
    ModulateRegbaseTilingData* tilingData =
        context_->GetTilingData<ModulateRegbaseTilingData>();
    tilingData->inputB = tilingData_.inputB;
    tilingData->inputL = tilingData_.inputL;
    tilingData->inputD = tilingData_.inputD;
    tilingData->formerCoreNum = tilingData_.formerCoreNum;
    tilingData->tailCoreNum = tilingData_.tailCoreNum;
    tilingData->formerDataNum = tilingData_.formerDataNum;
    tilingData->tailDataNum = tilingData_.tailDataNum;
}


} // namespace optiling
