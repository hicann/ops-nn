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
 * \file rms_norm_quant_v2_tiling_arch35_full_load.cpp
 * \brief
 */

#include "norm/norm_common/op_host/norm_tiling_check_common.h"
#include "rms_norm_quant_v2_tiling.h"
#include "util/math_util.h"
#include "op_api/op_util.h"

using namespace Ops::Base;
using namespace ge;

namespace optiling {
constexpr int64_t R_MAX_VALUE = 16384;
constexpr uint32_t CONST_ZERO = 0;
constexpr uint32_t CONST_ONE = 1;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t LOG_2 = 2;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t RETAINED_SIZE_256 = 256;
constexpr uint32_t FLOAT_SIZE = 4;


bool RmsNormQuantV2RegbaseTilingFullLoad::IsCapable()
{
    tilingParams.rXDtypeAlign = Ops::Base::CeilAlign(tilingParams.r, tilingParams.xDtypeAlignNum);  //r向上对齐到 一个Block能容纳x个数 的整数倍 
    tilingParams.rAlign = Ops::Base::CeilAlign(tilingParams.r, static_cast<int64_t>(BLOCK_SIZE)); //r向上对齐到 BlockSize 的整数倍
    int64_t tmpPower = std::floor(std::log(tilingParams.rXDtypeAlign -1) / std::log(LOG_2));
    tilingParams.binaryAdd = std::pow(LOG_2, tmpPower);  //二分累加折叠点

    int64_t tmpUBSize = Ops::Base::CeilDiv(Ops::Base::CeilDiv(tilingParams.binaryAdd, tilingParams.vecLength),static_cast<int64_t>(BLOCK_SIZE)) * BLOCK_SIZE;

    int64_t betaNum        = tilingParams.hasBeta ? CONST_ONE : CONST_ZERO;
    int64_t scalesNum      = tilingParams.hasScales2 ? CONST_TWO : CONST_ONE;
    int64_t zeroPointsNum  = (tilingParams.hasZeroPoints1 ? CONST_ONE : CONST_ZERO)
                            + (tilingParams.hasZeroPoints2 ? CONST_ONE : CONST_ZERO);
    int64_t yNum           = tilingParams.hasY2 ? CONST_TWO : CONST_ONE;
    int64_t r = tilingParams.rAlign;
    int64_t yDtypeSize = context_->GetOutputDesc(Y1_INDEX)->GetDataType() == ge::DT_INT4 ? 1 : ge::GetSizeByDataType(context_->GetOutputDesc(Y1_INDEX)->GetDataType());
    int64_t rstdBufSizePerRow = (tilingParams.rstdFlag != 0) ? (DOUBLE_BUFFER * FLOAT_SIZE) : FLOAT_SIZE;
    tilingParams.ubFactor = ((static_cast<int64_t>(tilingParams.maxUbSize) - RETAINED_SIZE_256 - r * tilingParams.xDtypeSize - r * betaNum * tilingParams.xDtypeSize
                             - r * scalesNum * tilingParams.scaleDtypeSize - r * zeroPointsNum * tilingParams.zeroPointDtypeSize)/
                             (DOUBLE_BUFFER * r * tilingParams.xDtypeSize + DOUBLE_BUFFER * r * yNum * yDtypeSize
                             + rstdBufSizePerRow + tmpUBSize));
    OP_CHECK_IF(tilingParams.r > R_MAX_VALUE,
                    OP_LOGI(context_->GetNodeName(),
                            "AR full load template is not capable. actual r is %ld, larger than %ld", tilingParams.r, R_MAX_VALUE),
                    return false);
    OP_CHECK_IF(tilingParams.ubFactor < CONST_ONE,
                    OP_LOGI(context_->GetNodeName(),
                            "AR full load template is not capable. actual ubFactor is %ld, can not full load ", tilingParams.ubFactor),
                    return false);
    return true;
}

ge::graphStatus RmsNormQuantV2RegbaseTilingFullLoad::DoOpTiling()
{
    OP_LOGD(nodeName.c_str(), "Enter RmsNormQuantV2RegbaseTiling DoOpTiling.");
    if (tilingParams.needGetCompileInfo) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tilingParams.maxUbSize);
        tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    }
    
    // Set align params
    tilingParams.rScaleAlign = Ops::Base::CeilAlign(tilingParams.r, tilingParams.scaleDtypeAlignNum); //  能容纳的scale个数  
    tilingParams.rZeroPointAlign = Ops::Base::CeilAlign(tilingParams.r, tilingParams.zeroPointDtypeAlignNum);

    tilingParams.blockFactor = Ops::Base::CeilDiv(tilingParams.a, tilingParams.totalCoreNum);
    tilingParams.ubFactor = std::min(tilingParams.ubFactor,tilingParams.blockFactor);
    tilingParams.usedCoreNum = Ops::Base::CeilDiv(tilingParams.a, tilingParams.blockFactor);
    tilingParams.blockTail = tilingParams.a - (tilingParams.usedCoreNum - 1) * tilingParams.blockFactor;
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void RmsNormQuantV2RegbaseTilingFullLoad::SetTilingData()
{
    tilingData.a = tilingParams.a;
    tilingData.r = tilingParams.r;
    tilingData.q = tilingParams.q;
    tilingData.blockFactor = tilingParams.blockFactor;
    tilingData.blockTail = tilingParams.blockTail;
    tilingData.ubFactor = tilingParams.ubFactor;
    tilingData.binaryAdd = tilingParams.binaryAdd;
    tilingData.optionMask = tilingParams.optionMask;
    tilingData.divMode = tilingParams.divMode;
    tilingData.dstDtype = tilingParams.dstDtype;
    tilingData.epsilon = tilingParams.epsilon;
    tilingData.avgFactor = tilingParams.avgFactor;
    tilingData.rstdFlag = tilingParams.rstdFlag;
}

void RmsNormQuantV2RegbaseTilingFullLoad::PrintTilingData()
{
    OP_LOGI(
        nodeName.c_str(),
        "TilingData a: %ld, r: %ld, q: %ld, blockFactor: %ld, "
        "blockTail: %ld, ubFactor: %ld, binaryAdd: %ld, "
        "optionMask: %lu, divMode: %ld, dstDtype: %ld, "
        "epsilon: %f, avgFactor: %f, rstdFlag: %u.",
        tilingData.a, tilingData.r, tilingData.q, tilingData.blockFactor,
        tilingData.blockTail, tilingData.ubFactor, tilingData.binaryAdd,
        tilingData.optionMask, tilingData.divMode, tilingData.dstDtype,
        tilingData.epsilon, tilingData.avgFactor, tilingData.rstdFlag);
}

ge::graphStatus RmsNormQuantV2RegbaseTilingFullLoad::PostTiling()
{
    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %ld.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t usrWorkspaceSize = tilingParams.workspaceSize;
    size_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        sizeof(tilingData) > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(tilingData),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));
    return ge::GRAPH_SUCCESS;
}
uint64_t RmsNormQuantV2RegbaseTilingFullLoad::GetTilingKey() const
{
    rms_norm_quant_v2::RmsNormQuantV2TilingKey tilingKey;
    tilingKey.SetComputeMode(rms_norm_quant_v2::ComputeMode::FULL_LOAD);
    uint64_t key = tilingKey.GetTilingKey();
    OP_LOGI(nodeName.c_str(), "TilingKey is %lu.", key);
    return key;
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormQuantV2, RmsNormQuantV2RegbaseTilingFullLoad, 100);
REGISTER_OPS_TILING_TEMPLATE(RmsNormQuantV3, RmsNormQuantV2RegbaseTilingFullLoad, 100);
} // namespace optiling
