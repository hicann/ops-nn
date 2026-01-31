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
#include "op_util.h"

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
    tilingParams.ubFactor = ((static_cast<int64_t>(tilingParams.maxUbSize) - RETAINED_SIZE_256 - r * tilingParams.xDtypeSize - r * betaNum * tilingParams.xDtypeSize 
                             - r * scalesNum * tilingParams.scaleDtypeSize - r * zeroPointsNum * tilingParams.zeroPointDtypeSize)/
                             (DOUBLE_BUFFER * r * tilingParams.xDtypeSize + DOUBLE_BUFFER * r * yNum * yDtypeSize
                             + FLOAT_SIZE + tmpUBSize));
    OP_CHECK_IF(tilingParams.r > R_MAX_VALUE,
                    OP_LOGI(context_->GetNodeName(),
                            "AR full load template is not capable. actual r is %ld, larger than %ld", tilingParams.r, R_MAX_VALUE),
                    return false);
    OP_CHECK_IF(tilingParams.ubFactor < CONST_ONE,
                    OP_LOGI(context_->GetNodeName(),
                            "AR full load template is not capable. actual ubFactor is %ld", tilingParams.ubFactor, R_MAX_VALUE),
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
    tilingData.set_a(tilingParams.a);
    tilingData.set_r(tilingParams.r);
    tilingData.set_q(tilingParams.q);
    tilingData.set_blockFactor(tilingParams.blockFactor);
    tilingData.set_blockTail(tilingParams.blockTail);
    tilingData.set_ubFactor(tilingParams.ubFactor);
    tilingData.set_binaryAdd(tilingParams.binaryAdd);
    tilingData.set_optionMask(tilingParams.optionMask);
    tilingData.set_divMode(tilingParams.divMode);
    tilingData.set_dstDtype(tilingParams.dstDtype);
    tilingData.set_epsilon(tilingParams.epsilon);
    tilingData.set_avgFactor(tilingParams.avgFactor);
}

void RmsNormQuantV2RegbaseTilingFullLoad::PrintTilingData()
{
    OP_LOGI(
        nodeName.c_str(),
        "TilingData a: %lu, r: %lu, q: %lu, blockFactor: %lu, "
        "blockTail: %lu, ubFactor: %lu, binaryAdd: %lu, "
        "optionMask: %lu, divMode: %lu, dstDtype: %lu, "
        "epsilon: %f, avgFactor: %f.",
        tilingData.get_a(), tilingData.get_r(), tilingData.get_q(), tilingData.get_blockFactor(),
        tilingData.get_blockTail(), tilingData.get_ubFactor(), tilingData.get_binaryAdd(),
        tilingData.get_optionMask(), tilingData.get_divMode(), tilingData.get_dstDtype(), 
        tilingData.get_epsilon(), tilingData.get_avgFactor());
}
ge::graphStatus RmsNormQuantV2RegbaseTilingFullLoad::PostTiling()
{
    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

    size_t usrWorkspaceSize = tilingParams.workspaceSize;
    size_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}
uint64_t RmsNormQuantV2RegbaseTilingFullLoad::GetTilingKey() const
{
    uint64_t tilingKey = RMSNORMQUANTV2_REGBASE_NORMAL;
    OP_LOGI(nodeName.c_str(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormQuantV2, RmsNormQuantV2RegbaseTilingFullLoad, 100);
} // namespace optiling
