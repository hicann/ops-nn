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
 * \file quant_matmul_activation_quant_mx_tiling.cpp
 * \brief
 */
#include <vector>
#include "quant_matmul_activation_quant_mx_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/quant_matmul_activation_quant_tiling_key.h"

using namespace QuantMatmulActivationQuantArch35TilingKey;

namespace {
constexpr int32_t MX_BASIC_API_TILING_PRIORITY = 0;
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
} // namespace

namespace optiling {

QuantMatmulActivationQuantMXBasicAPITiling::QuantMatmulActivationQuantMXBasicAPITiling(gert::TilingContext* context)
    : QuantMatmulActivationQuantHelper<AdaptiveSlidingWindowMXBasicAPITiling>(context)
{
    Reset();
}

void QuantMatmulActivationQuantMXBasicAPITiling::Reset()
{
    ResetActivationQuantTilingData(tilingData_);
    useWithoutBatchTilingData_ = false;
    tilingDataSize_ = sizeof(QMMAQ::QuantMatmulActivationQuantTilingData);
}

bool QuantMatmulActivationQuantMXBasicAPITiling::IsCapable()
{
    return IsMxQuant() && AdaptiveSlidingWindowMXBasicAPITiling::IsCapable();
}

const void* QuantMatmulActivationQuantMXBasicAPITiling::GetTilingData() const
{
    return static_cast<const void*>(&tilingData_);
}

void QuantMatmulActivationQuantMXBasicAPITiling::SetTilingData()
{
    AdaptiveSlidingWindowMXBasicAPITiling::SetTilingData();

    UpdateTilingData();
}

ge::graphStatus QuantMatmulActivationQuantMXBasicAPITiling::DoLibApiTiling()
{
    // 调用QBMM的Tiling
    auto ret = AdaptiveSlidingWindowMXBasicAPITiling::DoLibApiTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = UpdateTilingData();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulActivationQuantMXBasicAPITiling::UpdateTilingData()
{
    // 复制QBMM TilingData
    CopyV3BasicApiTilingData(AdaptiveSlidingWindowMXBasicAPITiling::tilingData_, tilingData_.mmTilingData);

    tilingDataSize_ = sizeof(QMMAQ::QuantMatmulActivationQuantTilingData);

    // 设置Quant部分的TilingData
    SetQuantParams(tilingData_);

    OP_TILING_CHECK(
        tilingData_.mmTilingData.matmulTiling.baseN % MX_BASEN_ALIGN != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "baseN",
                                              std::to_string(tilingData_.mmTilingData.matmulTiling.baseN).c_str(),
                                              "Invalid base block, "
                                              "baseN should be aligned to 32."),
        return ge::GRAPH_FAILED);

    if (tilingData_.mmTilingData.adaptiveSlidingWin.nTailTile != 0) {
        uint64_t nTailTile = tilingData_.mmTilingData.adaptiveSlidingWin.nTailTile;
        uint64_t baseN = tilingData_.mmTilingData.matmulTiling.baseN;
        uint64_t ceilDiv = ops::CeilDiv(baseN, nTailTile);
        OP_TILING_CHECK(ceilDiv % MX_BASEN_ALIGN != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "CeilDiv(baseN, nTailTile)",
                                                              std::to_string(ceilDiv).c_str(),
                                                              "CeilDiv(baseN, nTailTile) should be aligned to 32."),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t QuantMatmulActivationQuantMXBasicAPITiling::GetKernelType() const
{
    return isAFullLoad_ ? TPL_FULLLOAD : TPL_NO_FULLLOAD;
}

uint64_t QuantMatmulActivationQuantMXBasicAPITiling::GetTilingKey() const
{
    uint64_t bFormat = static_cast<ge::Format>(GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat())) ?
                           TPL_B_FORMAT_NZ :
                           TPL_B_FORMAT_ND;
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              GetKernelType(), bFormat);
}

// 为算子QuantMatmulActivationQuant注册Tiling类QuantMatmulActivationQuantMXBasicAPITiling，唯一Tiling实现
REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantMatmulActivationQuant, QuantMatmulActivationQuantMXBasicAPITiling,
                                   supportedNpuArch, MX_BASIC_API_TILING_PRIORITY);

} // namespace optiling
