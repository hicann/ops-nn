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
 * \file quant_matmul_activation_mx_quant_tiling.cpp
 * \brief MX matmul activation quantization tiling and serialization.
 */
#include <string>
#include <vector>
#include "quant_matmul_activation_mx_quant_tiling.h"
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
    withoutBatchTilingData_ = {};
    useWithoutBatchTilingData_ = false;
    tilingDataSize_ = sizeof(QMMAQ::QMMAQTilingData);
}

bool QuantMatmulActivationQuantMXBasicAPITiling::IsCapable() { return IsMxQuant(); }

bool QuantMatmulActivationQuantMXBasicAPITiling::CheckCoreNum() const
{
    // CoreNum==0 is rejected by the common platform-info checks before DoOpTiling().
    if (compileInfo_.aivNum != qmmv3_tiling_const::CORE_RATIO * compileInfo_.aicNum) {
        OP_LOGE(inputParams_.opName,
                "QuantMatmulActivationQuant MX tiling requires aicNum:aivNum = 1:2; got aicNum=%u, aivNum=%u.",
                compileInfo_.aicNum, compileInfo_.aivNum);
        return false;
    }
    return true;
}

const void* QuantMatmulActivationQuantMXBasicAPITiling::GetTilingData() const
{
    return GetBatchMode() == TPL_WITHOUT_BATCH ? static_cast<const void*>(&withoutBatchTilingData_) :
                                                 static_cast<const void*>(&tilingData_);
}

ge::graphStatus QuantMatmulActivationQuantMXBasicAPITiling::DoLibApiTiling()
{
    // 调用QBMM的Tiling
    auto ret = AdaptiveSlidingWindowMXBasicAPITiling::DoLibApiTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Base tile sizes and L1 buffer counts are finalized by DoLibApiTiling.
    // Validate and serialize only now, and propagate any failure to the caller.
    return UpdateTilingData();
}

ge::graphStatus QuantMatmulActivationQuantMXBasicAPITiling::UpdateTilingData()
{
    const auto ret = CopyMatmulTilingData(AdaptiveSlidingWindowMXBasicAPITiling::tilingData_, tilingData_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    SetQuantParams(tilingData_);
    const auto validateRet = ValidateTilingData();
    if (validateRet != ge::GRAPH_SUCCESS) {
        return validateRet;
    }
    SetFinalTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulActivationQuantMXBasicAPITiling::ValidateTilingData() const
{
    const bool isSwiglu = tilingData_.activationType == QMMAQ::ActivationAlg::SWIGLU;
    const uint64_t baseNAlign = isSwiglu ? SWIGLU_BASEN_ALIGN : GELU_BASEN_ALIGN;
    OP_TILING_CHECK(
        tilingData_.baseN % baseNAlign != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "baseN", std::to_string(tilingData_.baseN).c_str(),
                                              "Invalid base block, "
                                              "baseN should be aligned to the required vector/MMAD boundary."),
        return ge::GRAPH_FAILED);

    if (!isSwiglu && tilingData_.nTailTile != 0U) {
        const uint64_t ceilDiv = ops::CeilDiv(static_cast<uint64_t>(tilingData_.baseN),
                                              static_cast<uint64_t>(tilingData_.nTailTile));
        OP_TILING_CHECK(ceilDiv % GELU_BASEN_ALIGN != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "CeilDiv(baseN, nTailTile)",
                                                              std::to_string(ceilDiv).c_str(),
                                                              "CeilDiv(baseN, nTailTile) should be aligned to 32."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void QuantMatmulActivationQuantMXBasicAPITiling::SetFinalTilingData()
{
    if (GetBatchMode() == TPL_WITHOUT_BATCH) {
        withoutBatchTilingData_ = {tilingData_.m,
                                   tilingData_.n,
                                   tilingData_.k,
                                   tilingData_.kL1,
                                   tilingData_.scaleKL1,
                                   tilingData_.dstTypeMax,
                                   tilingData_.baseM,
                                   tilingData_.baseN,
                                   tilingData_.baseK,
                                   tilingData_.mTailTile,
                                   tilingData_.nTailTile,
                                   tilingData_.mBaseTailSplitCnt,
                                   tilingData_.nBaseTailSplitCnt,
                                   tilingData_.mTailMain,
                                   tilingData_.nTailMain,
                                   tilingData_.nBufferNum,
                                   tilingData_.isBias,
                                   tilingData_.dbL0C,
                                   tilingData_.weightMustHitL2,
                                   tilingData_.activationType,
                                   tilingData_.scaleAlg,
                                   tilingData_.roundMode};
        tilingDataSize_ = sizeof(QMMAQ::QMMAQWithoutBatchTilingData);
    } else {
        CopyBatchTilingData(AdaptiveSlidingWindowMXBasicAPITiling::tilingData_.params, tilingData_);
        tilingDataSize_ = sizeof(QMMAQ::QMMAQTilingData);
    }
}

uint64_t QuantMatmulActivationQuantMXBasicAPITiling::GetKernelType() const
{
    if (tilingData_.activationType == QMMAQ::ActivationAlg::SWIGLU) {
        return isAFullLoad_ ? TPL_SWIGLU_FULLLOAD : TPL_SWIGLU_NO_FULLLOAD;
    }
    return isAFullLoad_ ? TPL_GELU_FULLLOAD : TPL_GELU_NO_FULLLOAD;
}

uint64_t QuantMatmulActivationQuantMXBasicAPITiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              GetBatchMode(), GetKernelType());
}

uint64_t QuantMatmulActivationQuantMXBasicAPITiling::GetBatchMode() const
{
    return inputParams_.batchA == 1UL && inputParams_.batchB == 1UL && inputParams_.batchC == 1UL ? TPL_WITHOUT_BATCH :
                                                                                                    TPL_WITH_BATCH;
}

// 为算子QuantMatmulActivationQuant注册Tiling类QuantMatmulActivationQuantMXBasicAPITiling，唯一Tiling实现
REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantMatmulActivationQuant, QuantMatmulActivationQuantMXBasicAPITiling,
                                   supportedNpuArch, MX_BASIC_API_TILING_PRIORITY);

} // namespace optiling
