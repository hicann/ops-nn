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
 * \file quant_batch_matmul_inplace_add_mx_basic_api_tiling.cpp
 * \brief
 */
#include <vector>
#include "quant_batch_matmul_inplace_add_mx_basic_api_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/quant_batch_matmul_inplace_add_tiling_key.h"

using namespace QuantBatchMatmulInplaceAddArch35TilingKey;

namespace {
constexpr int32_t MX_BASIC_API_TILING_PRIORITY = 0;
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
} // namespace

namespace optiling {

QuantBatchMatmulInplaceAddMXBasicAPITiling::QuantBatchMatmulInplaceAddMXBasicAPITiling(gert::TilingContext* context)
    : QuantBatchMatmulInplaceAddHelper<AdaptiveSlidingWindowMXBasicAPITiling>(context)
{
    Reset();
}

void QuantBatchMatmulInplaceAddMXBasicAPITiling::Reset()
{
    ResetInplaceTilingData(withoutBatchTilingData_);
    tilingDataSize_ = sizeof(QMMIA::QbmmiaWithoutBatchTilingData);
}

bool QuantBatchMatmulInplaceAddMXBasicAPITiling::IsCapable() { return IsMxQuant() && inputParams_.batchC == 1UL; }

const void* QuantBatchMatmulInplaceAddMXBasicAPITiling::GetTilingData() const { return &withoutBatchTilingData_; }

void QuantBatchMatmulInplaceAddMXBasicAPITiling::SetTilingData()
{
    AdaptiveSlidingWindowMXBasicAPITiling::SetTilingData();
    UpdateTilingData();
}

ge::graphStatus QuantBatchMatmulInplaceAddMXBasicAPITiling::DoLibApiTiling()
{
    auto ret = AdaptiveSlidingWindowMXBasicAPITiling::DoLibApiTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    UpdateTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantBatchMatmulInplaceAddMXBasicAPITiling::GetKernelType() const
{
    return isAFullLoad_ ? TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH :
                          TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH;
}

uint64_t QuantBatchMatmulInplaceAddMXBasicAPITiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              GetKernelType());
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantBatchMatmulInplaceAdd, QuantBatchMatmulInplaceAddMXBasicAPITiling,
                                   supportedNpuArch, MX_BASIC_API_TILING_PRIORITY);

void QuantBatchMatmulInplaceAddMXBasicAPITiling::UpdateTilingData()
{
    CopyV3WithoutBatchTilingData(AdaptiveSlidingWindowMXBasicAPITiling::withoutBatchTilingData_,
                                 withoutBatchTilingData_);
    tilingDataSize_ = sizeof(QMMIA::QbmmiaWithoutBatchTilingData);
}

} // namespace optiling
