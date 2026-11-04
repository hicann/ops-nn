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
#include <string>
#include <vector>
#include "base/err_msg.h"
#include <vector>
#include "quant_batch_matmul_inplace_add_mx_basic_api_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/quant_batch_matmul_inplace_add_tiling_key.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace QuantBatchMatmulInplaceAddArch35TilingKey;

namespace {
constexpr int32_t MX_BASIC_API_TILING_PRIORITY = 0;
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(Ops::Base::DAV_3510),
                                               static_cast<int32_t>(NpuArch::DAV_9201)};
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

bool QuantBatchMatmulInplaceAddMXBasicAPITiling::IsCapable()
{
    return IsMxBasicApiCapable() && inputParams_.batchC == 1UL;
}

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
    // TE HIF4新方案合入并且适配后放开，如果未命中QBMMV3中BLAZE tiling
    // 基类不会生成 withoutBatch 数据，直接报错避免下发全零 tiling。
    if (inputParams_.isHif4PerGroup && basicTiling_.baseK % (qmmv3_tiling_const::MXFP_DIVISOR_SIZE * 2UL) != 0UL) {
        {
            const std::string _msg = "When the dtype of x1 and x2 is HIFLOAT4, baseK[" +
                                     std::to_string(basicTiling_.baseK) + "] must be an even multiple of 64.";
            OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "%s", _msg.c_str());
            const std::vector<const char*> _msgKey = {"message"};
            const std::vector<const char*> _msgValue = {_msg.c_str()};
            REPORT_PREDEFINED_ERR_MSG("EZ1001", _msgKey, _msgValue);
        }
        return ge::GRAPH_FAILED;
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
