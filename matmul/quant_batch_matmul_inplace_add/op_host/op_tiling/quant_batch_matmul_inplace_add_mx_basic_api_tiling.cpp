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

QuantBatchMatmulInplaceAddMXBasicAPITiling::QuantBatchMatmulInplaceAddMXBasicAPITiling(
    gert::TilingContext* context)
    : QuantBatchMatmulInplaceAddHelper<AdaptiveSlidingWindowMXBasicAPITiling>(context)
{
    Reset();
}

void QuantBatchMatmulInplaceAddMXBasicAPITiling::Reset()
{
    ResetInplaceTilingData(basicTilingData_);
    withoutBatchTilingData_ = QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData();
    useWithoutBatchTilingData_ = false;
    tilingDataSize_ = sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData);
}

bool QuantBatchMatmulInplaceAddMXBasicAPITiling::IsCapable()
{
    return IsMxQuant() && inputParams_.batchC == 1UL;
}

const void* QuantBatchMatmulInplaceAddMXBasicAPITiling::GetTilingData() const
{
    return useWithoutBatchTilingData_ ? static_cast<const void*>(&withoutBatchTilingData_) :
                                        static_cast<const void*>(&basicTilingData_);
}

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
    if (IsTensorapiCapable()) {
        return isAFullLoad_ ? TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH :
                              TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH;
    }
    return isAFullLoad_ ? TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI : TPL_NO_VEC_EPILOGUE_WITH_MMAPI;
}

uint64_t QuantBatchMatmulInplaceAddMXBasicAPITiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB), GetKernelType());
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulInplaceAdd, QuantBatchMatmulInplaceAddMXBasicAPITiling, supportedNpuArch,
    MX_BASIC_API_TILING_PRIORITY);

void QuantBatchMatmulInplaceAddMXBasicAPITiling::UpdateTilingData()
{
    useWithoutBatchTilingData_ = IsTensorapiCapable();
    if (useWithoutBatchTilingData_) {
        SetWithoutBatchTilingData();
        tilingDataSize_ = sizeof(QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData);
        return;
    }
    CopyV3BasicApiTilingData(AdaptiveSlidingWindowMXBasicAPITiling::tilingData_, basicTilingData_);
    tilingDataSize_ = sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData);
}

void QuantBatchMatmulInplaceAddMXBasicAPITiling::SetWithoutBatchTilingData()
{
    const auto& v3TilingData = AdaptiveSlidingWindowMXBasicAPITiling::tilingData_;
    const auto& matmulTiling = v3TilingData.matmulTiling;
    withoutBatchTilingData_.m = static_cast<uint32_t>(inputParams_.mSize);
    withoutBatchTilingData_.n = static_cast<uint32_t>(inputParams_.nSize);
    withoutBatchTilingData_.k = static_cast<uint32_t>(inputParams_.kSize);
    withoutBatchTilingData_.scaleKL1 = matmulTiling.scaleKL1;
    withoutBatchTilingData_.baseM = static_cast<uint16_t>(basicTiling_.baseM);
    withoutBatchTilingData_.baseN = static_cast<uint16_t>(basicTiling_.baseN);
    withoutBatchTilingData_.baseK = static_cast<uint16_t>(basicTiling_.baseK);
    withoutBatchTilingData_.stepKa = matmulTiling.stepKa;
    withoutBatchTilingData_.stepKb = matmulTiling.stepKb;
    withoutBatchTilingData_.groupSizeM = static_cast<uint16_t>(inputParams_.groupSizeM);
    withoutBatchTilingData_.groupSizeN = static_cast<uint16_t>(inputParams_.groupSizeN);
    withoutBatchTilingData_.groupSizeK = static_cast<uint16_t>(inputParams_.groupSizeK);
    withoutBatchTilingData_.mTailTile = static_cast<uint16_t>(adaptiveWin_.mTailTile);
    withoutBatchTilingData_.nTailTile = static_cast<uint16_t>(adaptiveWin_.nTailTile);
    withoutBatchTilingData_.mBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.mBaseTailSplitCnt);
    withoutBatchTilingData_.nBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.nBaseTailSplitCnt);
    withoutBatchTilingData_.mTailMain = static_cast<uint16_t>(adaptiveWin_.mTailMain);
    withoutBatchTilingData_.nTailMain = static_cast<uint16_t>(adaptiveWin_.nTailMain);
    withoutBatchTilingData_.x1QuantMode = static_cast<uint8_t>(v3TilingData.params.x1QuantMode);
    withoutBatchTilingData_.x2QuantMode = static_cast<uint8_t>(v3TilingData.params.x2QuantMode);
    withoutBatchTilingData_.isBias = matmulTiling.isBias;
    withoutBatchTilingData_.biasDtype = static_cast<uint8_t>(inputParams_.biasDtype);
    withoutBatchTilingData_.nBufferNum = matmulTiling.nBufferNum;
    withoutBatchTilingData_.dbL0C = matmulTiling.dbL0C;
}

} // namespace optiling
