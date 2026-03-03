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
 * \file quant_batch_matmul_v3_tiling_arch20.cpp
 * \brief
 */

#include <vector>
#include <cmath>
#include "log/log.h"
#include "matmul/common/op_host/math_util.h"
#include "error_util.h"
#include "quant_batch_matmul_v3_tiling_arch20.h"
#include "quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_tiling_key.h"
using Ops::NN::MathUtil;

namespace optiling {

constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;

bool IsSocVersionArch20Pertoken(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("Arch20Pertoken: ", "context is nullptr"), return false);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2002 && context->GetInputDesc(3) != nullptr && context->GetInputDesc(5) != nullptr) {
        return true;
    }
    return false;
}

template <typename PpTilingDataType>
uint64_t Swizzl(PpTilingDataType& tilingData)
{
    uint64_t swizzlDirect = 0UL;
    uint64_t swizzlCount = 1UL;
    float m0 = tilingData.m0;
    float n0 = tilingData.n0;
    float m = tilingData.m;
    float k = tilingData.k;
    float n = tilingData.n;
    float mincost = m * k + k * n;

    for (uint32_t i = 1; i <= tilingData.blockDim; ++i) {
        int c = static_cast<int32_t>((tilingData.blockDim + i - 1) / i);
        float cost;
        // B0 + A < A0 + B
        if (i * n0 + m < m0 * c + n) {
            swizzlDirect = 1UL; // Nz
            cost = n0 * i + m0 * c;
            if (cost <= mincost) {
                mincost = cost;
                swizzlCount = i;
            }
        } else {
            swizzlDirect = 0UL; // Zn
            cost = m0 * i + n0 * c;
            if (cost < mincost) {
                mincost = cost;
                swizzlCount = i;
            }
        }
    }
    tilingData.swizzlDirect = swizzlDirect;
    tilingData.swizzlCount = swizzlCount;
    return swizzlDirect;
}

ge::graphStatus QuantBatchMatmulPertokenArch20::PostTiling()
{
    size_t tilingDataSize = sizeof(QuantMatmulPertokenTilingDataArch20);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void*>(&qbmmTilingDataArch20_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulPertokenArch20::DoTiling()
{
    auto aShape = context_->GetInputShape(0)->GetOriginShape();
    auto bShape = context_->GetInputShape(1)->GetOriginShape();
    auto cShape = context_->GetOutputShape(0)->GetOriginShape();
    size_t aDims = aShape.GetDimNum();
    size_t bDims = bShape.GetDimNum();
    if (context_->GetInputDesc(4) != nullptr && context_->GetInputDesc(3) == nullptr) {
        qbmmTilingDataArch20_.withBias = true;
    }

    std::vector<int64_t> oriShapeTable;
    if (bDims > 2 && aDims > 2) { // Input dimsize greater than 2
        oriShapeTable = {aShape[aDims - 3], aShape[aDims - 2], aShape[aDims - 1], bShape[bDims - 2]}; // NT
    } else {
        oriShapeTable = {1, aShape[aDims - 2], aShape[aDims - 1], bShape[bDims - 2]}; // NT
    }

    qbmmTilingDataArch20_.batchSize = oriShapeTable.at(0);
    qbmmTilingDataArch20_.m = oriShapeTable.at(1);
    qbmmTilingDataArch20_.k = oriShapeTable.at(2);
    qbmmTilingDataArch20_.n = oriShapeTable.at(3);
    qbmmTilingDataArch20_.m0 = CONST_256;
    qbmmTilingDataArch20_.k0 = CONST_256;
    qbmmTilingDataArch20_.n0 = CONST_128;
    qbmmTilingDataArch20_.mLoop = ops::CeilDiv(qbmmTilingDataArch20_.m, qbmmTilingDataArch20_.m0);
    qbmmTilingDataArch20_.nLoop = ops::CeilDiv(qbmmTilingDataArch20_.n, qbmmTilingDataArch20_.n0);
    qbmmTilingDataArch20_.kLoop = ops::CeilDiv(qbmmTilingDataArch20_.k, qbmmTilingDataArch20_.k0);
    qbmmTilingDataArch20_.coreLoop =
        qbmmTilingDataArch20_.batchSize * qbmmTilingDataArch20_.mLoop * qbmmTilingDataArch20_.nLoop;

    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    qbmmTilingDataArch20_.blockDim = std::min(qbmmTilingDataArch20_.coreLoop, static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic()));
    Swizzl<QuantMatmulPertokenTilingDataArch20>(qbmmTilingDataArch20_);
    uint64_t TRANS = 1;                // B trans
    uint64_t KERNEL_TEMPLATE_TYPE = 1; // basic
    uint64_t IS_PERTOKEN = 1;          // pertoken
    uint64_t OPTION_ATTRS = 0;
    tilingKey_ = GET_TPL_TILING_KEY(TRANS, KERNEL_TEMPLATE_TYPE, IS_PERTOKEN, OPTION_ATTRS);
    OP_LOGD(
        context_->GetNodeName(),
        "batchSize = %ld, m = %ld, k = %ld, n = %ld, \
        m0 = % ld, \
        k0 = % ld, n0 = % ld, mLoop = % ld, nLoop = % ld, kLoop = % ld, coreLoop = % ld, blockDim = % ld, \
        swizzlDirect = % ld, swizzlCount = % ld, withBias = % ld, \
        Tiling Key is 0x % x ", 
        qbmmTilingDataArch20_.batchSize,
        qbmmTilingDataArch20_.m, qbmmTilingDataArch20_.k, qbmmTilingDataArch20_.n, qbmmTilingDataArch20_.m0,
        qbmmTilingDataArch20_.k0, qbmmTilingDataArch20_.n0, qbmmTilingDataArch20_.mLoop, qbmmTilingDataArch20_.nLoop,
        qbmmTilingDataArch20_.kLoop, qbmmTilingDataArch20_.coreLoop, qbmmTilingDataArch20_.blockDim,
        qbmmTilingDataArch20_.swizzlDirect, qbmmTilingDataArch20_.swizzlCount, qbmmTilingDataArch20_.withBias,
        tilingKey_);
    ge::graphStatus ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
