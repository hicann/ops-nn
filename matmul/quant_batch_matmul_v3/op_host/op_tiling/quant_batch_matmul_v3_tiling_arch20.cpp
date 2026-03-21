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

constexpr uint32_t X1_IDX = 0;
constexpr uint32_t X2_IDX = 1;
constexpr uint32_t SCALE_IDX = 2;
constexpr uint32_t OFFSET_IDX = 3;
constexpr uint32_t BIAS_IDX = 4;
constexpr uint32_t PERTOKEN_IDX = 5;
constexpr uint32_t CONST_ZERO = 0;
constexpr uint32_t CONST_ONE = 1;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t CONST_THREE = 3;

constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;

bool IsSocVersionArch20Pertoken(const gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("Arch20Pertoken: ", "context is nullptr"), return false);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2002 && context->GetOptionalInputDesc(PERTOKEN_IDX) != nullptr) {
        return true;
    }
    return false;
}

template <typename PpTilingDataType>
uint64_t Swizzle(PpTilingDataType& tilingData)
{
    uint64_t swizzleDirect = 0UL;
    uint64_t swizzleCount = 1UL;
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
            swizzleDirect = 1UL; // Nz
            cost = n0 * i + m0 * c;
            if (cost <= mincost) {
                mincost = cost;
                swizzleCount = i;
            }
        } else {
            swizzleDirect = 0UL; // Zn
            cost = m0 * i + n0 * c;
            if (cost < mincost) {
                mincost = cost;
                swizzleCount = i;
            }
        }
    }
    tilingData.swizzleDirect = swizzleDirect;
    tilingData.swizzleCount = swizzleCount;
    return swizzleDirect;
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
    context_->SetBlockDim(qbmmTilingDataArch20_.blockDim);

    size_t sysWorkspaceSize = static_cast<size_t>(16 * 1024 * 1024);  // 24M same as ppmatmul tiling
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);

    currentWorkSpace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulPertokenArch20::DoTiling()
{
    auto aShape = context_->GetInputShape(X1_IDX)->GetOriginShape();
    auto bShape = context_->GetInputShape(X2_IDX)->GetOriginShape();
    auto cShape = context_->GetOutputShape(X1_IDX)->GetOriginShape();
    size_t aDims = aShape.GetDimNum();
    size_t bDims = bShape.GetDimNum();
    if (context_->GetOptionalInputDesc(BIAS_IDX) != nullptr && context_->GetOptionalInputDesc(OFFSET_IDX) == nullptr) {
        qbmmTilingDataArch20_.withBias = true;
        auto biasShape = context_->GetOptionalInputShape(BIAS_IDX)->GetOriginShape();
        if (biasShape.GetDimNum() == 1){
            qbmmTilingDataArch20_.biasWithBatch = false;
        } else if(biasShape.GetDimNum() == CONST_THREE) {
            qbmmTilingDataArch20_.biasWithBatch = true;
        } else{
            OP_LOGW(context_->GetNodeName(),"Arch20 Pertoken mode bias only support [n] or [b, 1, n]");
        }
    }

    std::vector<int64_t> oriShapeTable;
    if (bDims > 2 && aDims > 2) { // Input dimsize greater than 2
        oriShapeTable = {aShape[aDims - CONST_THREE], aShape[aDims - CONST_TWO], aShape[aDims - CONST_ONE], bShape[bDims - CONST_TWO]}; // NT
    } else {
        oriShapeTable = {1, aShape[aDims - CONST_TWO], aShape[aDims - CONST_ONE], bShape[bDims - CONST_TWO]}; // NT
    }

    qbmmTilingDataArch20_.batchSize = oriShapeTable.at(CONST_ZERO);
    qbmmTilingDataArch20_.m = oriShapeTable.at(CONST_ONE);
    qbmmTilingDataArch20_.k = oriShapeTable.at(CONST_TWO);
    qbmmTilingDataArch20_.n = oriShapeTable.at(CONST_THREE);
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
    Swizzle<QuantMatmulPertokenTilingDataArch20>(qbmmTilingDataArch20_);
    uint64_t TRANS = 1;                // B trans
    uint64_t KERNEL_TEMPLATE_TYPE = 1; // basic
    uint64_t IS_PERTOKEN = 1;          // pertoken
    uint64_t OPTION_ATTRS = 0;         // option_none
    tilingKey_ = GET_TPL_TILING_KEY(TRANS, KERNEL_TEMPLATE_TYPE, IS_PERTOKEN, OPTION_ATTRS);
    ge::graphStatus ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
