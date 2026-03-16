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
 * \file pp_matmul_int8_tiling.cc
 * \brief
 */
#include "common/op_host/op_tiling/tiling_type.h"
#include "op_cache_tiling.h"
#include "log/log.h"
#include "error_util.h"
#include "pp_matmul_int8_tiling.h"
#include "quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_tiling_key.h"

using Ops::NN::MathUtil;

namespace {
constexpr uint64_t KERNEL_TEMPLATE_TYPE_PPMATMUL = 3;
constexpr uint64_t PPMATMUL_PRIORITY_M = 1024;
constexpr uint64_t PPMATMUL_WORKSPACE_SIZE = 24 * 1024 * 1024;
constexpr uint64_t NO_BATCH_DIM_SUM = 2;
}  // namespace

namespace optiling {

PpMatmulInt8Tiling::PpMatmulInt8Tiling(gert::TilingContext *context)
    : QuantBatchMatmulV3TilingBase(context, false),
      tilingData_(tilingDataSelf_)
{
    Reset();
}

void PpMatmulInt8Tiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = PpMatmulTilingData();
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to clear tiling data"), return );
    }
}

ge::graphStatus PpMatmulInt8Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PpMatmulInt8Tiling::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(PpMatmulTilingData);
    return ge::GRAPH_SUCCESS;
}

bool PpMatmulInt8Tiling::IsCapable()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    auto inputAShape = context_->GetInputShape(0)->GetOriginShape();
    uint32_t M = inputAShape.GetDimNum() == NO_BATCH_DIM_SUM ? inputAShape[0] : inputAShape[1];
    auto biasShape = GetBiasShape(GetBiasIdx());
    auto attrs = context_->GetAttrs();
    if (attrs) {
        size_t idx = 0;
        auto dtypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        OP_TILING_CHECK(!dtypePtr,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "There should be at least the required dtype attr."),
                        return false);
        auto transposeX1Ptr = attrs->GetAttrPointer<bool>(idx++);
        auto transposeX2Ptr = attrs->GetAttrPointer<bool>(idx++);
        bool transA = transposeX1Ptr ? *transposeX1Ptr : false;
        bool transB = transposeX2Ptr ? *transposeX2Ptr : false;
        if (socVersion == platform_ascendc::SocVersion::ASCEND310P && M >= PPMATMUL_PRIORITY_M &&
            biasShape != nullptr && *dtypePtr != ge::DT_BF16 && !transA && transB) {
            return true;
        }
    }
    return false;
}

ge::graphStatus PpMatmulInt8Tiling::DoOpTiling()
{
    optiling::transpose_batch_mat_mul::TransposeBatchMatMulEinsumTiling tbmmEinsumTiling(context_, true);
    tbmmEinsumTiling.DoTiling();
    ppMatmulDefaultTilingData_ = tbmmEinsumTiling.ppMatmulDefaultTilingData_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PpMatmulInt8Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}
uint64_t PpMatmulInt8Tiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(1, KERNEL_TEMPLATE_TYPE_PPMATMUL, 0, 0);  // 13
}

ge::graphStatus PpMatmulInt8Tiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<size_t>(PPMATMUL_WORKSPACE_SIZE);  // 24M same as ppmatmul tiling
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PpMatmulInt8Tiling::PostTiling()
{
    tilingData_.batch = ppMatmulDefaultTilingData_.opShape.batchSize;
    tilingData_.m = ppMatmulDefaultTilingData_.opShape.m;
    tilingData_.k = ppMatmulDefaultTilingData_.opShape.k;
    tilingData_.n = ppMatmulDefaultTilingData_.opShape.n;
    tilingData_.m0 = ppMatmulDefaultTilingData_.opShape.m0;
    tilingData_.k0 = ppMatmulDefaultTilingData_.opShape.k0;
    tilingData_.n0 = ppMatmulDefaultTilingData_.opShape.n0;
    tilingData_.mLoop = ppMatmulDefaultTilingData_.mLoop;
    tilingData_.kLoop = ppMatmulDefaultTilingData_.kLoop;
    tilingData_.nLoop = ppMatmulDefaultTilingData_.nLoop;
    tilingData_.coreLoop = ppMatmulDefaultTilingData_.coreLoop;
    tilingData_.swizzleCount = ppMatmulDefaultTilingData_.swizzleCount;
    tilingData_.tilingKey = GetTilingKey();
    tilingData_.blockDim = ppMatmulDefaultTilingData_.blockDim;
    tilingData_.swizzleDirect = ppMatmulDefaultTilingData_.swizzleDirect;
    tilingData_.splitk = ppMatmulDefaultTilingData_.splitk;
    tilingData_.enShuffleK = ppMatmulDefaultTilingData_.enShuffleK;

    OP_TILING_CHECK(
        tilingDataSize_ % sizeof(uint64_t) != 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8.", tilingDataSize_),
        return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(ppMatmulDefaultTilingData_.blockDim);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling