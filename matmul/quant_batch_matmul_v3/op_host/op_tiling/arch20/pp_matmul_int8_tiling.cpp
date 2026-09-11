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
#include "common/op_host/op_tiling/tiling_type_mm.h"
#include "op_cache_tiling.h"
#include "graph/utils/type_utils.h"
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
constexpr size_t INDEX_ATTR_DTYPE = 0;
constexpr size_t INDEX_ATTR_TRANS_A = 1;
constexpr size_t INDEX_ATTR_TRANS_B = 2;

ge::Format GetPrimaryStorageFormat(const gert::CompileTimeTensorDesc* desc)
{
    if (desc == nullptr) {
        return ge::FORMAT_RESERVED;
    }
    return static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
}
} // namespace

namespace optiling {

PpMatmulInt8Tiling::PpMatmulInt8Tiling(gert::TilingContext* context)
    : QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{
    Reset();
}

void PpMatmulInt8Tiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = PpMatmulTilingData();
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

ge::graphStatus PpMatmulInt8Tiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus PpMatmulInt8Tiling::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(PpMatmulTilingData);
    return ge::GRAPH_SUCCESS;
}

bool PpMatmulInt8Tiling::IsCapable()
{
    const char* opName = context_->GetNodeName();
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGI(opName, "platformInfo is null."), return false);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }

    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto yDesc = context_->GetOutputDesc(0);
    auto x1ShapePtr = context_->GetInputShape(GetX1Idx());
    auto x2ShapePtr = context_->GetInputShape(GetX2Idx());
    OP_TILING_CHECK(
        (x1Desc == nullptr || x2Desc == nullptr || yDesc == nullptr || x1ShapePtr == nullptr || x2ShapePtr == nullptr),
        OP_LOGI(opName, "Input/output desc or shape is null."), return false);

    // Kernel always loads/stores NZ. ACLNN inserts TransData on x1/y; graph mode usually does not.
    OP_TILING_CHECK((GetPrimaryStorageFormat(x1Desc) != ge::FORMAT_FRACTAL_NZ ||
                     GetPrimaryStorageFormat(x2Desc) != ge::FORMAT_FRACTAL_NZ ||
                     GetPrimaryStorageFormat(yDesc) != ge::FORMAT_FRACTAL_NZ),
                    OP_LOGI(opName, "PpMatmul only supports NZ x1/x2/y, fallback to TBE."), return false);
    OP_TILING_CHECK((x1Desc->GetDataType() != ge::DT_INT8 || x2Desc->GetDataType() != ge::DT_INT8 ||
                     yDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGI(opName, "PpMatmul only supports int8 x int8 -> float16, fallback to TBE."), return false);
    OP_TILING_CHECK(context_->GetOptionalInputShape(GetOffsetIdx()) != nullptr,
                    OP_LOGI(opName, "PpMatmul does not support offset, fallback to TBE."), return false);

    const auto& inputAShape = x1ShapePtr->GetOriginShape();
    const auto& inputBShape = x2ShapePtr->GetOriginShape();
    OP_TILING_CHECK((inputAShape.GetDimNum() < NO_BATCH_DIM_SUM || inputBShape.GetDimNum() < NO_BATCH_DIM_SUM),
                    OP_LOGI(opName, "x1/x2 origin shape rank is invalid."), return false);
    uint32_t M = inputAShape.GetDimNum() == NO_BATCH_DIM_SUM ? inputAShape[0] : inputAShape[1];
    uint32_t K = inputAShape.GetDimNum() == NO_BATCH_DIM_SUM ? inputAShape[1] : inputAShape[2];
    // IsCapable requires transB=true, so x2 origin is [N, K] or [B, N, K].
    uint32_t N = inputBShape.GetDimNum() == NO_BATCH_DIM_SUM ? inputBShape[0] : inputBShape[1];
    OP_TILING_CHECK((K == 1 || N == 1), OP_LOGI(opName, "When format of x2 is FRACTAL_NZ, n or k cannot be 1."),
                    return false);

    auto biasShape = GetBiasShape(GetBiasIdx());
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr || biasShape == nullptr || M < PPMATMUL_PRIORITY_M) {
        return false;
    }
    auto dtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DTYPE);
    OP_TILING_CHECK(!dtypePtr, CUBE_INNER_ERR_REPORT(opName, "There should be at least the required dtype attr."),
                    return false);
    auto transposeX1Ptr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_A);
    auto transposeX2Ptr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_B);
    bool transA = transposeX1Ptr ? *transposeX1Ptr : false;
    bool transB = transposeX2Ptr ? *transposeX2Ptr : false;
    if (*dtypePtr != ge::DT_BF16 && !transA && transB) {
        return true;
    }
    return false;
}

ge::graphStatus PpMatmulInt8Tiling::DoOpTiling()
{
    optiling::transpose_batch_mat_mul::TransposeBatchMatMulEinsumTiling tbmmEinsumTiling(context_, true);
    ge::graphStatus ret = tbmmEinsumTiling.DoTiling();
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "PpMatmulInt8 DoTiling failed."), return ret);
    ppMatmulDefaultTilingData_ = tbmmEinsumTiling.ppMatmulDefaultTilingData_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PpMatmulInt8Tiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }
uint64_t PpMatmulInt8Tiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(1, KERNEL_TEMPLATE_TYPE_PPMATMUL, 0, 0); // 13
}

ge::graphStatus PpMatmulInt8Tiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<size_t>(PPMATMUL_WORKSPACE_SIZE); // 24M same as ppmatmul tiling
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
                           static_cast<void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(ppMatmulDefaultTilingData_.blockDim);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
