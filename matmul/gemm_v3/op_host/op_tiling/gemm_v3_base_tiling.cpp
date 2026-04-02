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
 * \file gemm_v3_base_tiling.cpp
 * \brief
 */

#include "gemm_v3_base_tiling.h"
#include "../../../mat_mul_v3/op_host/op_tiling/matmul_v3_common.h"
#include "../../../mat_mul_v3/op_kernel/mat_mul_v3_tiling_key.h"
#include "../../op_kernel/gemm_v3_tiling_data.h"
#include "error_util.h"

namespace {
static constexpr uint32_t INDEX_A = 0;
static constexpr uint32_t INDEX_B = 1;
static constexpr uint32_t INDEX_C = 2;
static constexpr uint32_t INDEX_Y = 0;
static constexpr size_t INDEX_ATTR_ALPHA = 0;
static constexpr size_t INDEX_ATTR_BETA = 1;
static constexpr size_t INDEX_ATTR_TRANS_A = 2;
static constexpr size_t INDEX_ATTR_TRANS_B = 3;
static constexpr size_t MIN_NUM_DIM = 2;
static constexpr size_t LAST_DIM_OFFSET = 1;
static constexpr size_t SECOND_LAST_DIM_OFFSET = 2;
static constexpr uint64_t NUM_BUFFER = 2;
} // namespace

namespace optiling {
namespace gemm_v3 {
ge::graphStatus GemmV3BaseTiling::GetInputDims(const gert::Shape& shapeA, const gert::Shape& shapeB)
{
    size_t numDimA = shapeA.GetDimNum();
    size_t numDimB = shapeB.GetDimNum();
    if (numDimA < MIN_NUM_DIM) {
        OP_LOGE(params_.opName, "illegal number of A dim: (%zu)", numDimA);
        return ge::GRAPH_FAILED;
    }
    if (numDimB < MIN_NUM_DIM) {
        OP_LOGE(params_.opName, "illegal number of B dim: (%zu)", numDimB);
        return ge::GRAPH_FAILED;
    }
    if (numDimA != numDimB) {
        OP_LOGE(params_.opName, "A and B dimension mismatch: %zu vs %zu.", numDimA, numDimB);
        return ge::GRAPH_FAILED;
    }

    if (numDimA == MIN_NUM_DIM) {
        numBatchA_ = 1ULL;
        numBatchB_ = 1ULL;
        params_.batchSize = 1ULL;
    } else {
        numBatchA_ = shapeA.GetDim(0);
        numBatchB_ = shapeB.GetDim(0);
        params_.batchSize = std::max(numBatchA_, numBatchB_);
    }

    int64_t ka = shapeA.GetDim(params_.transA != 0 ? numDimA - SECOND_LAST_DIM_OFFSET : numDimA - LAST_DIM_OFFSET);
    int64_t kb = shapeB.GetDim(params_.transB != 0 ? numDimB - LAST_DIM_OFFSET : numDimB - SECOND_LAST_DIM_OFFSET);
    if (ka != kb) {
        OP_LOGE(params_.opName, "unequal input kDim values: k_left[%ld], k_right[%ld]", ka, kb);
        return ge::GRAPH_FAILED;
    }
    params_.k = ka;
    params_.m = shapeA.GetDim(params_.transA != 0 ? numDimA - LAST_DIM_OFFSET : numDimA - SECOND_LAST_DIM_OFFSET);
    params_.n = shapeB.GetDim(params_.transB != 0 ? numDimB - SECOND_LAST_DIM_OFFSET : numDimB - LAST_DIM_OFFSET);
    return ge::GRAPH_SUCCESS;
}

bool GemmV3BaseTiling::IsCapable()
{
    if (hwInfo_.socVersion != platform_ascendc::SocVersion::ASCEND910B &&
        hwInfo_.socVersion != platform_ascendc::SocVersion::ASCEND910_93) {
        return false;
    }
    ge::DataType dtypeC = context_->GetInputDesc(INDEX_C)->GetDataType();
    ge::DataType dtypeY = context_->GetOutputDesc(INDEX_Y)->GetDataType();
    if (dtypeC != dtypeY) {
        OP_LOGW(params_.opName,
                "self dtype (%s) and out dtype (%s) are not the same.",
                Ops::Base::ToString(dtypeC).c_str(),
                Ops::Base::ToString(dtypeY).c_str());
        return false;
    }
    if (dtypeY != ge::DT_FLOAT16 && dtypeY != ge::DT_BF16) {
        OP_LOGW(params_.opName,
                "invalid out dtype (%s), only support half or bfloat16 output.",
                Ops::Base::ToString(dtypeY).c_str());
        return false;
    }
    return true;
}

ge::graphStatus GemmV3BaseTiling::GetPlatformInfo()
{
    if (!context_->GetPlatformInfo()) {
        OP_LOGW(context_->GetNodeName(), "platformInfo is null");
        return ge::GRAPH_FAILED;
    }
    tiling_.GetHardwareInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3BaseTiling::GetShapeAttrsInfo()
{
    if (!InitParams()) {
        return ge::GRAPH_FAILED;
    }
    alpha_ = *(context_->GetAttrs()->GetAttrPointer<float>(INDEX_ATTR_ALPHA));
    beta_ = *(context_->GetAttrs()->GetAttrPointer<float>(INDEX_ATTR_BETA));
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus GemmV3BaseTiling::DoOpTiling()
{
    if (!tiling_.GetMatMulTilingData()) {
        return ge::GRAPH_FAILED;
    }
    using namespace optiling::matmul_v3;
    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint64_t>(TilingEnableFullLoad::BASE),
                                    static_cast<uint64_t>(TilingEnableSplitCore::BASE),
                                    static_cast<uint64_t>(TilingEnableFixOpti::BASE),
                                    static_cast<uint64_t>(MixNd2NzType::NO_ND2NZ),
                                    static_cast<uint64_t>(TilingEnableSpecialOpti::BASE),
                                    static_cast<uint64_t>(TilingEnableFp32Addmm::TRUE));
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus GemmV3BaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t GemmV3BaseTiling::GetTilingKey() const { return tilingKey_; }

ge::graphStatus GemmV3BaseTiling::GetWorkspaceSize()
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspaceSize_ =
        NUM_BUFFER * tilingData_.opShape.m0 * tilingData_.opShape.n0 * tilingData_.blockDim * sizeof(float) +
        platformInfo.GetLibApiWorkSpaceSize();
    OP_LOGD(params_.opName, "workspaceSize: %llu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3BaseTiling::PostTiling()
{
    tiling_.PrintTiling();
    size_t sizeTilingData = sizeof(GemmV3TilingData);
    OP_TILING_CHECK(sizeTilingData % sizeof(uint64_t) != 0,
                    OP_LOGE(params_.opName, "tiling data size[%zu] is not aligned to 8", sizeTilingData),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    context_->GetRawTilingData()->SetDataSize(sizeTilingData);
    context_->SetBlockDim(tilingData_.blockDim);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspaces is null"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    auto tilingPtr = static_cast<GemmV3TilingData*>(context_->GetRawTilingData()->GetData());
    tilingPtr->numBatchA = numBatchA_;
    tilingPtr->numBatchB = numBatchB_;
    tilingPtr->m = tilingData_.opShape.m;
    tilingPtr->k = tilingData_.opShape.k;
    tilingPtr->n = tilingData_.opShape.n;
    tilingPtr->transA = tiling_.matMulInfo_.transA;
    tilingPtr->transB = tiling_.matMulInfo_.transB;
    tilingPtr->m0 = tilingData_.opShape.m0;
    tilingPtr->k0 = tilingData_.opShape.k0;
    tilingPtr->n0 = tilingData_.opShape.n0;
    tilingPtr->mLoop = tilingData_.mLoop;
    tilingPtr->kLoop = tilingData_.kLoop;
    tilingPtr->nLoop = tilingData_.nLoop;
    tilingPtr->coreLoop = tilingData_.coreLoop;
    tilingPtr->swizzleCount = tilingData_.swizzleCount;
    tilingPtr->blockDim = tilingData_.blockDim;
    tilingPtr->swizzleDirect = tilingData_.swizzleDirect;
    tilingPtr->splitk = tilingData_.splitk;
    tilingPtr->enShuffleK = tilingData_.enShuffleK;
    tilingPtr->alpha = alpha_;
    tilingPtr->beta = beta_;
    return ge::GRAPH_SUCCESS;
}

bool GemmV3BaseTiling::InitParams()
{
    params_.opName = context_->GetNodeName();

    params_.formatA = context_->GetInputDesc(INDEX_A)->GetStorageFormat();
    params_.formatB = context_->GetInputDesc(INDEX_B)->GetStorageFormat();

    params_.transA = *(context_->GetAttrs()->GetAttrPointer<bool>(INDEX_ATTR_TRANS_A));
    params_.transB = *(context_->GetAttrs()->GetAttrPointer<bool>(INDEX_ATTR_TRANS_B));

    auto shapeA = context_->GetInputShape(INDEX_A)->GetOriginShape();
    auto shapeB = context_->GetInputShape(INDEX_B)->GetOriginShape();

    OP_LOGD(params_.opName, "shapeA: %s", Ops::Base::ToString(shapeA).c_str());
    OP_LOGD(params_.opName, "shapeB: %s", Ops::Base::ToString(shapeB).c_str());
    OP_LOGD(params_.opName, "params_.transA: %d", params_.transA);
    OP_LOGD(params_.opName, "params_.transB: %d", params_.transB);

    if (GetInputDims(shapeA, shapeB) != ge::GRAPH_SUCCESS) {
        return false;
    }

    OP_LOGD(params_.opName, "params_.m: %d", params_.m);
    OP_LOGD(params_.opName, "params_.k: %d", params_.k);
    OP_LOGD(params_.opName, "params_.n: %d", params_.n);

    params_.dtypeA = context_->GetInputDesc(INDEX_A)->GetDataType();
    params_.dtypeB = context_->GetInputDesc(INDEX_B)->GetDataType();
    params_.sizeInDtype = static_cast<float>(ge::GetSizeByDataType(params_.dtypeA));
    return true;
}
} // namespace gemm_v3
} // namespace optiling
