/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_emu_split_weight_tiling.h"
#include "matmul_emu_split_weight_tiling_key.h"
#include "error_util.h"
#include "matmul/common/op_host/math_util.h"
#include <cmath>

namespace {
static constexpr uint32_t INDEX_X = 0;
static constexpr uint32_t INDEX_W_HIGH = 1;
static constexpr uint32_t INDEX_W_LOW = 2;
static constexpr uint32_t INDEX_Y = 0;
static constexpr uint32_t INDEX_ATTR_W_LOW_SCALE = 0;
static constexpr uint32_t INDEX_ATTR_TRANS_X = 1;
static constexpr uint32_t INDEX_ATTR_TRANS_W = 2;
static constexpr uint32_t INDEX_ATTR_Y_DTYPE = 3;
static constexpr int32_t Y_DTYPE_FP32 = 0;
static constexpr uint32_t NUM_TWO = 2UL;
static constexpr float EXPECTED_SCALE = 0.00390625f;
static constexpr uint64_t GM_STAGE_BUFFER_NUM = 2UL; // Y0 | Y1
static constexpr uint64_t DATA_SIZE_FP32 = 4UL;

static constexpr uint64_t SYS_WORKSPACE_RESERVED = 16UL * 1024UL * 1024UL; // 框架预留16MB
} // namespace

namespace optiling {
namespace matmul_emu_split_weight {

bool MatmulEmuSplitWeightBaseTiling::IsCapable() { return true; }

ge::graphStatus MatmulEmuSplitWeightBaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "platformInfo is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightBaseTiling::ExtractAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "attrs is null"),
                    return ge::GRAPH_FAILED);

    const int32_t* yDtypePtr = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_Y_DTYPE);
    yDtype_ = (yDtypePtr != nullptr) ? *yDtypePtr : Y_DTYPE_FP32;

    scale_ = *(attrs->GetAttrPointer<float>(INDEX_ATTR_W_LOW_SCALE));

    const bool* transXPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X);
    transX_ = (transXPtr != nullptr) ? *transXPtr : false;
    const bool* transWPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_W);
    transW_ = (transWPtr != nullptr) ? *transWPtr : false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightBaseTiling::ExtractShape()
{
    auto xShape = context_->GetInputShape(INDEX_X)->GetOriginShape();
    auto wHighShape = context_->GetInputShape(INDEX_W_HIGH)->GetOriginShape();
    auto wLowShape = context_->GetInputShape(INDEX_W_LOW)->GetOriginShape();

    m_ = transX_ ? static_cast<uint64_t>(xShape.GetDim(1)) : static_cast<uint64_t>(xShape.GetDim(0));
    k_ = transX_ ? static_cast<uint64_t>(xShape.GetDim(0)) : static_cast<uint64_t>(xShape.GetDim(1));
    n_ = transW_ ? static_cast<uint64_t>(wHighShape.GetDim(0)) : static_cast<uint64_t>(wHighShape.GetDim(1));

    wHighK_ = transW_ ? static_cast<uint64_t>(wHighShape.GetDim(1)) : static_cast<uint64_t>(wHighShape.GetDim(0));
    wLowK_ = transW_ ? static_cast<uint64_t>(wLowShape.GetDim(1)) : static_cast<uint64_t>(wLowShape.GetDim(0));
    wLowN_ = transW_ ? static_cast<uint64_t>(wLowShape.GetDim(0)) : static_cast<uint64_t>(wLowShape.GetDim(1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightBaseTiling::ValidateShape()
{
    auto xShape = context_->GetInputShape(INDEX_X)->GetOriginShape();
    auto wHighShape = context_->GetInputShape(INDEX_W_HIGH)->GetOriginShape();
    auto wLowShape = context_->GetInputShape(INDEX_W_LOW)->GetOriginShape();

    OP_TILING_CHECK(
        xShape.GetDimNum() != NUM_TWO || wHighShape.GetDimNum() != NUM_TWO || wLowShape.GetDimNum() != NUM_TWO,
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "all inputs must be 2D"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(m_ == 0 || m_ > static_cast<uint64_t>(INT32_MAX) || n_ == 0 ||
                        n_ > static_cast<uint64_t>(INT32_MAX) || k_ == 0 || k_ > static_cast<uint64_t>(INT32_MAX),
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                                          "m, k, n of x, w must be within the range (0, INT32_MAX], "
                                          "got m=%lu, k=%lu, n=%lu",
                                          m_, k_, n_),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(k_ != wHighK_,
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "x K(%lu) must match w_high K(%lu)", k_, wHighK_),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(k_ != wLowK_,
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "x K(%lu) must match w_low K(%lu)", k_, wLowK_),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(n_ != wLowN_,
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "w_high N(%lu) must match w_low N(%lu)", n_, wLowN_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightBaseTiling::ValidateAttrs()
{
    OP_TILING_CHECK(yDtype_ != Y_DTYPE_FP32,
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "y_dtype only supports 0 (FP32), got %d", yDtype_),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        std::fabs(scale_ - EXPECTED_SCALE) > 1e-7f,
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "w_low_scale only supports 1/256 (0.00390625), got %f", scale_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightBaseTiling::GetShapeAttrsInfo()
{
    if (ExtractAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ExtractShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void MatmulEmuSplitWeightBaseTiling::CalcCoreNum()
{
    uint64_t mTileNum = ops::CeilDiv(m_, baseM_);
    uint64_t nTileNum = ops::CeilDiv(n_, baseN_);
    uint64_t totalTiles = mTileNum * nTileNum;
    usedCoreNum_ = std::min(aicNum_, totalTiles);
    if (usedCoreNum_ == 0) {
        usedCoreNum_ = 1;
    }
}

void MatmulEmuSplitWeightBaseTiling::SetTilingData()
{
    tilingData_.m = static_cast<uint32_t>(m_);
    tilingData_.n = static_cast<uint32_t>(n_);
    tilingData_.k = static_cast<uint32_t>(k_);
    // baseM/baseN/baseK/kL1 are compile-time Catlass L1TileShape/L0TileShape on A2;
    // they are not consumed by the A2 kernel and are left at zero.
    tilingData_.usedCoreNum = static_cast<uint32_t>(usedCoreNum_);
    tilingData_.transX = static_cast<uint8_t>(transX_);
    tilingData_.transW = static_cast<uint8_t>(transW_);
    tilingData_.yDtype = static_cast<uint8_t>(yDtype_);
    tilingData_.scale = scale_;

    MatmulEmuSplitWeightBaseTilingKey tilingKeyObj;
    tilingKeyObj.SetTrans(transX_, transW_);
    // Swizzle direction is baked into the tiling key so the kernel instantiates
    // GemmIdentityBlockSwizzle<3, SwizzleDir> at compile time.
    uint64_t swizzleDir = (m_ > n_) ? 0u : 1u;
    tilingKeyObj.SetSwizzle(swizzleDir);
    tilingKey_ = tilingKeyObj.GetTilingKey();
}

ge::graphStatus MatmulEmuSplitWeightBaseTiling::DoOpTiling()
{
    CalcCoreNum();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightBaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t MatmulEmuSplitWeightBaseTiling::GetTilingKey() const { return tilingKey_; }

ge::graphStatus MatmulEmuSplitWeightBaseTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_RESERVED +
                     GM_STAGE_BUFFER_NUM * static_cast<size_t>(m_) * static_cast<size_t>(n_) * DATA_SIZE_FP32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightBaseTiling::PostTiling()
{
    size_t sizeTilingData = sizeof(MatmulEmuSplitWeightTilingData);
    OP_TILING_CHECK(sizeTilingData % sizeof(uint64_t) != 0,
                    OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", sizeTilingData),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    context_->GetRawTilingData()->SetDataSize(sizeTilingData);
    context_->SetBlockDim(tilingData_.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspaces is null"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    auto tilingPtr = static_cast<MatmulEmuSplitWeightTilingData*>(context_->GetRawTilingData()->GetData());
    *tilingPtr = tilingData_;
    return ge::GRAPH_SUCCESS;
}

} // namespace matmul_emu_split_weight
} // namespace optiling
