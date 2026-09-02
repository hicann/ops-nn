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
#include "matmul/common/op_host/math_util_nn.h"
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
static constexpr uint64_t DATA_SIZE_BF16 = 2UL;
static constexpr uint64_t DATA_SIZE_FP32 = 4UL;
static constexpr uint64_t DB_SIZE = 2UL;
static constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
static constexpr uint64_t HARDCODED_BASE_M = 256UL;
static constexpr uint64_t HARDCODED_BASE_N = 112;
static constexpr uint64_t NUM_TWO = 2UL;
static constexpr uint64_t NUM_FOUR = 4UL;
static constexpr float EXPECTED_SCALE = 0.00390625f;
} // namespace

namespace optiling {
namespace matmul_emu_split_weight {

bool MatmulEmuSplitWeightTiling::IsCapable()
{
    if (aicNum_ == 0 || aivNum_ != aicNum_ * 2) {
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                              "MatmulEmuSplitWeight is only supported for aicNum:aivNum=1:2, aicNum=%lu, aivNum=%lu",
                              aicNum_, aivNum_);
        return false;
    }
    return true;
}

ge::graphStatus MatmulEmuSplitWeightTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "platformInfo is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    OP_TILING_CHECK(socVersion != platform_ascendc::SocVersion::ASCEND950,
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                                          "MatmulEmuSplitWeight only supports Ascend950, current socVersion is %d",
                                          static_cast<int32_t>(socVersion)),
                    return ge::GRAPH_FAILED);
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    return ge::GRAPH_SUCCESS;
}

// ====== Extract phases ======

ge::graphStatus MatmulEmuSplitWeightTiling::ExtractAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "attrs is null"),
                    return ge::GRAPH_FAILED);

    const int32_t* yDtypePtr = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_Y_DTYPE);
    yDtype_ = (yDtypePtr != nullptr) ? *yDtypePtr : Y_DTYPE_FP32;

    const float* scalePtr = attrs->GetAttrPointer<float>(INDEX_ATTR_W_LOW_SCALE);
    OP_TILING_CHECK(scalePtr == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "wLowScale attr is null"),
                    return ge::GRAPH_FAILED);
    scale_ = *scalePtr;

    const bool* transXPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X);
    transX_ = (transXPtr != nullptr) ? *transXPtr : false;
    const bool* transWPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_W);
    transW_ = (transWPtr != nullptr) ? *transWPtr : false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightTiling::ExtractShape()
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

// ====== Validate phases ======

ge::graphStatus MatmulEmuSplitWeightTiling::ValidateDtype() const
{
    ge::DataType dtypeX = context_->GetInputDesc(INDEX_X)->GetDataType();
    ge::DataType dtypeWHigh = context_->GetInputDesc(INDEX_W_HIGH)->GetDataType();
    ge::DataType dtypeWLow = context_->GetInputDesc(INDEX_W_LOW)->GetDataType();
    ge::DataType dtypeY = context_->GetOutputDesc(INDEX_Y)->GetDataType();
    ge::DataType expectedOutDtype = ge::DT_FLOAT;
    OP_TILING_CHECK(
        dtypeX != ge::DT_BF16 || dtypeWHigh != ge::DT_BF16 || dtypeWLow != ge::DT_BF16 || dtypeY != expectedOutDtype,
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                              "MatmulEmuSplitWeight only supports BF16 input and FP32 output, "
                              "got x=%d, w_high=%d, w_low=%d, y=%d",
                              static_cast<int32_t>(dtypeX), static_cast<int32_t>(dtypeWHigh),
                              static_cast<int32_t>(dtypeWLow), static_cast<int32_t>(dtypeY)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightTiling::ValidateFormat() const
{
    auto formatX = ge::GetPrimaryFormat(context_->GetInputDesc(INDEX_X)->GetStorageFormat());
    auto formatWHigh = ge::GetPrimaryFormat(context_->GetInputDesc(INDEX_W_HIGH)->GetStorageFormat());
    auto formatWLow = ge::GetPrimaryFormat(context_->GetInputDesc(INDEX_W_LOW)->GetStorageFormat());
    auto formatY = ge::GetPrimaryFormat(context_->GetOutputDesc(INDEX_Y)->GetStorageFormat());
    OP_TILING_CHECK(
        formatX != ge::FORMAT_ND || formatWHigh != ge::FORMAT_ND || formatWLow != ge::FORMAT_ND ||
            formatY != ge::FORMAT_ND,
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                              "MatmulEmuSplitWeight only supports ND format, got x=%d, w_high=%d, w_low=%d, y=%d",
                              static_cast<int32_t>(formatX), static_cast<int32_t>(formatWHigh),
                              static_cast<int32_t>(formatWLow), static_cast<int32_t>(formatY)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightTiling::ValidateShape() const
{
    auto xShape = context_->GetInputShape(INDEX_X)->GetOriginShape();
    auto wHighShape = context_->GetInputShape(INDEX_W_HIGH)->GetOriginShape();
    auto wLowShape = context_->GetInputShape(INDEX_W_LOW)->GetOriginShape();
    // 仅支持2维
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

ge::graphStatus MatmulEmuSplitWeightTiling::ValidateAttrs() const
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

// ====== Input null-check ======

ge::graphStatus MatmulEmuSplitWeightTiling::ValidateInputsNotNull()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INDEX_X));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INDEX_X));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INDEX_W_HIGH));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INDEX_W_HIGH));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INDEX_W_LOW));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INDEX_W_LOW));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(INDEX_Y));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(INDEX_Y));
    return ge::GRAPH_SUCCESS;
}

// ====== GetShapeAttrsInfo: orchestrates extract + validate phases ======

ge::graphStatus MatmulEmuSplitWeightTiling::GetShapeAttrsInfo()
{
    if (ValidateInputsNotNull() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ExtractAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ExtractShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateFormat() != ge::GRAPH_SUCCESS) {
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

// ====== Tiling calculation phases ======

void MatmulEmuSplitWeightTiling::CalcBaseMN()
{
    uint64_t mAlign = ops::CeilAlign(m_, BASIC_BLOCK_SIZE_16);
    uint64_t nAlign = ops::CeilAlign(n_, BASIC_BLOCK_SIZE_16);

    baseM_ = std::min(mAlign, HARDCODED_BASE_M);
    baseN_ = std::min(nAlign, HARDCODED_BASE_N);

    uint64_t mTileNum = ops::CeilDiv(m_, baseM_);
    uint64_t nTileNum = ops::CeilDiv(n_, baseN_);
    if (mTileNum * nTileNum < aicNum_) {
        CalcBasicBlock();
    }
}

void MatmulEmuSplitWeightTiling::CalcBasicBlock()
{
    uint64_t mCore = ops::CeilDiv(m_, baseM_);
    uint64_t nCore = ops::CeilDiv(n_, baseN_);
    if (mCore == 0UL || nCore == 0UL) {
        return;
    }
    if (mCore <= nCore) {
        baseM_ = ops::CeilAlign(ops::CeilDiv(m_, aicNum_ / nCore), BASIC_BLOCK_SIZE_16);
        mCore = ops::CeilDiv(m_, baseM_);
        nCore = aicNum_ / mCore;
        baseN_ = ops::CeilAlign(ops::CeilDiv(n_, nCore), BASIC_BLOCK_SIZE_16);
    } else {
        baseN_ = ops::CeilAlign(ops::CeilDiv(n_, aicNum_ / mCore), BASIC_BLOCK_SIZE_16);
        nCore = ops::CeilDiv(n_, baseN_);
        mCore = aicNum_ / nCore;
        baseM_ = ops::CeilAlign(ops::CeilDiv(m_, mCore), BASIC_BLOCK_SIZE_16);
    }
    mCore = ops::CeilDiv(m_, baseM_);
    nCore = ops::CeilDiv(n_, baseN_);

    // 循环平衡baseM/baseN
    while (baseN_ >= baseM_ * NUM_TWO && nCore < aicNum_ / NUM_TWO) {
        nCore = nCore * NUM_TWO;
        mCore = aicNum_ / nCore;
        baseM_ = ops::CeilAlign(ops::CeilDiv(m_, mCore), BASIC_BLOCK_SIZE_16);
        baseN_ = ops::CeilAlign(ops::CeilDiv(n_, nCore), BASIC_BLOCK_SIZE_16);
        mCore = ops::CeilDiv(m_, baseM_);
        nCore = ops::CeilDiv(n_, baseN_);
    }

    while (baseM_ >= baseN_ * NUM_TWO && mCore < aicNum_ / NUM_TWO) {
        mCore = mCore * NUM_TWO;
        nCore = aicNum_ / mCore;
        baseM_ = ops::CeilAlign(ops::CeilDiv(m_, mCore), BASIC_BLOCK_SIZE_16);
        baseN_ = ops::CeilAlign(ops::CeilDiv(n_, nCore), BASIC_BLOCK_SIZE_16);
        mCore = ops::CeilDiv(m_, baseM_);
        nCore = ops::CeilDiv(n_, baseN_);
    }
}

void MatmulEmuSplitWeightTiling::CalcBaseK()
{
    uint64_t kAlign = ops::CeilAlign(k_, BASIC_BLOCK_SIZE_16);
    uint64_t l0aBudget = l0aSize_ / DB_SIZE / DATA_SIZE_BF16;
    uint64_t baseKMaxL0A = ops::FloorAlign(l0aBudget / baseM_, BASIC_BLOCK_SIZE_16);
    uint64_t l0bBudget = l0bSize_ / DB_SIZE / DATA_SIZE_BF16;
    uint64_t baseKMaxL0B = ops::FloorAlign(l0bBudget / (NUM_TWO * baseN_), BASIC_BLOCK_SIZE_16);
    baseK_ = std::min({kAlign, baseKMaxL0A, baseKMaxL0B});
}

void MatmulEmuSplitWeightTiling::CalcKL1()
{
    uint64_t kAlign = ops::CeilAlign(k_, BASIC_BLOCK_SIZE_16);
    uint64_t aL1OneBuffer = baseM_ * baseK_ * DATA_SIZE_BF16;
    uint64_t bL1OneBuffer = baseN_ * baseK_ * DATA_SIZE_BF16;
    uint64_t totalL1 = NUM_TWO * aL1OneBuffer + NUM_FOUR * bL1OneBuffer;
    uint64_t stepK = l1Size_ / totalL1;
    if (stepK == 0) {
        stepK = 1;
    }
    kL1_ = std::min(kAlign, baseK_ * stepK);
    kL1_ = std::max(kL1_, baseK_);
    kL1_ = ops::FloorAlign(kL1_, BASIC_BLOCK_SIZE_16);
    if (kL1_ == 0) {
        kL1_ = baseK_;
    }
}

void MatmulEmuSplitWeightTiling::CalcCoreNum()
{
    uint64_t mTileNum = ops::CeilDiv(m_, baseM_);
    uint64_t nTileNum = ops::CeilDiv(n_, baseN_);
    uint64_t totalTiles = mTileNum * nTileNum;
    usedCoreNum_ = std::min(aicNum_, totalTiles);
    if (usedCoreNum_ == 0) {
        usedCoreNum_ = 1;
    }
}

void MatmulEmuSplitWeightTiling::SetTilingData()
{
    tilingData_.m = static_cast<uint32_t>(m_);
    tilingData_.n = static_cast<uint32_t>(n_);
    tilingData_.k = static_cast<uint32_t>(k_);
    tilingData_.baseM = static_cast<uint32_t>(baseM_);
    tilingData_.baseN = static_cast<uint32_t>(baseN_);
    tilingData_.baseK = static_cast<uint32_t>(baseK_);
    tilingData_.kL1 = static_cast<uint32_t>(kL1_);
    tilingData_.usedCoreNum = static_cast<uint32_t>(usedCoreNum_);
    tilingData_.transX = static_cast<uint8_t>(transX_);
    tilingData_.transW = static_cast<uint8_t>(transW_);
    tilingData_.yDtype = static_cast<uint8_t>(yDtype_);
    tilingData_.scale = scale_;

    MatmulEmuSplitWeightTilingKey tilingKeyObj;
    tilingKeyObj.SetTrans(transX_, transW_);
    tilingKey_ = tilingKeyObj.GetTilingKey();
}

// ====== DoOpTiling: orchestrates all tiling calculation phases ======

ge::graphStatus MatmulEmuSplitWeightTiling::DoOpTiling()
{
    CalcBaseMN();
    CalcBaseK();
    CalcKL1();
    CalcCoreNum();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulEmuSplitWeightTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t MatmulEmuSplitWeightTiling::GetTilingKey() const { return tilingKey_; }

ge::graphStatus MatmulEmuSplitWeightTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MatmulEmuSplitWeightTiling::PostTiling()
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
    workspaces[0] = 0;

    auto tilingPtr = static_cast<MatmulEmuSplitWeightTilingData*>(context_->GetRawTilingData()->GetData());
    *tilingPtr = tilingData_;
    return ge::GRAPH_SUCCESS;
}

} // namespace matmul_emu_split_weight
} // namespace optiling
