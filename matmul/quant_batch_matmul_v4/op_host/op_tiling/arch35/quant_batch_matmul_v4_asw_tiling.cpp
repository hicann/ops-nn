/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_asw_tiling.cpp
 * \brief
 */

#include "quant_batch_matmul_v4_asw_tiling.h"

#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_key.h"
#include "error_util.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "op_cache_tiling.h"
#include "op_api/op_util.h"
#include "quant_batch_matmul_v4_checker_for_mmads8s4.h"
#include "quant_batch_matmul_v4_tiling.h"
#include "op_host/tiling_templates_registry.h"

using Ops::NN::MathUtil;

namespace {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t DB_SIZE = 2;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
}  // namespace

namespace optiling {

bool AdaptiveSlidingWindowTilingV4::CheckDtype() const
{
    auto checker = std::unique_ptr<QuantBatchMatmulV4Checker4MmadS8S4>(
        new (std::nothrow) QuantBatchMatmulV4Checker4MmadS8S4(context_, inputParams_));
    OP_TILING_CHECK(checker == nullptr,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to instantiate checker"),
        return false);

    OP_TILING_CHECK(!checker->CheckDtype(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype fail"),
        return false);

    return true;
}

bool AdaptiveSlidingWindowTilingV4::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                                               const gert::StorageShape *biasShape,
                                               const gert::StorageShape *pertokenShape,
                                               const gert::StorageShape *x2TableShape,
                                               const std::vector<int64_t> &dimValueOfMKN) const
{
    if (x2TableShape != nullptr) {
        auto x2TableShapeLen = x2TableShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(x2TableShapeLen != 2,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x2 table shape should be 2 dim"), return false);
        // the x2Table is transposed
        inputParams_.x2TableNSize =
            static_cast<uint64_t>(x2TableShape->GetStorageShape().GetDim(x2TableShapeLen - LAST_SECOND_DIM_INDEX));
        inputParams_.x2TableKSize =
            static_cast<uint64_t>(x2TableShape->GetStorageShape().GetDim(x2TableShapeLen - LAST_FIRST_DIM_INDEX));
    }
    auto checker = std::unique_ptr<QuantBatchMatmulV4Checker4MmadS8S4>(
        new (std::nothrow) QuantBatchMatmulV4Checker4MmadS8S4(context_, inputParams_));
    OP_TILING_CHECK(checker == nullptr,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to instantiate checker"),
        return false);
    OP_TILING_CHECK(!checker->CheckShape(mandtoryShape, biasShape, pertokenShape, dimValueOfMKN),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckShape fail"),
        return false);

    return true;
}

ge::graphStatus AdaptiveSlidingWindowTilingV4::CheckContext()
{
    auto x1Shape = context_->GetInputShape(GetX1Idx());
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x2Shape = context_->GetInputShape(GetX2Idx());
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Function context_->GetAttrs() failed!"),
                    return ge::GRAPH_FAILED);
    auto dtypeAttr = attrs->GetAttrPointer<int64_t>(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dtypeAttr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                              context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveSlidingWindowTilingV4::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(GetX1Idx())->GetDataType();
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    inputParams_.bDtype = x2Desc->GetDataType();
    auto scaleDesc = context_->GetOptionalInputDesc(GetScaleIdx());
    inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(GetPertokenIdx());
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : ge::DT_INT32;
    auto x2TableDesc = context_->GetOptionalInputDesc(GetX2TableIdx());
    inputParams_.x2TableDtype = x2TableDesc != nullptr ? x2TableDesc->GetDataType() : inputParams_.x2TableDtype;

    // 当前AdaptiveSlidingWindowTilingV4仅支持LUT场景，x2Table必须存在
    OP_TILING_CHECK(
        x2TableDesc == nullptr, CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2Table does not exist."), return false);
    inputParams_.isLut = true;

    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    isUbQuant_ = inputParams_.cDtype == ge::DT_BF16 || pertokenScaleDesc != nullptr;
    SetFormat();

    OP_TILING_CHECK(!CheckDtype(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype failed!"), return false);
    return true;
}

bool AdaptiveSlidingWindowTilingV4::AnalyzeInputs()
{
    auto x1Shape = context_->GetInputShape(GetX1Idx())->GetOriginShape();
    auto x2Shape = context_->GetInputShape(GetX2Idx())->GetOriginShape();
    auto scaleShape = context_->GetOptionalInputShape(GetScaleIdx());
    auto pertokenShape = context_->GetOptionalInputShape(GetPertokenIdx());
    inputParams_.isPertoken = pertokenShape != nullptr;
    auto biasShape = context_->GetOptionalInputShape(GetBiasIdx());
    inputParams_.hasBias = biasShape != nullptr;
    inputParams_.batchBias = inputParams_.hasBias ? GetBatchSize(biasShape->GetStorageShape()) : 1;
    auto x2TableShape = context_->GetOptionalInputShape(GetX2TableIdx());
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    OP_TILING_CHECK(x1ShapeLen != 2,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Input x1 dimension should equal to 2, but x1 dimension: %zu.", x1ShapeLen),
                    return false);
    OP_TILING_CHECK(x2ShapeLen != 2,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Input x2 dimension should equal to 2, but x2 dimension: %zu.", x2ShapeLen),
                    return false);

    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);
    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);

    const std::vector<gert::Shape *> mandtoryShape = {&x1Shape, &x2Shape};

    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    AnalyzeBatchInfo(x1Shape, x2Shape);
    OP_TILING_CHECK(
        !InferOutBatchDim(x1Shape, x2Shape),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "batch dim can not be broadcasted or the batch dims of output do not match with input."),
        return false);
    if (scaleShape != nullptr && !SetQuantMode(scaleShape->GetStorageShape(), pertokenShape)) {
        return false;
    }
    if (!CheckShape(mandtoryShape, biasShape, pertokenShape, x2TableShape, dimValueOfMKN)) {
        return false;
    }
    OP_TILING_CHECK(!CheckOutputShapeAvailable(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Multiple of output shape dims should be in boundary of INT64_MAX"),
                                          return false);

    auto isPerTensorStr = inputParams_.isPerTensor ? "true" : "false";
    auto isPertokenStr = inputParams_.isPertoken ? "true" : "false";
    OP_LOGD(inputParams_.opName, "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %s, isPertoken: %s",
            inputParams_.batchA, inputParams_.batchB, inputParams_.batchC, isPerTensorStr, isPertokenStr);
    return true;
}

bool AdaptiveSlidingWindowTilingV4::CalcBasicBlock()
{
    adaptiveWin_.baseM = std::min(inputParams_.mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    adaptiveWin_.baseM =
        !inputParams_.transA
            ? ops::CeilAlign(adaptiveWin_.baseM, CUBE_BLOCK)
            : ops::CeilAlign(adaptiveWin_.baseM, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype));
    // lut场景，baseN = groupN, baseK = groupK, 但当shape k/n较小时，baseN baseK也需要后续进行对齐调整
    adaptiveWin_.baseN = std::min(inputParams_.nSize, inputParams_.groupSizeN);
    adaptiveWin_.baseK = std::min(inputParams_.kSize, inputParams_.groupSizeK);
    adaptiveWin_.baseN =
        inputParams_.transB
            ? ops::CeilAlign(adaptiveWin_.baseN, CUBE_BLOCK)
            : ops::CeilAlign(adaptiveWin_.baseN,
                             GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype, inputParams_.isLut));
    uint64_t maxAlignSize = std::max(
        static_cast<uint64_t>(GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype)),
        static_cast<uint64_t>(GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.bDtype, inputParams_.isLut)));
    adaptiveWin_.baseK = ops::CeilAlign(adaptiveWin_.baseK, maxAlignSize);
    return true;
}

bool AdaptiveSlidingWindowTilingV4::SetPlatformInfoForTiling()
{
    // mc2和qbmm把compileInfo都赋值给compileInfoPtr_，后续硬件信息可以直接从compileInfoPtr_中获取
    if (!compileInfoInit_) {
        // 初始化qbmmv3的compileInfo
        InitCompileInfo();
        auto mmCompileInfo = reinterpret_cast<const QuantBatchMatmulV4CompileInfo *>(context_->GetCompileInfo());
        OP_TILING_CHECK(mmCompileInfo == nullptr,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "get compile info is null"), return false);
        try {
            compileInfoPtr_ = std::make_unique<QuantBatchMatmulV4CompileInfo>(*mmCompileInfo);
        } catch (const std::bad_alloc &e) {
            OP_LOGE(inputParams_.opName, "failed to instantiate compile info");
            return false;
        }
    }
    OP_LOGE_IF(compileInfoPtr_->aicNum <= 0, false, inputParams_.opName, "aicNum <= 0");
    aicoreParams_.aicNum = compileInfoPtr_->aicNum;
    inputParams_.libApiWorkSpaceSize = compileInfoPtr_->workspaceNum;
    aicoreParams_.ubSize = compileInfoPtr_->ubSize;
    aicoreParams_.l1Size = compileInfoPtr_->l1Size;
    aicoreParams_.l0aSize = compileInfoPtr_->l0aSize;
    aicoreParams_.l0cSize = compileInfoPtr_->l0cSize;
    aicoreParams_.blockDim = 0;
    return true;
}

void AdaptiveSlidingWindowTilingV4::IsABFullLoad()
{
    // supportMmadS8S4平台 LUT 场景不支持AB全载
    isABFullLoad_ = false;
    return;
}

void AdaptiveSlidingWindowTilingV4::IsBFullLoad()
{
    // supportMmadS8S4平台 LUT 场景不支持B全载
    isBFullLoad_ = false;
    return;
}

void AdaptiveSlidingWindowTilingV4::CalcTailBasicBlockAfullLoad()
{
    // supportMmadS8S4平台 LUT AL1全载场景不会对N方向做尾块从切分
    return;
}

void AdaptiveSlidingWindowTilingV4::CalcTailBasicBlock()
{
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    // In the A8W1/A8W2/A8W4 lut scenario, only the baseM direction is eligible for splitting.
    while (CalUsedCoreNum(mTile + 1UL, 1UL) <= aicoreParams_.aicNum) {
        mTile += 1UL;
    }
    adaptiveWin_.mTailTile = mTile;
    adaptiveWin_.nTailTile = nTile;
}

bool AdaptiveSlidingWindowTilingV4::IsCalL1TilingDepth4MmadS8S4() const
{
    // LUT场景 L1 depth计算逻辑由CalL1TilingDepth4MmadS8S4函数确定
    return true;
}

void AdaptiveSlidingWindowTilingV4::CalL1TilingDepth4MmadS8S4(uint64_t leftL1Size)
{
    // LUT 场景采用mm api Norm模板，stepK无意义，默认1
    basicTiling_.stepKa = 1U;
    basicTiling_.stepKb = 1U;

    // LUT 场景采用mm api Norm模板，depthA1 depthB1尽可能用满L1空间。分asw和al1full两种情况讨论
    basicTiling_.depthA1 = 1U;
    basicTiling_.depthB1 = 1U;

    uint64_t maxDepth = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));

    uint64_t oneBaseADataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t oneBaseBDataSize = GetSizeWithDataType(
        static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype, inputParams_.isLut);

    if (isAFullLoad_) {
        basicTiling_.depthB1 =
            std::min(ops::FloorDiv(leftL1Size - singleCoreASizeWithFullLoad_, oneBaseBDataSize), maxDepth);
    } else {
        basicTiling_.depthA1 = std::min(ops::FloorDiv(leftL1Size, oneBaseADataSize + oneBaseBDataSize), maxDepth);
        basicTiling_.depthB1 = basicTiling_.depthA1;
    }
}
bool AdaptiveSlidingWindowTilingV4::Is4BitInput(ge::DataType dtype, bool isLut) const
{
    bool is4BitInput = false;
    if (isLut) {
        // lut查表逻辑: 原始数据DT_INT2和DT_UINT1，查表后转DT_INT4; 原始数据DT_INT4, 查表后转DT_INT8
        is4BitInput = (dtype == ge::DT_INT2 || dtype == ge::DT_UINT1);
    } else {
        is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_INT4);
    }
    return is4BitInput;
}

bool AdaptiveSlidingWindowTilingV4::Is8BitInput(ge::DataType dtype, bool isLut) const
{
    // lut查表逻辑: 原始数据DT_INT2和DT_UINT1，查表后转DT_INT4; 原始数据DT_INT4, 查表后转DT_INT8
    return (isLut == true && dtype == ge::DT_INT4);
}

uint64_t AdaptiveSlidingWindowTilingV4::GetShapeWithDataType(uint64_t size, ge::DataType dtype, bool isLut) const
{
    if (Is4BitInput(dtype, isLut)) {
        return size + size;
    } else if (Is8BitInput(dtype, isLut)) {
        return size;
    } else {
        return size / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t AdaptiveSlidingWindowTilingV4::GetSizeWithDataType(uint64_t shape, ge::DataType dtype, bool isLut) const
{
    if (Is4BitInput(dtype, isLut)) {
        return (shape + 1) >> 1;
    } else if (Is8BitInput(dtype, isLut)) {
        return shape;
    } else {
        return shape * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t AdaptiveSlidingWindowTilingV4::GetTilingKey() const
{
    uint64_t trans = (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
    KernelTemplateType kernelType = isAFullLoad_ ? KernelTemplateType::LUT_AL1FULL : KernelTemplateType::LUT_ASW;
    return GET_TPL_TILING_KEY(
        trans, static_cast<uint64_t>(QuantType::NONE),
        static_cast<uint64_t>(false), static_cast<uint64_t>(true),
        static_cast<uint64_t>(kernelType));
}

bool AdaptiveSlidingWindowTilingV4::CheckCoreNum() const
{
    // 预留校验接口
    return true;
}

}  // namespace optiling