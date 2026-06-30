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
 * \file adaptive_sliding_window_basic_api_v4_tiling.cpp
 * \brief
 */

#include "adaptive_sliding_window_basic_api_v4_tiling.h"

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
constexpr uint64_t PER_BLOCK_SIZE = 128;
constexpr size_t DIM_NUM_TWO = 2;

// 控核比例
constexpr uint32_t CORE_RATIO = 2U;
}  // namespace

namespace optiling {

bool AdaptiveSlidingWindowBasicTilingV4::CheckPerTileShape(
    const gert::Shape& x1Shape, const gert::Shape& x2Shape, const gert::Shape& pertokenShape,
    const gert::Shape& scaleShape)
{
    auto biasShape = context_->GetOptionalInputShape(GetBiasIdx());
    inputParams_.hasBias = biasShape != nullptr;

    inputParams_.batchBias = inputParams_.hasBias ? GetBatchSize(biasShape->GetStorageShape()) : 1;
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();

    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);
    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);

    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    AnalyzeBatchInfo(x1Shape, x2Shape);
    if (!InferOutBatchDim(x1Shape, x2Shape)) {
        OP_LOGD(
            inputParams_.opName,
            "batch dim can not be broadcasted or the batch dims of output do not match with input.");
        return false;
    }

    if (!CheckInputValidInPertileMode(scaleShape, pertokenShape, x1Shape, x2Shape)) {
        OP_LOGD(inputParams_.opName, "CheckInputValidInPertileMode failed.");
        return false;
    }
    inputParams_.isPerBlock = true;
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckPertileDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(GetX1Idx())->GetDataType();
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    inputParams_.bDtype = x2Desc->GetDataType();
    auto scaleDesc = context_->GetOptionalInputDesc(GetScaleIdx());
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(GetPertokenIdx());
    if (scaleDesc == nullptr || pertokenScaleDesc == nullptr) {
        OP_LOGD(inputParams_.opName, "x1Scale or x2Scale is nullptr.");
        return false;
    }

    inputParams_.scaleDtype = scaleDesc->GetDataType();
    inputParams_.perTokenScaleDtype = pertokenScaleDesc->GetDataType();
    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : ge::DT_INT32;
    auto x2TableDesc = context_->GetOptionalInputDesc(GetX2TableIdx());
    inputParams_.x2TableDtype = x2TableDesc != nullptr ? x2TableDesc->GetDataType() : inputParams_.x2TableDtype;
    auto outputDesc = context_->GetOutputDesc(0);
    inputParams_.cDtype = outputDesc != nullptr ? outputDesc->GetDataType() : ge::DT_BF16;

    OP_TILING_CHECK(!CheckDtype(), OP_LOGE(inputParams_.opName, "CheckDtype failed!"), return false);

    bool isA8W8GBDtype = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
                         inputParams_.perTokenScaleDtype == ge::DT_FLOAT && inputParams_.scaleDtype == ge::DT_FLOAT &&
                         inputParams_.cDtype == ge::DT_BF16 &&
                         ((inputParams_.hasBias && inputParams_.biasDtype == ge::DT_FLOAT) || !inputParams_.hasBias);
    if (!isA8W8GBDtype) {
        return false;
    }
    return true;
}
bool AdaptiveSlidingWindowBasicTilingV4::IsCapable()
{
    if (context_ == nullptr) {
        OP_LOGE(inputParams_.opName, "context_ is nullptr.");
        return false;
    }
    auto* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (!(ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510)) {
        return false;
    }
    if (!CheckPertileDtype()) {
        return false;
    }

    auto x1Shape = context_->GetInputShape(GetX1Idx());
    auto x2Shape = context_->GetInputShape(GetX2Idx());
    auto x1ScaleShape = context_->GetOptionalInputShape(GetPertokenIdx());
    auto x2ScaleShape = context_->GetOptionalInputShape(GetScaleIdx());
    if (x1ScaleShape == nullptr || x2ScaleShape == nullptr) {
        OP_LOGD(inputParams_.opName, "x1ScaleShape or x2ScaleShape is nullptr.");
        return false;
    }
    auto x1OriginShape = x1Shape->GetOriginShape();
    auto x2OriginShape = x2Shape->GetOriginShape();
    auto x1ScaleOriginShape = x1ScaleShape->GetOriginShape();
    auto x2ScaleOriginShape = x2ScaleShape->GetOriginShape();
    CheckPerTileShape(x1OriginShape, x2OriginShape, x1ScaleOriginShape, x2ScaleOriginShape);
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckDtype() const
{
    return true;
}

ge::graphStatus AdaptiveSlidingWindowBasicTilingV4::CheckContext()
{
    auto x1Shape = context_->GetInputShape(GetX1Idx());
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x2Shape = context_->GetInputShape(GetX2Idx());
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
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
        OP_LOGE(
            inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveSlidingWindowBasicTilingV4::AnalyzeDtype()
{
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::AnalyzeInputs()
{
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckInputValidInPertileMode(
    const gert::Shape& scaleShape, const gert::Shape& pertokenShape, const gert::Shape& x1Shape,
    const gert::Shape& x2Shape) const
{
    auto scaleShapeLen = scaleShape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto pertokenShapeLen = pertokenShape.GetDimNum();
    auto x1ShapeLen = x1Shape.GetDimNum();
    if (!CheckDimValidInPertileMode(x1ShapeLen, x2ShapeLen, pertokenShapeLen, scaleShapeLen)) {
        return false;
    }
    if (!CheckBatchValidInPertileMode(scaleShape, pertokenShape, x1Shape, x2Shape)) {
        return false;
    }

    if (!CheckGroupValidInPertileMode()) {
        return false;
    }

    if (!CheckShapeValidInPertileMode(scaleShape, pertokenShape, x1Shape, x2Shape)) {
        return false;
    }

    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckGroupValidInPertileMode() const
{
    OP_TILING_CHECK(inputParams_.groupSizeM != 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "groupSizeM", std::to_string(inputParams_.groupSizeM).c_str(),
                        "When the quant mode is G-B, the value of groupSizeM must be 1"),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeK != PER_BLOCK_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "groupSizeK", std::to_string(inputParams_.groupSizeK).c_str(),
                        "When the quant mode is G-B, the value of groupSizeK must be 128"),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeN != PER_BLOCK_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "groupSizeN", std::to_string(inputParams_.groupSizeN).c_str(),
                        "When the quant mode is G-B, the value of groupSizeN must be 128"),
                    return false);
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckShapeValidInPertileMode(const gert::Shape& scaleShape,
                                                              const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto  x1ShapeLen = x1Shape.GetDimNum();
    auto  x2ShapeLen = x2Shape.GetDimNum();
    OP_TILING_CHECK(
        (ops::CeilDiv(static_cast<uint64_t>(x2Shape.GetDim(x2ShapeLen - 2)), PER_BLOCK_SIZE) !=
             static_cast<uint64_t>(scaleShape.GetDim(x2ShapeLen - 2)) ||
         ops::CeilDiv(static_cast<uint64_t>(x2Shape.GetDim(x2ShapeLen - 1)), PER_BLOCK_SIZE) !=
             static_cast<uint64_t>(scaleShape.GetDim(x2ShapeLen - 1))),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "scale, x2", "scaleShape[-1], scaleShape[-2]",
            "When the quant mode is G-B, the shape of scale must be equal to ceilDiv(x2Shape, 128)"),
        return false);
    int64_t x1MIndex = inputParams_.transA ? (x1ShapeLen - 1) : (x1ShapeLen - 2);
    int64_t x1KIndex = inputParams_.transA ? (x1ShapeLen - 2) : (x1ShapeLen - 1);
    uint64_t x1M = x1Shape.GetDim(x1MIndex);
    uint64_t scaleX1M = pertoken.GetDim(x1MIndex);
    uint64_t x1K = x1Shape.GetDim(x1KIndex);
    uint64_t scaleX1K = pertoken.GetDim(x1KIndex);
    OP_TILING_CHECK(
        (ops::CeilDiv(x1M, inputParams_.groupSizeM) != scaleX1M),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "x1, pertokenScale", "m dim mismatch",
            "When the quant mode is G-B, the m dim of pertokenScale must be equal to CeilDiv(x1M, groupSizeM)"),
        return false);
    OP_TILING_CHECK(
        (ops::CeilDiv(x1K, inputParams_.groupSizeK) != scaleX1K),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "x1, pertokenScale", "k dim mismatch",
            "When the quant mode is G-B, the k dim of pertokenScale must be equal to CeilDiv(x1K, groupSizeK)"),
        return false);
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckDimValidInPertileMode(size_t x1ShapeLen, size_t x2ShapeLen,
                                                            size_t pertokenShapeLen, size_t scaleShapeLen) const
{
    OP_TILING_CHECK(
        scaleShapeLen != x2ShapeLen,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x2, scale", "dimension count mismatch",
                                                 "The shape dims of x2 and scale must be equal"),
        return false);
    OP_TILING_CHECK(
        pertokenShapeLen != x1ShapeLen,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            inputParams_.opName, "x1, pertokenScale", "dimension count mismatch",
            "The shape dims of x1 and pertokenScale must be equal"),
        return false);
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckBatchValidInPertileMode(const gert::Shape& scaleShape,
                                                              const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto x1ShapeLen = x1Shape.GetDimNum();
    if (x2ShapeLen > DIM_NUM_TWO) {
        for (size_t i = 0; i < x2ShapeLen - DIM_NUM_TWO; ++i) {
            OP_TILING_CHECK(
                scaleShape.GetDim(i) != x2Shape.GetDim(i),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opName, "x2, scale", "batch dim mismatch",
                    "When the quant mode is G-B, the batch dims of x2 and scale must be equal"),
                return false);
        }
    }
    if (x1ShapeLen > DIM_NUM_TWO) {
        for (size_t i = 0; i < x1ShapeLen - DIM_NUM_TWO; ++i) {
            OP_TILING_CHECK(
                pertoken.GetDim(i) != x1Shape.GetDim(i),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opName, "x1, pertokenScale", "batch dim mismatch",
                    "When the quant mode is G-B, the batch dims of x1 and pertokenScale must be equal"),
                return false);
        }
    }
    return true;
}

 bool AdaptiveSlidingWindowBasicTilingV4::SetPlatformInfoForTiling()
 {
     if (!compileInfoInit_) {
         InitCompileInfo();
         auto mmCompileInfo = reinterpret_cast<const QuantBatchMatmulV4CompileInfo *>(context_->GetCompileInfo());
         OP_TILING_CHECK(mmCompileInfo == nullptr,
                         OP_LOGE(inputParams_.opName, "get compile info is null"), return false);
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

bool AdaptiveSlidingWindowBasicTilingV4::CheckCoreNum() const
{
    auto aicNum = compileInfoPtr_->aicNum;
    auto aivNum = compileInfoPtr_->aivNum;
    if (aivNum != CORE_RATIO * aicNum) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "aicNum, aivNum", std::to_string(aicNum) + ", " + std::to_string(aivNum),
            "aicNum:aivNum must be 1:2");
        return false;
    }
    return true;
}

uint64_t AdaptiveSlidingWindowBasicTilingV4::GetTilingKey() const
{
    uint64_t trans = (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
    KernelTemplateType kernelType = isAFullLoad_ ? KernelTemplateType::LUT_AL1FULL : KernelTemplateType::LUT_ASW;
    return GET_TPL_TILING_KEY(
        trans, static_cast<uint64_t>(QuantType::PER_TILE),
        static_cast<uint64_t>(false), static_cast<uint64_t>(false),
        static_cast<uint64_t>(kernelType));
}

}  // namespace optiling
