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
#include "common/inc/error_util.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "op_cache_tiling.h"
#include "op_util.h"
#include "quant_batch_matmul_v4_checker_for_mmads8s4.h"
#include "quant_batch_matmul_v4_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

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
}  // namespace

namespace optiling {

bool AdaptiveSlidingWindowBasicTilingV4::IsCapable()
{
    if (context_ == nullptr) {
        OP_LOGE(inputParams_.opName, "context_ is nullptr.");
        return false;
    }
    platform_ascendc::SocVersion socVersion;
    auto* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    
    bool isDavid = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510;
    bool isA8W8GBDtype = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
                    inputParams_.perTokenScaleDtype == ge::DT_FLOAT && 
                    inputParams_.scaleDtype == ge::DT_FLOAT && inputParams_.cDtype == ge::DT_BF16 && 
                    ((inputParams_.hasBias && inputParams_.biasDtype == ge::DT_FLOAT) || !inputParams_.hasBias);

    auto x1Shape = context_->GetInputShape(GetX1Idx());
    auto x2Shape = context_->GetInputShape(GetX2Idx());
    auto x1ScaleShape = context_->GetOptionalInputShape(GetPertokenIdx());
    auto x2ScaleShape = context_->GetOptionalInputShape(GetScaleIdx());
    if (x1Shape == nullptr || x2Shape == nullptr || x1ScaleShape == nullptr || x2ScaleShape == nullptr) {
        OP_LOGE(inputParams_.opName, "In G-B quantification, x1Shape, x2Shape, x1ScaleShape and "
        "x2ScaleShape can not be nullptr.");
        return false;
    }
    auto x1OriginShape = x1Shape->GetOriginShape();
    auto x2OriginShape = x2Shape->GetOriginShape();
    auto x1ScaleOriginShape = x1ScaleShape->GetOriginShape();
    auto x2ScaleOriginShape = x2ScaleShape->GetOriginShape();

    bool isA8W8GBDim = x1OriginShape.GetDimNum() == x1ScaleOriginShape.GetDimNum() &&
                    x2OriginShape.GetDimNum() == x2ScaleOriginShape.GetDimNum();

    return isDavid && isA8W8GBDtype && isA8W8GBDim;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckDtype() const
{
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "Expected dtype of X1 and X2 to be DT_INT8, but actual X1 dtype is %s and X1 dtype is %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);
    OP_TILING_CHECK(inputParams_.perTokenScaleDtype != ge::DT_FLOAT || inputParams_.scaleDtype != ge::DT_FLOAT,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "Expected dtype of X1Scale and X2Scale to be DT_FLOAT, but actual X1Scale dtype is %s "
            "and X2Scale dtype is %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
        return false);
        auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    OP_TILING_CHECK(biasDesc != nullptr && inputParams_.biasDtype != ge::DT_FLOAT,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "Expected dtype of bias to be DT_FLOAT, but actual is %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
        return false);
    OP_TILING_CHECK(inputParams_.cDtype != ge::DT_BF16,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "Expected dtype of output to be DT_BF16, but actual is %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
        return false);
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

bool AdaptiveSlidingWindowBasicTilingV4::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(GetX1Idx())->GetDataType();
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    inputParams_.bDtype = x2Desc->GetDataType();
    auto scaleDesc = context_->GetOptionalInputDesc(GetScaleIdx());
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(GetPertokenIdx());
    OP_TILING_CHECK(pertokenScaleDesc == nullptr || scaleDesc == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "In G-B quantification, "
                    "X1Scale and X2Scale can not be nullptr."),
                    return false);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    inputParams_.perTokenScaleDtype = pertokenScaleDesc->GetDataType();
    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : ge::DT_INT32;
    auto x2TableDesc = context_->GetOptionalInputDesc(GetX2TableIdx());
    inputParams_.x2TableDtype = x2TableDesc != nullptr ? x2TableDesc->GetDataType() : inputParams_.x2TableDtype;
    auto outputDesc = context_->GetOutputDesc(0);
    inputParams_.cDtype = outputDesc != nullptr ? outputDesc->GetDataType() : ge::DT_BF16;

    OP_TILING_CHECK(!CheckDtype(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype failed!"), return false);
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::AnalyzeInputs()
{
    auto x1Shape = context_->GetInputShape(GetX1Idx())->GetOriginShape();
    auto x2Shape = context_->GetInputShape(GetX2Idx())->GetOriginShape();
    const gert::Shape& scaleShape = context_->GetOptionalInputShape(GetScaleIdx())->GetStorageShape();
    auto pertokenShape = context_->GetOptionalInputShape(GetPertokenIdx());
    inputParams_.isPertoken = pertokenShape != nullptr;
    auto biasShape = context_->GetOptionalInputShape(GetBiasIdx());
    inputParams_.hasBias = biasShape != nullptr;
    inputParams_.batchBias = inputParams_.hasBias ? GetBatchSize(biasShape->GetStorageShape()) : 1;
    auto x2TableShape = context_->GetOptionalInputShape(GetX2TableIdx());
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
    OP_TILING_CHECK(
        !InferOutBatchDim(x1Shape, x2Shape),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "batch dim can not be broadcasted or the batch dims of output do not match with input."),
        return false);

    OP_TILING_CHECK(!CheckInputValidInPerblockMode(scaleShape, pertokenShape, x1Shape, x2Shape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckInputValidInPerblockMode failed."),
                    return false); 
    inputParams_.isPerBlock = true;

    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckInputValidInPerblockMode(const gert::Shape& scaleShape,
                                                              const gert::StorageShape *pertokenShape,
                                                              const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto scaleShapeLen = scaleShape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto &pertoken = pertokenShape->GetStorageShape();
    auto pertokenShapeLen = pertoken.GetDimNum();
    auto x1ShapeLen = x1Shape.GetDimNum();
    if (!CheckDimValidInPerblockMode(x1ShapeLen, x2ShapeLen, pertokenShapeLen, scaleShapeLen)) {
        return false;
    }
    if (!CheckBatchValidInPerblockMode(scaleShape, pertoken, x1Shape, x2Shape)) {
        return false;
    }

    if (!CheckGroupValidInPerblockMode()) {
        return false;
    }

    if (!CheckShapeValidInPerblockMode(scaleShape, pertoken, x1Shape, x2Shape)) {
        return false;
    }

    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckGroupValidInPerblockMode() const
{
    OP_TILING_CHECK(inputParams_.groupSizeM != 1,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "In G-B quantification, input or infered groupSizeM should be 1, but now is %lu, \
groupSizeM = (groupSize >> 32) & 0xFFFF.",
                                          inputParams_.groupSizeM),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeK != PER_BLOCK_SIZE,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "In G-B quantification, input or infered groupSizeK should be 128, but now is %lu, \
groupSizeK = groupSize & 0xFFFF.",
                                          inputParams_.groupSizeK),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeN != PER_BLOCK_SIZE,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "In G-B quantification, input or infered groupSizeN should be 128, but now is %lu, \
groupSizeN = (groupSize >> 16) & 0xFFFF.",
                                          inputParams_.groupSizeN),
                    return false);
  return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckShapeValidInPerblockMode(const gert::Shape& scaleShape,
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
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "In G-B quantification, the size of last two dimensions of scale should be both equal to \
the size of last two dimensions of x2 ceildivided by groupSize 128, but now, \
scaleShape[-1] is %ld, x2Shape[-1] is %ld, scaleShape[-2] is %ld, x2Shape[-2] is %ld.",
            scaleShape.GetDim(x2ShapeLen - 1), x2Shape.GetDim(x2ShapeLen - 1),
            scaleShape.GetDim(x2ShapeLen - 2), x2Shape.GetDim(x2ShapeLen - 2)),
        return false);
    int64_t x1MIndex = inputParams_.transA ? (x1ShapeLen - 1) : (x1ShapeLen - 2);
    int64_t x1KIndex = inputParams_.transA ? (x1ShapeLen - 2) : (x1ShapeLen - 1);
    uint64_t x1M = x1Shape.GetDim(x1MIndex);
    uint64_t scaleX1M = pertoken.GetDim(x1MIndex);
    uint64_t x1K = x1Shape.GetDim(x1KIndex);
    uint64_t scaleX1K = pertoken.GetDim(x1KIndex);
    OP_TILING_CHECK(
        (ops::CeilDiv(x1M, inputParams_.groupSizeM) != scaleX1M),
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "In G-B quantification, the m dimension size of x1 ceildivided by groupSizeM should be equal to \
the m dimension size of pertokenScale, but now, groupSizeM is %lu, \
m dimension size of pertokenScale is %lu, m dimension size of x1Shape is %lu.",
            inputParams_.groupSizeM, scaleX1M, x1M),
        return false);
    OP_TILING_CHECK(
        (ops::CeilDiv(x1K, inputParams_.groupSizeK) != scaleX1K),
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "In G-B quantification, the k dimension size of x1 ceildivided by groupSizeK should be equal to \
the k dimension size of pertokenScale, but now, groupSizeK is %lu, \
k dimension size of pertokenScale is %lu, k dimension size of x1Shape is %lu.",
            inputParams_.groupSizeK, scaleX1K, x1K),
        return false);
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckDimValidInPerblockMode(size_t x1ShapeLen, size_t x2ShapeLen,
                                                            size_t pertokenShapeLen, size_t scaleShapeLen) const
{
    OP_TILING_CHECK(scaleShapeLen != x2ShapeLen,
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName,
                        "In G-B quantification, x2 dimension and scale dimension should be equal, \
but x2 dimension is: %zu, scale dimension is: %zu.",
                        x2ShapeLen, scaleShapeLen),
                    return false);
    OP_TILING_CHECK(
        pertokenShapeLen != x1ShapeLen,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "In G-B quantification, x1 dimension and pertoken dimension should be equal, \
but x1 dimension is: %zu, pertoken dimension is: %zu.",
            x1ShapeLen, pertokenShapeLen),
        return false);
    return true;
}

bool AdaptiveSlidingWindowBasicTilingV4::CheckBatchValidInPerblockMode(const gert::Shape& scaleShape,
                                                              const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto x1ShapeLen = x1Shape.GetDimNum();
    if (x2ShapeLen > DIM_NUM_TWO) {
        for (size_t i = 0; i < x2ShapeLen - DIM_NUM_TWO; ++i) {
            OP_TILING_CHECK(scaleShape.GetDim(i) != x2Shape.GetDim(i),
                            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                                  "In G-B quantification, x2 batch and scale batch should be equal,"
                                                  "but at dimension %zu, x2 batch: %ld, scale batch: %ld.",
                                                  i, x2Shape.GetDim(i), scaleShape.GetDim(i)), return false);
        }
    }
    if (x1ShapeLen > DIM_NUM_TWO) {
        for (size_t i = 0; i < x1ShapeLen - DIM_NUM_TWO; ++i) {
            OP_TILING_CHECK(pertoken.GetDim(i) != x1Shape.GetDim(i),
                            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                                  "In G-B quantification, x1 batch and pertoken batch should be equal,"
                                                  "but at dimension %zu, x1 batch: %ld, pertoken batch: %ld.",
                                                  i, x1Shape.GetDim(i), pertoken.GetDim(i)), return false);
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