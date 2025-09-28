/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_perblock_tiling.cpp
 * \brief
 */
#include "quant_batch_matmul_v4_perblock_tiling.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "vector"
#include "error_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"

namespace optiling {

ge::graphStatus QuantBatchMatmulV4PerblockTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        CheckContext() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(inputParams_.opName, "The context_ is invalid."),
        return ge::GRAPH_PARAM_INVALID);
    OP_TILING_CHECK(
        !AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeInputs(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to analyze context_ info."), return ge::GRAPH_PARAM_INVALID);
    OP_LOGD(
        inputParams_.opName, "Input params: M K N [%ld, %ld, %ld], transposeA[%s], transposeB[%s]", inputParams_.mSize,
        inputParams_.kSize, inputParams_.nSize, inputParams_.transA ? "true" : "false",
        inputParams_.transB ? "true" : "false");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PerblockTiling::CheckContext()
{
    auto x1Shape = context_->GetInputShape(X1_IDX);
    auto x2Shape = context_->GetInputShape(X2_IDX);
    auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_IDX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_IDX);
    auto biasShape = context_->GetOptionalInputShape(BIAS_IDX);
    auto outputShape = context_->GetOutputShape(Y_OUTPUT_IDX);
    auto x1Desc = context_->GetInputDesc(X1_IDX);
    auto x2Desc = context_->GetInputDesc(X2_IDX);
    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_IDX);
    auto x2SclaeDesc = context_->GetOptionalInputDesc(X2_SCALE_IDX);
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_IDX);
    auto outputDesc = context_->GetOutputDesc(Y_OUTPUT_IDX);

    auto attrs = context_->GetAttrs();

    OP_TILING_CHECK(attrs == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Function context_.GetAttrs() failed!"),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2ScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, biasShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2SclaeDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4PerblockTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    const int64_t *groupSizePtr = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_IDX);
    OP_TILING_CHECK(groupSizePtr == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size can not be nullptr."),
                    return false);
    inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    inputParams_.groupSizeM = (static_cast<uint64_t>(inputParams_.groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    inputParams_.groupSizeK = (static_cast<uint64_t>(inputParams_.groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    inputParams_.groupSizeN = static_cast<uint64_t>(inputParams_.groupSize) & GROUP_MNK_BIT_SIZE;
    OP_TILING_CHECK(inputParams_.groupSize == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size cannot be 0."),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeM == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size M cannot be 0."),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeK == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size K cannot be 0."),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeN == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size N cannot be 0."),
                    return false);

    // check transposeX1
    auto transposeX1 = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_IDX);
    OP_TILING_CHECK(
        transposeX1 == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "TransposeX1 should not be nullptr!"), return false);
    OP_TILING_CHECK(*transposeX1 != false,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Only transposeX1 = false is supported."),
                    return false;);
    inputParams_.transA = transposeX1 != nullptr && *transposeX1;
    // check transposeX2
    auto transposeX2 = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_IDX);
    OP_TILING_CHECK(
        transposeX2 == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "TransposeX2 should not be nullptr."), return false);
    OP_TILING_CHECK(*transposeX2 != true,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Only transposeX2 = true is supported."),
                    return false;);
    inputParams_.transB = transposeX2 != nullptr && *transposeX2;
    return true;
}

bool QuantBatchMatmulV4PerblockTiling::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_IDX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(X2_IDX)->GetDataType();
    inputParams_.perTokenScaleDtype = context_->GetOptionalInputDesc(X1_SCALE_IDX)->GetDataType();
    inputParams_.scaleDtype = context_->GetOptionalInputDesc(X2_SCALE_IDX)->GetDataType();
    inputParams_.biasDtype = context_->GetOptionalInputDesc(BIAS_IDX)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(Y_OUTPUT_IDX)->GetDataType();
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_INT8,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x1 should be int8, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_INT8,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x2 should be int8, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x1Scale should be float32, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.scaleDtype != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x2Scale should be float32, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.biasDtype != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of bias should be float32, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.cDtype != ge::DT_BF16,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Output dtype should be bfloat16, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),return false);
    return true;
}

bool QuantBatchMatmulV4PerblockTiling::AnalyzeInputs()
{
    auto x1Shape = GetShape(X1_IDX);
    auto x2Shape = GetShape(X2_IDX);
    auto x1ShapeDim = x1Shape.GetDimNum();
    auto x2ShapeDim = x2Shape.GetDimNum();
    OP_TILING_CHECK(
        x1ShapeDim != 2,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input x1 dimension should equal to 2, but x1 dimension: %zu.", x1ShapeDim),
        return false);
    OP_TILING_CHECK(
        x2ShapeDim != 2,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input x2 dimension should equal to 2, but x2 dimension: %zu.", x2ShapeDim),
        return false);
    auto x1Inner = x1Shape.GetDim(0);
    auto x1Outer = x1Shape.GetDim(1);
    auto x2Inner = x2Shape.GetDim(0);
    auto x2Outer = x2Shape.GetDim(1);

    auto x2kSize = inputParams_.transB ? x2Outer : x2Inner;
    auto x2nSize = inputParams_.transB ? x2Inner : x2Outer;
    OP_TILING_CHECK(x1Outer != x2kSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "The size of k dimension in x1[%lu] is not equal to the size of k dimension in x2[%lu].",
                                                x1Outer, x2Inner), return false);
    OP_TILING_CHECK(x2nSize % basicTiling_.baseN != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "The size of N [%lu] does not align with baseN [%lu].",
                                                x2Outer, basicTiling_.baseN), return false);
    inputParams_.mSize = x1Inner;
    inputParams_.kSize = x1Outer;
    inputParams_.nSize = inputParams_.transB ? x2Inner : x2Outer;

    auto aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X1_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(aFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "aFormat Only support Nd"), return false);
    auto bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X2_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(bFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "bFormat Only support Nd"), return false);
    return AnalyzeScaleInputs() && AnalyzeBiasInputs();
}

bool QuantBatchMatmulV4PerblockTiling::AnalyzeScaleInputs()
{
    return AnalyzeX1ScaleInput() && AnalyzeX2ScaleInput();
}

bool QuantBatchMatmulV4PerblockTiling::AnalyzeX1ScaleInput()
{
    auto x1ScaleShape = GetOptionShape(X1_SCALE_IDX);
    auto x1ScaleShapeDim = x1ScaleShape.GetDimNum();
    OP_TILING_CHECK(
        x1ScaleShapeDim != 2,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input x1Scale dimension should be 2, but x1Scale dimension %zu",
            x1ScaleShapeDim),
        return false);
    auto x1ScaleShapeInner = x1ScaleShape.GetDim(0);
    auto x1ScaleShapeOuter = x1ScaleShape.GetDim(1);
    OP_TILING_CHECK(
        static_cast<uint64_t>(x1ScaleShapeInner) != inputParams_.mSize,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "x1Scale inner dim should be mSize %zu, but x1Scale inner dim is %zu",
            inputParams_.mSize, x1ScaleShapeInner),
        return false);
    OP_TILING_CHECK(
        static_cast<uint64_t>(x1ScaleShapeOuter) != ops::CeilDiv(inputParams_.kSize, inputParams_.groupSizeK),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "x1Scale outer dim should be CeilDiv(kSize %zu / groupSizeK %zu), but x1Scale outer dim is %zu",
            inputParams_.kSize, inputParams_.groupSizeK, x1ScaleShapeOuter),
        return false);
    return true;
}

bool QuantBatchMatmulV4PerblockTiling::AnalyzeX2ScaleInput()
{
    auto x2ScaleShape = GetOptionShape(X2_SCALE_IDX);
    auto x2ScaleShapeDim = x2ScaleShape.GetDimNum();
    OP_TILING_CHECK(
        x2ScaleShapeDim != 2,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input x2Scale dimension should be 2, but x2Scale dimension is %zu",
            x2ScaleShapeDim),
        return false);
    auto x2ScaleShapeInner = x2ScaleShape.GetDim(0);
    auto x2ScaleShapeOuter = x2ScaleShape.GetDim(1);

    auto x2ScaleKSize = inputParams_.transB ? x2ScaleShapeOuter : x2ScaleShapeInner;
    auto x2ScaleNSize = inputParams_.transB ? x2ScaleShapeInner : x2ScaleShapeOuter;
    OP_TILING_CHECK(
        static_cast<uint64_t>(x2ScaleKSize) != ops::CeilDiv(inputParams_.kSize, inputParams_.groupSizeK),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "x2Scale inner dim should be CeilDiv(kSize %zu / groupSizeK %zu), but x2Scale inner dim is %zu",
            inputParams_.kSize, inputParams_.groupSizeK, x2ScaleKSize),
        return false);
    OP_TILING_CHECK(
        static_cast<uint64_t>(x2ScaleNSize) != ops::CeilDiv(inputParams_.nSize, inputParams_.groupSizeN),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "x2Scale outer dim should be CeilDiv(nSize %zu / groupSizeN %zu), but x2Scale outer dim is %zu",
            inputParams_.nSize, inputParams_.groupSizeN, x2ScaleNSize),
        return false);
    return true;
}

bool QuantBatchMatmulV4PerblockTiling::AnalyzeBiasInputs()
{
    auto biasShape = GetOptionShape(BIAS_IDX);
    auto biasShapeLen = biasShape.GetDimNum();
    OP_TILING_CHECK(biasShapeLen != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input bias dimension should equal to 1, but bias dimension %zu",
                                                    biasShapeLen),
                    return false);
    auto biasShapeInner = biasShape.GetDim(0);
    OP_TILING_CHECK(static_cast<uint64_t>(biasShapeInner) != inputParams_.nSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input bias size should equal to nSize %zu, but bias size is %zu",
                                                    inputParams_.nSize, biasShapeInner),
                    return false);
    return true;
}

ge::graphStatus QuantBatchMatmulV4PerblockTiling::SetBasicTilingValue() {
    OP_LOGE_IF(!QuantBatchMatmulV3BasicTiling::DoBasicTiling(), ge::GRAPH_FAILED, inputParams_.opName, "DoBasicTiling failed.");

    constexpr uint32_t BASE_M = 128;
    constexpr uint32_t BASE_N = 256;
    constexpr uint32_t BASE_K = 128;
    constexpr uint32_t DEPTH_A1 = 8;
    constexpr uint32_t DEPTH_B1 = 8;
    constexpr uint32_t STEP_KA = 4;
    constexpr uint32_t STEP_KB = 4;
    constexpr uint32_t DB_L0C = 1;
    constexpr uint32_t DB_L0A = 2;
    constexpr uint32_t DB_L0B = 2;
    constexpr uint32_t ALIGN32 = 32;

    basicTiling_.baseM = BASE_M;
    basicTiling_.baseN = BASE_N;
    basicTiling_.baseK = BASE_K;
    basicTiling_.depthA1 = DEPTH_A1;
    basicTiling_.depthB1 = DEPTH_B1;
    basicTiling_.stepKa = STEP_KA;
    basicTiling_.stepKb = STEP_KB;
    basicTiling_.dbL0c = DB_L0C;
    tilingData_->matmulTiling.set_dbL0A(DB_L0A);
    tilingData_->matmulTiling.set_dbL0B(DB_L0B);

    OP_LOGE_IF(basicTiling_.baseN % ALIGN32 != 0, ge::GRAPH_FAILED, inputParams_.opName, "BaseN aligns with 32.");

    QuantBatchMatmulV3BasicTiling::DoL2CacheTiling();
    if (basicTiling_.mTileCntl2 == 1 && basicTiling_.nTileCntl2 == 1) {
        basicTiling_.mTileBlock = ops::CeilDiv(inputParams_.mSize, basicTiling_.baseM);
        basicTiling_.nTileBlock = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    }
    basicTiling_.usedCoreNum = std::min(basicTiling_.usedCoreNum, basicTiling_.mTileBlock * basicTiling_.nTileBlock);

    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4PerblockTiling::SetTilingData() {
    tilingData_->matmulTiling.set_M(inputParams_.mSize);
    tilingData_->matmulTiling.set_N(inputParams_.nSize);
    tilingData_->matmulTiling.set_Ka(inputParams_.kSize);
    tilingData_->matmulTiling.set_Kb(inputParams_.kSize);
    tilingData_->matmulTiling.set_usedCoreNum(basicTiling_.usedCoreNum);
    tilingData_->matmulTiling.set_singleCoreM(basicTiling_.baseM);
    tilingData_->matmulTiling.set_singleCoreN(basicTiling_.baseN);
    tilingData_->matmulTiling.set_singleCoreK(basicTiling_.singleCoreK);
    tilingData_->matmulTiling.set_baseM(basicTiling_.baseM);
    tilingData_->matmulTiling.set_baseN(basicTiling_.baseN);
    tilingData_->matmulTiling.set_baseK(basicTiling_.baseK);
    tilingData_->matmulTiling.set_depthA1(basicTiling_.depthA1);
    tilingData_->matmulTiling.set_depthB1(basicTiling_.depthB1);
    tilingData_->matmulTiling.set_stepM(basicTiling_.stepM);
    tilingData_->matmulTiling.set_stepN(basicTiling_.stepN);
    tilingData_->matmulTiling.set_stepKa(basicTiling_.stepKa);
    tilingData_->matmulTiling.set_stepKb(basicTiling_.stepKb);
    tilingData_->matmulTiling.set_iterateOrder(basicTiling_.iterateOrder);
    tilingData_->matmulTiling.set_dbL0C(basicTiling_.dbL0c);  // 1: off, 2:on
    tilingData_->tileL2cacheTiling.set_mTileCntL2(basicTiling_.mTileCntl2);
    tilingData_->tileL2cacheTiling.set_nTileCntL2(basicTiling_.nTileCntl2);
    tilingData_->tileL2cacheTiling.set_mTileBlock(basicTiling_.mTileBlock);
    tilingData_->tileL2cacheTiling.set_nTileBlock(basicTiling_.nTileBlock);
    tilingData_->tileL2cacheTiling.set_calOrder(basicTiling_.calOrder);
    tilingData_->tileL2cacheTiling.set_isBasicTiling(1U);
    tilingData_->set_groupSizeM(inputParams_.groupSizeM);
    tilingData_->set_groupSizeN(inputParams_.groupSizeN);
    tilingData_->set_groupSizeK(inputParams_.groupSizeK);
    tilingData_->set_ubCalcM(basicTiling_.baseM / CV_PARALL_NUM);
    tilingData_->set_ubCalcN(basicTiling_.baseN);
    tilingData_->set_transA(inputParams_.transA);
    tilingData_->set_transB(inputParams_.transB);
}

ge::graphStatus QuantBatchMatmulV4PerblockTiling::DoOpTiling()
{
    InitCompileInfo();
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfoForTiling fail");
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);

    SetTransAttr(trans_);
    OP_TILING_CHECK(
        SetBasicTilingValue() == ge::GRAPH_FAILED,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "unable to set basic tiling data"),
        return ge::GRAPH_FAILED);
    SetTilingData();

    constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
    workspaceSize_ = SYS_WORKSPACE_SIZE +
                     compileInfo_.aicNum * basicTiling_.baseN * basicTiling_.baseM * sizeof(int32_t) * CV_PARALL_NUM;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PerblockTiling::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingDataManager_ = std::make_unique<QuantBatchMatmulV4PerblockTilingData>();
        OP_TILING_CHECK(
            tilingDataManager_ == nullptr,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "instantiate tilingData_ failed!"),
            return ge::GRAPH_FAILED);
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams_.opName, "actual tiling data size %zu > tiling data capacity %zu",
                        tilingData_->GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
                    return ge::GRAPH_SUCCESS;);
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantBatchMatmulV4PerblockTiling::GetTilingKey() const
{
    return RecursiveSum(inputParams_.transA, inputParams_.transB, QuantBatchMatmulV4QuantType::MX,
                        /*hasAntiQuantOffset*/0, /*weightNz*/0, KernelTemplateType::PERBLOCK_BASIS);
}

const gert::Shape QuantBatchMatmulV4PerblockTiling::GetShape(const size_t index)
{
    return context_->GetInputShape(index)->GetStorageShape();
}

const gert::Shape QuantBatchMatmulV4PerblockTiling::GetOptionShape(const size_t index)
{
    return context_->GetOptionalInputShape(index)->GetStorageShape();
}

ge::graphStatus QuantBatchMatmulV4PerblockTiling::DoLibApiTiling()
{
    context_->SetBlockDim(compileInfo_.aicNum);
    context_->SetScheduleMode(1);   // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要做此设置避免影响整网其他算子
    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4PerblockTiling::PrintTilingData()
{
    if (CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }
    PrintMatmulTilingData();
}

void QuantBatchMatmulV4PerblockTiling::PrintMatmulTilingData()
{
    if (CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }
    optiling::TCubeTiling &matmulTiling = tilingData_->matmulTiling;
    std::stringstream ss;
    ss << " M: " << matmulTiling.get_M() << " N: " << matmulTiling.get_N() << " Ka: " << matmulTiling.get_Ka()
       << " Kb: " << matmulTiling.get_Kb() << " singleCoreM: " << matmulTiling.get_singleCoreM()
       << " singleCoreN: " << matmulTiling.get_singleCoreN() << " singleCoreK: " << matmulTiling.get_singleCoreK()
       << " baseM: " << matmulTiling.get_baseM() << " baseN: " << matmulTiling.get_baseN()
       << " baseK: " << matmulTiling.get_baseK() << " depthA1: " << matmulTiling.get_depthA1()
       << " depthB1: " << matmulTiling.get_depthB1() << " stepM: " << matmulTiling.get_stepM()
       << " stepN: " << matmulTiling.get_stepN() << " stepka: " << matmulTiling.get_stepKa()
       << " stepkb: " << matmulTiling.get_stepKb() << " isBias: " << matmulTiling.get_isBias()
       << " transLength: " << matmulTiling.get_transLength() << " iterateOrder: " << matmulTiling.get_iterateOrder()
       << " shareMode: " << matmulTiling.get_shareMode() << " dbL0A: " << matmulTiling.get_dbL0A()
       << " dbL0B: " << matmulTiling.get_dbL0B() << " dbL0C: " << matmulTiling.get_dbL0C()
       << " usedL1Size: " << matmulTiling.get_shareL1Size() << " usedL0CSize: " << matmulTiling.get_shareL0CSize()
       << " usedUBSize: " << matmulTiling.get_shareUbSize() << " batchM: " << matmulTiling.get_batchM()
       << " singleBatchN: " << matmulTiling.get_singleBatchN() << " usedCoreNum: " << matmulTiling.get_usedCoreNum()
       << " batchN: " << matmulTiling.get_batchN() << " singleBatchM: " << matmulTiling.get_singleBatchM();
    OPS_LOG_D(inputParams_.opName, "matmulTiling: %s", ss.str().c_str());
}

bool QuantBatchMatmulV4PerblockTiling::IsCapable()
{
    return true;
}

void QuantBatchMatmulV4PerblockTiling::Reset()
{
    inputParams_.transA = false;
    inputParams_.transB = true;
    inputParams_.groupSize = 0UL;
    inputParams_.groupSizeM = 0UL;
    inputParams_.groupSizeK = 0UL;
    inputParams_.groupSizeN = 0UL;
    inputParams_.libApiWorkSpaceSize = 0U;
    inputParams_.mSize = 0L;
    inputParams_.kSize = 0L;
    inputParams_.nSize = 0L;
    inputParams_.aDtype = ge::DT_INT8;
    inputParams_.bDtype = ge::DT_INT8;
    inputParams_.cDtype = ge::DT_BF16;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT;
    inputParams_.scaleDtype = ge::DT_FLOAT;
    inputParams_.biasDtype = ge::DT_FLOAT;
    inputParams_.opName = nullptr;
}

ge::graphStatus QuantBatchMatmulV4PerblockTiling::PostTiling()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PerblockTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}


}// namespace optiling