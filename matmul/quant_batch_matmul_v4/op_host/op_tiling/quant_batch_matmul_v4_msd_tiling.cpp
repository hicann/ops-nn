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
 * \file quant_batch_matmul_v4_msd_tiling.cpp
 * \brief
 */
#include "quant_batch_matmul_v4_msd_tiling.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "vector"
#include "error_util.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "op_cache_tiling.h"
#include "tiling_base/tiling_key.h"

using Ops::NN::Optiling::RecursiveSum;
namespace optiling {
const std::vector<uint64_t> BASEN = {64, 128, 256, 512};    // baseN切分方案

ge::graphStatus QuantBatchMatmulV4MsdTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid context_."),
                    return ge::GRAPH_PARAM_INVALID);
    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeInputs(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to analyze context_ info."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(inputParams_.opName, "Input params: M K N [%ld, %ld, %ld], transposeA[%s], transposeB[%s]",
             inputParams_.mSize, inputParams_.kSize, inputParams_.nSize,
             inputParams_.transA ? "true" : "false", inputParams_.transB ? "true" : "false");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::CheckContext()
{
    auto x1Shape = context_->GetInputShape(X1_IDX);
    auto x1Desc = context_->GetInputDesc(X1_IDX);
    auto x2Shape = context_->GetInputShape(X2_IDX);
    auto x2Desc = context_->GetInputDesc(X2_IDX);
    auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_IDX);
    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_IDX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_IDX);
    auto x2SclaeDesc = context_->GetOptionalInputDesc(X2_SCALE_IDX);
    auto yOffsetShape = context_->GetOptionalInputShape(Y_OFFSET_IDX);
    auto yOffsetDesc = context_->GetOptionalInputDesc(Y_OFFSET_IDX);
    auto outputShape = context_->GetOutputShape(Y_OUTPUT_IDX);
    auto outputDesc = context_->GetOutputDesc(Y_OUTPUT_IDX);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Function context_.GetAttrs() failed!"),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2ScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2SclaeDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, yOffsetShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, yOffsetDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    const int64_t *groupSizePtr = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_IDX);
    OP_TILING_CHECK(groupSizePtr == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size can not be nullptr."),
                    return false);
    inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    OP_TILING_CHECK(inputParams_.groupSize != 256,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size only support group k 256 on the A8W4 scenario, actual group sizeN is [%lu]", inputParams_.groupSize),
                    return false);

    // check transposeX1
    auto transposeX1 = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_IDX);
    OP_TILING_CHECK(transposeX1 == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "TransposeX1 false can not be nullptr."),
                    return false);
    OP_TILING_CHECK(*transposeX1 != false,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Unsupported value [%d] for transposeX1, "
                                                   "Only transposeX1 = false is supported.", *transposeX1),
                    return false;
                    );
    inputParams_.transA = transposeX1 != nullptr && *transposeX1;
    // check transposeX2
    auto transposeX2 = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_IDX);
    OP_TILING_CHECK(transposeX2 == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "TransposeX2 false can not be nullptr."),
                    return false);
    OP_TILING_CHECK(*transposeX2 != false,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Unsupported value [%d] for transposeX2, "
                                                   "Only transposeX2 = false is supported.", *transposeX2),
                    return false;
                    );
    inputParams_.transB = transposeX2 != nullptr && *transposeX2;
    return true;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_IDX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(X2_IDX)->GetDataType();
    inputParams_.x1ScaleDtype = context_->GetOptionalInputDesc(X1_SCALE_IDX)->GetDataType();
    inputParams_.x2ScaleDtype = context_->GetOptionalInputDesc(X2_SCALE_IDX)->GetDataType();
    inputParams_.yOffsetDtype = context_->GetOptionalInputDesc(Y_OFFSET_IDX)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(Y_OUTPUT_IDX)->GetDataType();
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_INT8,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x1 should be int8, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_INT4,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x2 should be int4, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.x1ScaleDtype != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x1Scale should be float32, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.x1ScaleDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.x2ScaleDtype != ge::DT_UINT64,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of x2Scale should be uint64, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str()),return false);
    OP_TILING_CHECK(inputParams_.yOffsetDtype != ge::DT_FLOAT,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Input dtype of yOffset should be float32, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.yOffsetDtype).c_str()),return false);
    OP_TILING_CHECK(!(inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"Output dtype  should be float16 or bfloat16, actual dtype is %s",
                                                   ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),return false);
    return true;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeInputs()
{
    auto x1Shape = GetShape(X1_IDX);
    auto x2Shape = GetShape(X2_IDX);
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    OP_TILING_CHECK(x1ShapeLen !=2 || x2ShapeLen != 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x1 dimension and x2 dimension should equal to 2, but x1 dimension: %zu, x2 dimension: %zu.",
                                                   x1ShapeLen, x2ShapeLen), return false);
    auto x1Inner = x1Shape.GetDim(0);
    auto x1Outer = x1Shape.GetDim(1);
    auto x2Inner = x2Shape.GetDim(0);
    auto x2Outer = x2Shape.GetDim(1);
    OP_TILING_CHECK(x1Outer != x2Inner,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "The size of k dimension in x1[%lu] is not equal to the size of k dimension in x2[%lu].",
                                                   x1Outer, x2Inner), return false);
    inputParams_.mSize = x1Inner;
    inputParams_.kSize = x1Outer;
    inputParams_.nSize = x2Outer;

    auto aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X1_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(aFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "aFormat Only support Nd"), return false);
    auto bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X2_IDX)->GetStorageFormat()));
    OP_TILING_CHECK(bFormat != ge::FORMAT_ND, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "bFormat Only support Nd"), return false);
    return AnalyzeScaleInputs() && AnalyzeYOffsetInputs();
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeScaleInputs()
{
    auto x1ScaleShape = GetOptionShape(X1_SCALE_IDX);
    auto x2ScaleShape = GetOptionShape(X2_SCALE_IDX);
    auto x1ScaleShapeLen = x1ScaleShape.GetDimNum();
    OP_TILING_CHECK(x1ScaleShapeLen != 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x1Scale dimension should equal to 2, but x1Scale dimension %zu", x1ScaleShapeLen),
                    return false);
    auto x1ScaleShapeInner = x1ScaleShape.GetDim(0);
    auto x1ScaleShapeOuter = x1ScaleShape.GetDim(1);
    OP_TILING_CHECK(static_cast<uint64_t>(x1ScaleShapeInner) != inputParams_.mSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x1Scale inner dim should equal to mSize %zu, but x1Scale inner dim is %zu",
                                                   inputParams_.mSize, x1ScaleShapeInner),
                    return false);
    OP_TILING_CHECK(x1ScaleShapeOuter != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x1Scale outer dim should equal to 1, but x1Scale outer dim is %zu",
                                                    x1ScaleShapeOuter),
                    return false);
    auto x2ScaleShapeLen = x2ScaleShape.GetDimNum();
    OP_TILING_CHECK(x2ScaleShapeLen != 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x2Scale dimension should equal to 2, but x2Scale dimension is %zu", x2ScaleShapeLen),
                    return false);
    auto x2ScaleShapeInner = x2ScaleShape.GetDim(0);
    auto x2ScaleShapeOuter = x2ScaleShape.GetDim(1);
    OP_TILING_CHECK(static_cast<uint64_t>(x2ScaleShapeInner) * inputParams_.groupSize != inputParams_.kSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x2Scale inner dim should equal to nSize %zu / groupSizeK %zu, but x2Scale inner dim is %zu",
                                                   inputParams_.nSize, inputParams_.groupSize, x2ScaleShapeInner),
                    return false);
    OP_TILING_CHECK(static_cast<uint64_t>(x2ScaleShapeOuter) != inputParams_.nSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "x2Scale outer dim should equal to nSize %zu, but x2Scale outer dim is %zu",
                                                   inputParams_.nSize, x2ScaleShapeOuter),
                    return false);
    return true;
}

bool QuantBatchMatmulV4MsdTiling::AnalyzeYOffsetInputs()
{
    auto yOffsetShape = GetOptionShape(Y_OFFSET_IDX);
    auto yOffsetShapeLen = yOffsetShape.GetDimNum();
    OP_TILING_CHECK(yOffsetShapeLen != 1,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input yOffset dimension should equal to 1, but yOffset dimension %zu",
                                                    yOffsetShapeLen),
                    return false);
    auto yOffsetShapeInner = yOffsetShape.GetDim(0);
    OP_TILING_CHECK(static_cast<uint64_t>(yOffsetShapeInner) != inputParams_.nSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input yOffset size should equal to nSize %zu, but yOffset size is %zu",
                                                    inputParams_.nSize, yOffsetShapeInner),
                    return false);
    return true;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);
    constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
    constexpr uint32_t cvParallNum = 4;
    constexpr uint32_t THIRTY_TWO = 32;
    constexpr uint32_t UBCALSIZE = 32U * 256U;
    constexpr uint32_t UBRESTBYTES = 9U * 32U * 256U;
    constexpr uint32_t EIGHT = 8;
    uint32_t singleN = 256;
    uint32_t singleM = 128;
    tilingData_->set_coreNum(compileInfo_.aicNum);
    tilingData_->set_groupSize(inputParams_.groupSize);
    tilingData_->set_kSize(inputParams_.kSize);
    tilingData_->set_nSize(inputParams_.nSize);
    tilingData_->set_mSize(inputParams_.mSize);
    tilingData_->set_ubCalSize(UBCALSIZE);
    tilingData_->set_vBaseM(THIRTY_TWO);
    tilingData_->set_ubRestBytes(UBRESTBYTES);
    tilingData_->set_parallNum(cvParallNum);
    if(!SetMatmulTiling()) {
        return ge::GRAPH_FAILED;
    }

    workspaceSize_ = SYS_WORKSPACE_SIZE + cvParallNum * compileInfo_.aicNum * singleN * singleM * sizeof(int32_t) * EIGHT;
    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4MsdTiling::SetMatmulTiling()
{
    constexpr uint32_t baseM = 16;  //基本块大小，参考设计文档
    constexpr uint32_t baseK = 256; //基本块大小，参考设计文档
    uint64_t baseMNum = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(baseM));
    uint64_t aicNumN = ops::CeilDiv(static_cast<uint64_t>(compileInfo_.aicNum), baseMNum);
    uint64_t baseN = ops::CeilDiv(inputParams_.nSize, aicNumN);
    for (size_t i =0; i < BASEN.size(); ++i) {
        if (baseN < BASEN[i]) {
            baseN = BASEN[i];
            break;
        }
        if (i == BASEN.size() - 1) {
            baseN = BASEN[i];
        }
    }
    matmul_tiling::PlatformInfo platformInfo;
    InitPlatformInfo(&compileInfo_, platformInfo);
    matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBias(false);
    mm.SetOrgShape(baseM, inputParams_.nSize, inputParams_.kSize);
    mm.SetShape(baseM, baseN, inputParams_.kSize);
    mm.SetFixSplit(baseM, baseN, baseK);
    if (mm.GetTiling(tilingData_->matmulTiling) ==-1) {
        return false;
    }
    constexpr uint32_t stepK = 2;    //l0a l0b上开pingpong
    tilingData_->matmulTiling.set_dbL0C(1);
    tilingData_->matmulTiling.set_stepKa(stepK);
    tilingData_->matmulTiling.set_stepKb(stepK);
    tilingData_->matmulTiling.set_depthA1(1);
    tilingData_->matmulTiling.set_depthB1(1);
    tilingData_->matmulTiling.set_stepM(1);
    tilingData_->matmulTiling.set_stepN(1);
    tilingData_->matmulTiling.set_baseK(baseK / stepK);   //l0a l0b上开pingpong
    return true;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingDataManager_ = std::make_unique<QuantBatchMatmulV4MsdTilingData>();
        OP_TILING_CHECK(tilingDataManager_ == nullptr,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "failed to instantiate tilingData_"),
                        return ge::GRAPH_FAILED);
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,"tiling data capacity %zu < actual tiling data size %zu",
                                                   context_->GetRawTilingData()->GetCapacity(),
                                                   tilingData_->GetDataSize()), return ge::GRAPH_SUCCESS;);
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4MsdTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    QuantBatchMatmulV4CompileInfo compileInfo;
    compileInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo.l0bSize);

    compileInfo.workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo.socVersion = ascendcPlatform.GetSocVersion();

    TilingPrepareForOpCache(context_);
    compileInfoInit_ = true;
    compileInfo_= compileInfo;
}

void QuantBatchMatmulV4MsdTiling::InitPlatformInfo(const QuantBatchMatmulV4CompileInfo* compileInfoPtr_, matmul_tiling::PlatformInfo& platformInfo) const
{
    platformInfo.socVersion = compileInfoPtr_->socVersion;
    platformInfo.l1Size = compileInfoPtr_->l1Size;
    platformInfo.l0CSize = compileInfoPtr_->l0cSize;
    platformInfo.ubSize = compileInfoPtr_->ubSize;
    platformInfo.l0ASize = compileInfoPtr_->l0aSize;
    platformInfo.l0BSize = compileInfoPtr_->l0bSize;
}

uint64_t QuantBatchMatmulV4MsdTiling::GetTilingKey() const
{
    return RecursiveSum(inputParams_.transA, inputParams_.transB, inputParams_.antiQuantType,
                        inputParams_.hasAntiQuantOffset, inputParams_.weightNz, KernelTemplateType::MSD_BASIS);
}

const gert::Shape QuantBatchMatmulV4MsdTiling::GetShape(const size_t index) const
{
    return context_->GetInputShape(index)->GetStorageShape();
}

const gert::Shape QuantBatchMatmulV4MsdTiling::GetOptionShape(const size_t index) const
{
    return context_->GetOptionalInputShape(index)->GetStorageShape();
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::DoLibApiTiling()
{
    context_->SetBlockDim(compileInfo_.aicNum);
    context_->SetScheduleMode(1);   // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要做此设置避免影响整网其他算子
    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4MsdTiling::PrintTilingData() const
{
    if (CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }
    std::stringstream ss;
    ss << " CoreNum: " << tilingData_->get_coreNum() << " vBaseM: " << tilingData_->get_vBaseM()
       << " ubRestBytes: " << tilingData_->get_ubRestBytes() << " parallNum: " << tilingData_->get_parallNum()
       << " ubCalSize: " << tilingData_->get_ubCalSize() << " mSize: " << tilingData_->get_mSize()
       << " kSize: " << tilingData_->get_kSize() << " nSize: " << tilingData_->get_nSize()
       << " groupSize: " << tilingData_->get_groupSize();
    OPS_LOG_D(inputParams_.opName, "api tiling: %s", ss.str().c_str());
    PrintMatmulTilingData();
}

void QuantBatchMatmulV4MsdTiling::PrintMatmulTilingData() const
{
    if (CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }
    optiling::TCubeTiling &matmulTiling = tilingData_->matmulTiling;
    std::stringstream ss;
    ss << " usedCoreNum: " << matmulTiling.get_usedCoreNum() << " M: " << matmulTiling.get_M() << " N: " << matmulTiling.get_N()
       << " Ka: " << matmulTiling.get_Ka() << " Kb: " << matmulTiling.get_Kb() << " singleCoreM: " << matmulTiling.get_singleCoreM()
       << " singleCoreN: " << matmulTiling.get_singleCoreN() << " singleCoreK: " << matmulTiling.get_singleCoreK()
       << " baseM: " << matmulTiling.get_baseM() << " baseN: " << matmulTiling.get_baseN() << " baseK: " << matmulTiling.get_baseK()
       << " depthA1: " << matmulTiling.get_depthA1() << " depthB1: " << matmulTiling.get_depthB1()
       << " stepM: " << matmulTiling.get_stepM() << " stepN: " << matmulTiling.get_stepN() << " stepka: " << matmulTiling.get_stepKa()
       << " stepkb: " << matmulTiling.get_stepKb() << " isBias: " << matmulTiling.get_isBias()
       << " transLength: " << matmulTiling.get_transLength() << " iterateOrder: " << matmulTiling.get_iterateOrder()
       << " shareMode: " << matmulTiling.get_shareMode() << " dbL0A: " << matmulTiling.get_dbL0A()
       << " dbL0B: " << matmulTiling.get_dbL0B() << " dbL0C: " << matmulTiling.get_dbL0C()
       << " usedL1Size: " << matmulTiling.get_shareL1Size() << " usedL0CSize: " << matmulTiling.get_shareL0CSize()
       << " usedUBSize: " << matmulTiling.get_shareUbSize() << " batchM: " << matmulTiling.get_batchM()
       << " batchN: " << matmulTiling.get_batchN() << " singleBatchM: " << matmulTiling.get_singleBatchM()
       << " singleBatchN: " << matmulTiling.get_singleBatchN();
    OPS_LOG_D(inputParams_.opName, "matmulTiling: %s", ss.str().c_str());
}

bool QuantBatchMatmulV4MsdTiling::IsCapable()
{
    OP_TILING_CHECK(inputParams_.kSize > 18432 ,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "K should be less than 18432 on the A8W4 scenario,"
                                                    "but now is %lu", inputParams_.kSize),
                    return false);
    OP_TILING_CHECK(inputParams_.kSize % inputParams_.groupSize != 0 ,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "K should be divisible by groupSizek, but now k = %lu,"
                                                    "groupSize is %lu", inputParams_.kSize, inputParams_.groupSize),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSize != 256 ,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "groupSize should be 256, but now groupSize = %lu",
                                                    inputParams_.groupSize),
                    return false);
    return true;
}

void QuantBatchMatmulV4MsdTiling::Reset()
{
    inputParams_.transA = false;
    inputParams_.transB = false;
    inputParams_.hasAntiQuantOffset = false;
    inputParams_.weightNz = false;
    inputParams_.groupSize = 0UL;
    inputParams_.libApiWorkSpaceSize = 0U;
    inputParams_.mSize = 0L;
    inputParams_.kSize = 0L;
    inputParams_.nSize = 0L;
    inputParams_.aDtype = ge::DT_INT8;
    inputParams_.bDtype = ge::DT_INT4;
    inputParams_.cDtype = ge::DT_FLOAT16;
    inputParams_.x1ScaleDtype = ge::DT_FLOAT;
    inputParams_.x2ScaleDtype = ge::DT_UINT64;
    inputParams_.yOffsetDtype = ge::DT_FLOAT;
    inputParams_.antiQuantType = QuantBatchMatmulV4QuantType::MX;
    inputParams_.opName = nullptr;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::GetPlatformInfo()
{
    if (!compileInfoInit_) {
        auto compileInfoPtr = reinterpret_cast<const QuantBatchMatmulV4CompileInfo *>(context_->GetCompileInfo());
        OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");
        compileInfo_ = *compileInfoPtr;
    }

    aicoreParams_.blockDim = 0;
    aicoreParams_.ubSize = compileInfo_.ubSize;
    aicoreParams_.l1Size = compileInfo_.l1Size;
    aicoreParams_.l0cSize = compileInfo_.l0cSize;
    aicoreParams_.l0cSize = compileInfo_.l0cSize;
    aicoreParams_.l0aSize = compileInfo_.l0aSize;
    aicoreParams_.l0bSize = compileInfo_.l0bSize;
    inputParams_.libApiWorkSpaceSize = compileInfo_.workspaceNum;

    OP_LOGI(inputParams_.opName,
        "get platform: ubSize(%lu) l1Size(%lu) l0aSize(%lu) l0bSize(%lu) l0cSize(%lu)",
        aicoreParams_.ubSize,
        aicoreParams_.l1Size,
        aicoreParams_.l0aSize,
        aicoreParams_.l0bSize,
        aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::PostTiling()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4MsdTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}
}// namespace optiling