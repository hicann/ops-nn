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
* \file fused_quant_matmul_swiglu_tiling.cpp
* \brief
*/
#include "fused_quant_matmul_swiglu_tiling.h"

#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "op_api/op_util.h"

namespace optiling {

constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;

constexpr uint64_t WHITE_SHAPE_M = 0;
constexpr uint64_t WHITE_SHAPE_K = 1;
constexpr uint64_t WHITE_SHAPE_N = 2;

constexpr uint64_t shapeInputWhiteList[][3] = {
    /* m, k, n */
    {1, 4096, 5504},
    {3072, 4096, 5504},
    {1, 4096, 11008},
    {3072, 4096, 11008},

    {1, 2560, 6144},
    {3072, 2560, 6144},
    {1, 2560, 12288},
    {3072, 2560, 12288},

    {1, 2560, 1536},
    {3072, 2560, 1536},
    {1, 2560, 3072},
    {3072, 2560, 3072},

    {1, 3072, 5632},
    {3072, 3072, 5632},
    {1, 3072, 11264},
    {3072, 3072, 11264},

    {256, 2560, 1536}
};

static std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP =
{
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
    {ge::DT_INT32, matmul_tiling::DataType::DT_INT32},
    {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_HIFLOAT8},
    {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_FLOAT8_E4M3FN},
    {ge::DT_FLOAT8_E5M2, matmul_tiling::DataType::DT_FLOAT8_E5M2},
};

const std::unordered_map<std::string, FQMMFusedOpType> FUSED_OP_TYPE_STR_TO_ENUM_MAP = {
    {"", FQMMFusedOpType::NONE},
    {"relu", FQMMFusedOpType::RELU},
    {"swiglu", FQMMFusedOpType::SWIGLU},
};

uint32_t FusedQuantMatMulSwigluTiling::GetX3Idx() const
{
    return X3_INDEX_FQMM;
}

uint32_t FusedQuantMatMulSwigluTiling::GetYScaleIdx() const
{
    return Y_SCALE_INDEX_FQMM;
}

bool FusedQuantMatMulSwigluTiling::IsFusedSwigluType()
{
    return fusedOpType_ == (uint64_t)FQMMFusedOpType::SWIGLU;
}

ge::graphStatus FusedQuantMatMulSwigluTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(inputParams_.opName, "invalid context"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!AnalyzeAttrs(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "fail to analyze context attr info"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!IsFusedSwigluType(), OP_LOGI(inputParams_.opName, "is not swiglu fused"), return ge::GRAPH_PARAM_INVALID);

    OP_TILING_CHECK(!AnalyzeDtype() || !AnalyzeInputs(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "fail to analyze context dtype/inputs info"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(inputParams_.transA == true || inputParams_.transB == true,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "transA and transB should be false"), return ge::GRAPH_FAILED);

    OP_LOGD(inputParams_.opName, "input params: MKN[%ld, %ld, %ld], transA[%s], transB[%s], bias[%s]",
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, inputParams_.transA ? "true" : "false",
            inputParams_.transB ? "true" : "false", inputParams_.hasBias ? "true" : "false");

    return ge::GRAPH_SUCCESS;
}

bool FusedQuantMatMulSwigluTiling::IsCapable()
{
    if (!IsFusedSwigluType()) {
        return false;
    }
    OP_LOGI(inputParams_.opName, "fqmm for fusedOpType = SWIGLU");
    return true;
}

void FusedQuantMatMulSwigluTiling::SetFormat()
{
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ) {
        inputParams_.bFormat = ge::FORMAT_FRACTAL_NZ;
    }
}

bool FusedQuantMatMulSwigluTiling::AnalyzeDtype()
{
    auto scaleDesc = context_->GetOptionalInputDesc(GetScaleIdx());
    auto quantDesc = context_->GetOptionalInputDesc(GetX3Idx());
    auto quantDtype = quantDesc != nullptr ? quantDesc->GetDataType() : ge::DT_FLOAT;
    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    auto outDtype = context_->GetOutputDesc(0)->GetDataType();

    inputParams_.aDtype = context_->GetInputDesc(GetX1Idx())->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(GetX2Idx())->GetDataType();
    inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : inputParams_.biasDtype;
    inputParams_.cDtype = ge::DT_FLOAT16;

    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_INT8 && inputParams_.bDtype != ge::DT_INT4,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "a dtype should be int8 and b dtype should be int4, actual is %s and %s",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    return false);
    OP_TILING_CHECK(quantDesc != nullptr && quantDtype != ge::DT_FLOAT,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "quant dtype should be float, actual is %s",
                                          ge::TypeUtils::DataTypeToSerialString(quantDtype).c_str()),
                    return false);
    OP_TILING_CHECK(quantDesc != nullptr && outDtype != ge::DT_INT8,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "out dtype should be int8 when dtype is float, actual is %s",
                                          ge::TypeUtils::DataTypeToSerialString(outDtype).c_str()),
                    return false);

    SetFormat();
    return true;
}

bool FusedQuantMatMulSwigluTiling::AnalyzeInputs()
{
    auto x1Shape = GetX1Shape(GetX1Idx());
    auto x2Shape = GetX2Shape(GetX2Idx());
    auto biasShape = context_->GetOptionalInputShape(GetBiasIdx());
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    if (!CheckShapeInRangeForMandtoryInputs(x1ShapeLen, x2ShapeLen)) {
        return false;
    }

    OP_TILING_CHECK(x2ShapeLen != 3 || x2Shape.GetDim(0) != 2,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "fqmm swiglu x2 ori shape should be like (2, k, n) or (2, n, k)"),
                    return false);

    OP_TILING_CHECK(biasShape != nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "fqmm swiglu bias should not be set"),
                    return false);

    inputParams_.hasBias = biasShape != nullptr;

    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);
    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);

    for (size_t dim_idx = 0; dim_idx < x1ShapeLen - LAST_SECOND_DIM_INDEX; dim_idx++) {
        inputParams_.mSize *= x1Shape.GetDim(dim_idx);
    }

    bool isInWhiteList = false;
    size_t whiteListNum = sizeof(shapeInputWhiteList) / sizeof(shapeInputWhiteList[0]);
    for (size_t idx  = 0; idx < whiteListNum; idx++) {
        if (inputParams_.mSize == shapeInputWhiteList[idx][WHITE_SHAPE_M] &&
            inputParams_.kSize == shapeInputWhiteList[idx][WHITE_SHAPE_K] &&
            inputParams_.nSize == shapeInputWhiteList[idx][WHITE_SHAPE_N]) {
            isInWhiteList = true;
            break;
        }
    }
    OP_TILING_CHECK(isInWhiteList != true,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "shape m:%lu k:%lu n:%lu is not in white list",
                    inputParams_.mSize, inputParams_.kSize, inputParams_.nSize), return false);

    return true;
}

ge::graphStatus FusedQuantMatMulSwigluTiling::InitTilingData()
{
    matmul_tiling::MultiCoreMatmulTiling mm_;
    uint64_t halfL1 = compileInfo_.l1Size >> 1;
    auto aFormat = inputParams_.aFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto bFormat = inputParams_.bFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto cFormat = inputParams_.cFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;

    mm_.SetOrgShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize);
    mm_.SetShape(basicTiling_.baseM, basicTiling_.baseN, inputParams_.kSize);
    mm_.SetFixSplit(basicTiling_.baseM, basicTiling_.baseN);
    mm_.SetBufferSpace(-1, -1, -1);

    if (inputParams_.hasBias) {
        mm_.SetBias(true);
        mm_.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP[inputParams_.biasDtype]);
    } else {
        mm_.SetBias(false);
    }

    /* 预留L1给B矩阵和反量化参数，考虑到载入太多A头开销会比较大，按一半的L1支持，非全载预埋，暂不支持(当前典型场景刚好使用512K) */
    if (inputParams_.kSize * basicTiling_.baseM * 1 <= halfL1) {
        tilingData.baseParams.set_isFullLoadA(1);
        mm_.SetAType(matmul_tiling::TPosition::A1, aFormat, DTYPE_MAP[inputParams_.aDtype], inputParams_.transA);
    } else {
        OP_LOGE(inputParams_.opName, "Fused Matmul Swiglu Quant L1 is not enough!");
        return ge::GRAPH_FAILED;
    }

    mm_.SetBType(matmul_tiling::TPosition::GM, bFormat, DTYPE_MAP[inputParams_.bDtype], inputParams_.transB);
    mm_.SetCType(matmul_tiling::TPosition::VECIN, cFormat, DTYPE_MAP[inputParams_.cDtype]);

    if (mm_.GetTiling(tilingData.mmTilingData) == -1) {
        OP_LOGE(inputParams_.opName, "Fused Matmul Swiglu Quant Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    tilingData.mmTilingData.set_usedCoreNum(basicTiling_.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

void FusedQuantMatMulSwigluTiling::SetBaseBlockTiling()
{
    uint64_t mLoops;
    uint64_t nLoops;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t baseM = BASIC_BLOCK_SIZE_128;
    uint64_t baseN = BASIC_BLOCK_SIZE_256;

    basicTiling_.usedCoreNum = compileInfo_.aicNum;

    while (inputParams_.nSize < baseN) {
        baseN = baseN >> 1;
    }
    if (baseN < MIN_SHAPE_N) {
        baseN = MIN_SHAPE_N;
    }

    while (inputParams_.mSize < baseM) {
        baseM = baseM >> 1;
    }
    if (baseM < MIN_SHAPE_M) {
        baseM = MIN_SHAPE_M;
    }

    mLoops = ops::CeilDiv(inputParams_.mSize, baseM);
    if (mLoops < 1) {
        mLoops = 1;
    } else if (mLoops > basicTiling_.usedCoreNum) {
        mLoops = basicTiling_.usedCoreNum;
    }
    singleCoreM = (inputParams_.mSize + mLoops - 1) / mLoops;
    singleCoreM = ops::CeilAlign(singleCoreM, (uint64_t)ALIGN_M);
    singleCoreM = singleCoreM > baseM ? singleCoreM : baseM;
    mLoops = ops::CeilDiv(inputParams_.mSize, singleCoreM);

    nLoops = basicTiling_.usedCoreNum / mLoops;
    singleCoreN = ops::CeilDiv(inputParams_.nSize, nLoops);
    singleCoreN = ops::CeilAlign(singleCoreN, (uint64_t)ALIGN_N);
    while (baseN > MIN_SHAPE_N && baseN > singleCoreN) {
        baseN = baseN >> 1;
    }

    basicTiling_.baseM = baseM;
    basicTiling_.baseN = baseN;

    if (mLoops * nLoops < basicTiling_.usedCoreNum) {
        basicTiling_.usedCoreNum = static_cast<uint32_t>(mLoops * nLoops);
    }

    tilingData.baseParams.set_isDeqQuant(context_->GetOptionalInputShape(GetScaleIdx()) != nullptr);
    tilingData.baseParams.set_isQuant(context_->GetOptionalInputShape(GetX3Idx()) != nullptr);
    tilingData.baseParams.set_singleCoreM(singleCoreM);
    tilingData.baseParams.set_singleCoreN(singleCoreN);
    tilingData.baseParams.set_mLoops(mLoops);
    tilingData.baseParams.set_nLoops(nLoops);
    tilingData.baseParams.set_singleMTail(inputParams_.mSize - (mLoops - 1) * singleCoreM);
    tilingData.baseParams.set_singleNTail(inputParams_.nSize - (nLoops - 1) * singleCoreN);
}

ge::graphStatus FusedQuantMatMulSwigluTiling::DoOpTiling()
{
    SetBaseBlockTiling();
    OP_LOGE_IF(InitTilingData() != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED, context_->GetNodeName(), "InitTilingData fail");
    return ge::GRAPH_SUCCESS;
}

void FusedQuantMatMulSwigluTiling::PrintBaseParamsTilingData()
{
    FusedQuantMatmulSwigluBaseParams &tiling = tilingData.baseParams;
    std::stringstream ss;

    ss << " isFullLoadA: " << tiling.get_isFullLoadA()
       << " isDeqQuant: " << tiling.get_isDeqQuant() << " isQuant: " << tiling.get_isQuant()
       << " singleCoreM: " << tiling.get_singleCoreM() << " singleCoreN: " << tiling.get_singleCoreN()
       << " mLoops: " << tiling.get_mLoops() << " nLoops: " << tiling.get_nLoops()
       << " singleMTail: " << tiling.get_singleMTail() << " singleNTail: " << tiling.get_singleNTail();
    OPS_LOG_D(inputParams_.opName, "base params tiling: %s", ss.str().c_str());
}

void FusedQuantMatMulSwigluTiling::PrintMatmulTilingData()
{
    optiling::TCubeTiling &tiling = tilingData.mmTilingData;
    std::stringstream ss;

    ss << " usedCoreNum: " << tiling.get_usedCoreNum() << " M: " << tiling.get_M() << " N: " << tiling.get_N()
       << " Ka: " << tiling.get_Ka() << " Kb: " << tiling.get_Kb() << " singleCoreM: " << tiling.get_singleCoreM()
       << " singleCoreN: " << tiling.get_singleCoreN() << " singleCoreK: " << tiling.get_singleCoreK()
       << " baseM: " << tiling.get_baseM() << " baseN: " << tiling.get_baseN() << " baseK: " << tiling.get_baseK()
       << " depthA1: " << tiling.get_depthA1() << " depthB1: " << tiling.get_depthB1()
       << " stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN() << " stepka: " << tiling.get_stepKa()
       << " stepkb: " << tiling.get_stepKb() << " isBias: " << tiling.get_isBias()
       << " transLength: " << tiling.get_transLength()
       << " iterateOrder: " << ((tiling.get_iterateOrder() == 1) ? "orderM" : "orderN")
       << " shareMode: " << tiling.get_shareMode() << " dbL0A: " << tiling.get_dbL0A()
       << " dbL0B: " << tiling.get_dbL0B() << " dbL0C: " << tiling.get_dbL0C()
       << " usedL1Size: " << tiling.get_shareL1Size() << " usedL0CSize: " << tiling.get_shareL0CSize()
       << " usedUBSize: " << tiling.get_shareUbSize() << " batchM: " << tiling.get_batchM()
       << " batchN: " << tiling.get_batchN() << " singleBatchM: " << tiling.get_singleBatchM()
       << " singleBatchN: " << tiling.get_singleBatchN();
    OPS_LOG_D(inputParams_.opName, "matmul tiling: %s", ss.str().c_str());
}

ge::graphStatus FusedQuantMatMulSwigluTiling::DoLibApiTiling()
{
    PrintBaseParamsTilingData();
    PrintMatmulTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedQuantMatMulSwigluTiling::PostTiling()
{
    OP_LOGD(inputParams_.opName, "final tiling data size: %zu", tilingData.GetDataSize());

    OP_TILING_CHECK(tilingData.GetDataSize() % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "tiling data size[%zu] is not aligned to 8",
                                                tilingData.GetDataSize()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity(),
                    CUBE_INNER_ERR_REPORT(context_, "actual tiling data size %zu > context tiling data size %zu",
                                                tilingData.GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
                    return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
