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
 * \file quant_matmul_reduce_sum_tiling.cpp
 * \brief
 */
#include "quant_matmul_reduce_sum_tiling.h"

#include "error_util.h"
#include "log/log.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_key.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace optiling;

namespace {
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t X1_SCALE_INDEX = 4;
constexpr uint32_t X2_SCALE_INDEX = 5;

constexpr size_t LAST_BATCH_DIM_INDEX = 3;
constexpr size_t X_DIM_NUM_ND = 3;
constexpr size_t X1SCALE_DIM_NUM = 2;
constexpr size_t X2SCALE_DIM_NUM = 1;

constexpr int64_t L2_REAL_SIZE = 168; // B4真实的L2Size大小
constexpr int64_t L2_FAKE_SIZE = 96;  // B4被上层修改后的L2Size大小
constexpr int64_t MB_SIZE = 1048576;  // 1024 * 1024
} // namespace

namespace optiling {
ge::graphStatus QuantMatmulReduceSumTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

void QuantMatmulReduceSumTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }

    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo_.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0CSize);
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    // 纠正L2实际物理大小
    if (compileInfo_.l2Size == L2_FAKE_SIZE * MB_SIZE) {
        compileInfo_.l2Size = L2_REAL_SIZE * MB_SIZE;
    }

    if (compileInfo_.aicNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aicNum <= 0");
        return;
    }
    OP_LOGD(context_->GetNodeName(), "QuantMatmulReduceSum InitCompileInfo Success");
}

ge::graphStatus QuantMatmulReduceSumTiling::GetShapeAttrsInfo()
{
    // 获取输入信息
    tilingDataSize_ = tilingData_.GetDataSize();
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "No need to get shape and attrs from tiling context again");
        return ge::GRAPH_SUCCESS;
    }

    inputParams_.opName = context_->GetNodeName();
    OP_LOGI(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        AnalyzeAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid attrs."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        inputParams_.opName, "Input params: MKN[%ld, %ld, %ld], transA[%s], transB[%s].", inputParams_.mSize,
        inputParams_.kSize, inputParams_.nSize, inputParams_.transA ? "true" : "false",
        inputParams_.transB ? "true" : "false");

    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulReduceSumTiling::CheckContext()
{
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(X1_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X1_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(X2_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X2_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputShape(X1_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputDesc(X1_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputShape(X2_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputDesc(X2_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulReduceSumTiling::AnalyzeAttrs()
{
    inputParams_.transA = false;
    inputParams_.transB = false;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulReduceSumTiling::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_INDEX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(X2_INDEX)->GetDataType();
    inputParams_.x1scaleDtype = context_->GetOptionalInputDesc(X1_SCALE_INDEX)->GetDataType();
    inputParams_.x2scaleDtype = context_->GetOptionalInputDesc(X2_SCALE_INDEX)->GetDataType();
    inputParams_.yDtype = context_->GetOutputDesc(0)->GetDataType();

    OP_TILING_CHECK(
        inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8,
        OP_LOGE(inputParams_.opName, "x1 and x2 dtype should be int8."), return ge::GRAPH_FAILED);
    inputParams_.cDtype = ge::DT_INT32;
    OP_TILING_CHECK(
        inputParams_.x1scaleDtype != ge::DT_FLOAT,
        OP_LOGE(inputParams_.opName, "x1Scale dtype should be float."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        inputParams_.x2scaleDtype != ge::DT_BF16,
        OP_LOGE(inputParams_.opName, "x2Scale dtype should be bfloat16."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        inputParams_.yDtype != ge::DT_BF16, OP_LOGE(inputParams_.opName, "y dtype should be bfloat16."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulReduceSumTiling::AnalyzeShapes()
{
    const auto &x1Sahpe = context_->GetInputShape(X1_INDEX)->GetOriginShape();
    const auto &x2Shape = context_->GetInputShape(X2_INDEX)->GetOriginShape();
    const auto &x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_INDEX)->GetOriginShape();
    const auto &x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_INDEX)->GetOriginShape();
    auto x1DimNum = x1Sahpe.GetDimNum();
    auto x2DimNum = x2Shape.GetDimNum();
    if (x1DimNum != X_DIM_NUM_ND || x2DimNum != X_DIM_NUM_ND) {
        OP_LOGE(
            inputParams_.opName, "input x and w dimension should be 3, but got x: %zu, w: %zu", x1DimNum, x2DimNum);
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        x1Sahpe.GetDim(0) != x2Shape.GetDim(0),
        OP_LOGE(inputParams_.opName, "x and w batch size should be same"), return ge::GRAPH_FAILED);

    auto batchSize = x1Sahpe.GetDim(0);
    auto mSize = x1Sahpe.GetDim(x1DimNum - 2);
    auto x1KSize = x1Sahpe.GetDim(x1DimNum - 1);
    auto x2KSize = x2Shape.GetDim(x2DimNum - 2);
    auto nSize = x2Shape.GetDim(x2DimNum - 1);
    OP_TILING_CHECK(
        x1KSize != x2KSize, OP_LOGE(inputParams_.opName, "x and w K size should be same"),
        return ge::GRAPH_FAILED);
    inputParams_.mSize = mSize;
    inputParams_.kSize = x1KSize;
    inputParams_.nSize = nSize;
    inputParams_.batchNum = batchSize;

    OP_TILING_CHECK(
        x1ScaleShape.GetDimNum() != X1SCALE_DIM_NUM,
        OP_LOGE(inputParams_.opName, "x1Scale dimension should be 2"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        x1ScaleShape.GetDim(0) != batchSize || x1ScaleShape.GetDim(1) != mSize,
        OP_LOGE(inputParams_.opName, "x1Scale shape should (b, m)"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        x2ScaleShape.GetDimNum() != X2SCALE_DIM_NUM,
        OP_LOGE(inputParams_.opName, "x2Scale dimension should be 1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        x2ScaleShape.GetDim(0) != nSize, OP_LOGE(inputParams_.opName, "x2Scale shape should (n,)"),
        return ge::GRAPH_FAILED);

    inputParams_.isPertoken = true;

    return ge::GRAPH_SUCCESS;
}

void QuantMatmulReduceSumTiling::PrintTilingData()
{
    if (CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    optiling::TCubeTiling& tiling = tilingData_.matmulTiling;
    std::stringstream ss;
    ss << " usedCoreNum: " << tiling.get_usedCoreNum() << " M: " << tiling.get_M() << " N: " << tiling.get_N()
       << " Ka: " << tiling.get_Ka() << " Kb: " << tiling.get_Kb() << " singleCoreM: " << tiling.get_singleCoreM()
       << " singleCoreN: " << tiling.get_singleCoreN() << " singleCoreK: " << tiling.get_singleCoreK()
       << " baseM: " << tiling.get_baseM() << " baseN: " << tiling.get_baseN() << " baseK: " << tiling.get_baseK()
       << " depthA1: " << tiling.get_depthA1() << " depthB1: " << tiling.get_depthB1()
       << " stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN() << " stepka: " << tiling.get_stepKa()
       << " stepkb: " << tiling.get_stepKb() << " transLength: " << tiling.get_transLength()
       << " iterateOrder: " << ((tiling.get_iterateOrder() == 1) ? "orderM" : "orderN")
       << " shareMode: " << tiling.get_shareMode() << " dbL0A: " << tiling.get_dbL0A()
       << " dbL0B: " << tiling.get_dbL0B() << " dbL0C: " << tiling.get_dbL0C()
       << " usedL1Size: " << tiling.get_shareL1Size() << " usedL0CSize: " << tiling.get_shareL0CSize()
       << " usedUBSize: " << tiling.get_shareUbSize() << " batchM: " << tiling.get_batchM()
       << " batchN: " << tiling.get_batchN() << " singleBatchM: " << tiling.get_singleBatchM()
       << " singleBatchN: " << tiling.get_singleBatchN();
    OP_LOGD(inputParams_.opName, "api tiling: %s\n", ss.str().c_str());
}

ge::graphStatus QuantMatmulReduceSumTiling::CalUbSize()
{
    // 计算ubCalSize
    constexpr int64_t ubCalCount = 26;  // 总共需要需要26份ubCalcSize
    constexpr int64_t ubRestCount = 16; // ubRestBytes需要16份ubCalcSize
    int64_t x1ScaleInSize = 2 * inputParams_.baseM * ge::GetSizeByDataType(inputParams_.x1scaleDtype);
    int64_t x2ScaleInSize = 2 * inputParams_.baseN * ge::GetSizeByDataType(inputParams_.x2scaleDtype);
    int64_t resUbSize = static_cast<int64_t>(compileInfo_.ubSize) - x2ScaleInSize - x1ScaleInSize;
    int64_t maxUbCalSize = resUbSize / ubCalCount;
    maxUbCalSize = maxUbCalSize / 32 * 32; // 32字节对齐
    int64_t ubCalSize = inputParams_.baseM * inputParams_.baseN;
    if (ubCalSize > maxUbCalSize) {
        ubCalSize = maxUbCalSize;
    }
    OP_LOGE_IF(ubCalSize <= 0, ge::GRAPH_FAILED, "QuantMatmulReduceSum", "ubCalSize should greater than 0.");
    tilingData_.qbmmReduceSumParams.set_ubCalSize(ubCalSize);

    // 计算ubRestBytes
    tilingData_.qbmmReduceSumParams.set_ubRestBytes(ubCalSize * ubRestCount);
    return ge::GRAPH_SUCCESS;
}

bool QuantMatmulReduceSumTiling::SetMatmulTiling()
{
    TCubeTiling& mt = tilingData_.matmulTiling;
    matmul_tiling::MultiCoreMatmulTiling mm;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);

    mm.SetBias(false);
    mm.SetDim(compileInfo_.aicNum);
    mm.SetShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize);
    if (mm.GetTiling(mt) == -1) {
        OP_LOGE(context_->GetNodeName(), "QuantMatmulReduceSum Get Tiling Failed!");
        return false;
    }

    // k>=1024时，且baseM=128, baseN=256，baseK=128时，修改tiling提升性能
    if (mt.get_baseM() == 128 && mt.get_baseN() == 256 && mt.get_baseK() == 128 && inputParams_.kSize >= 1024 &&
        ge::GetSizeByDataType(inputParams_.aDtype) == 1 && ge::GetSizeByDataType(inputParams_.bDtype) == 1) {
        mt.set_depthA1(8); // 开启db，4 x 2 = 8
        mt.set_depthB1(8); // 开启db，4 x 2 = 8
        mt.set_stepKa(4);  // 左矩阵K方向一次加载4个base块
        mt.set_stepKb(4);  // 右矩阵K方向一次加载4个base块
        uint64_t a1Length = mt.get_baseM() * mt.get_baseK() * ge::GetSizeByDataType(inputParams_.aDtype);
        uint64_t b1Length = mt.get_baseN() * mt.get_baseK() * ge::GetSizeByDataType(inputParams_.bDtype);
        uint64_t usedL1Size = a1Length * mt.get_depthA1() + b1Length * mt.get_depthB1();
        if (usedL1Size > compileInfo_.l1Size) {
            OP_LOGE(context_->GetNodeName(), "usedL1Size(%lu) exceed l1Size(%lu)!", usedL1Size, compileInfo_.l1Size);
            return false;
        }
        mt.set_shareL1Size(usedL1Size);
    }
    return true;
}

bool QuantMatmulReduceSumTiling::SetQbmmTiling()
{
    tilingData_.qbmmReduceSumParams.set_batchNum(inputParams_.batchNum);
    tilingData_.qbmmReduceSumParams.set_coreNum(static_cast<uint32_t>(inputParams_.usedCoreNum));
    tilingData_.qbmmReduceSumParams.set_ubBaseK(static_cast<uint32_t>(inputParams_.baseK));
    tilingData_.qbmmReduceSumParams.set_ubBaseN(static_cast<uint32_t>(inputParams_.baseN));
    tilingData_.qbmmReduceSumParams.set_isPertoken(static_cast<uint32_t>(inputParams_.isPertoken));
    tilingData_.qbmmReduceSumParams.set_isDetermine(static_cast<uint32_t>(context_->GetDeterministic()));
    return CalUbSize() == ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulReduceSumTiling::DoOpTiling()
{
    OP_LOGE_IF(!SetMatmulTiling(), ge::GRAPH_FAILED, "QuantMatmulReduceSum", "SetMatmulTiling failed.");
    inputParams_.usedCoreNum = tilingData_.matmulTiling.get_usedCoreNum();
    inputParams_.baseM = tilingData_.matmulTiling.get_baseM();
    inputParams_.baseN = tilingData_.matmulTiling.get_baseN();
    inputParams_.baseK = tilingData_.matmulTiling.get_baseK();
    OP_LOGE_IF(!SetQbmmTiling(), ge::GRAPH_FAILED, "QuantMatmulReduceSum", "SetQbmmTiling failed.");
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulReduceSumTiling::DoLibApiTiling()
{
    tilingKey_ = 0; // aic:aiv=1:2
    // 当K比较大时，比如2048，cube的计算耗时相对较大，只需要启动一个vector核
    if (inputParams_.kSize >= 2048) {
        tilingKey_ = 1; // aic:aiv=1:1
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantMatmulReduceSumTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus QuantMatmulReduceSumTiling::GetWorkspaceSize()
{
    // system workspace size is 16 * 1024 * 1024 = 16M;
    constexpr int64_t sysWorkspaceSize = 16777216;

    // when do cv parallelism, pipeline num is 4, requiring 4 pieces of workspace
    int64_t mmOutSize = 4 * inputParams_.baseM * inputParams_.baseN * inputParams_.usedCoreNum * sizeof(int32_t);

    workspaceSize_ = sysWorkspaceSize + mmOutSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulReduceSumTiling::PostTiling()
{
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.matmulTiling.get_usedCoreNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("QuantMatmulReduceSum", QuantMatmulReduceSumTiling, 0);

ge::graphStatus QuantMatmulReduceSumTilingFunc(gert::TilingContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "QuantMatmulReduceSum", "TilingContext is null!");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForQuantMatmulReduceSum(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "QuantMatmulReduceSum", "TilingParseContext is null!");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantMatmulReduceSum)
    .Tiling(QuantMatmulReduceSumTilingFunc)
    .TilingParse<QuantMatmulReduceSumCompileInfo>(TilingParseForQuantMatmulReduceSum);
} // namespace optiling
