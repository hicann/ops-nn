/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <stdlib.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "log/log.h"
#include "../../../op_kernel/arch35/quant_matmul_activation_quant_tiling_key.h"

#define protected public
#define private public
#include "op_host/tiling_templates_registry.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_mx_basic_api_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_compile_info.h"
#include "../../../op_host/op_tiling/quant_matmul_activation_quant_helper.h"
#include "../../../op_kernel/arch35/quant_matmul_activation_quant_tiling_data.h"
#include "ut_string_utils.h"
#undef private
#undef protected

using namespace ut_str;
using namespace std;
using namespace ge;
using namespace ut_util;
using namespace optiling;

static const string kAscend950CompileInfo = R"({
    "hardware_info": {
        "BT_SIZE": 4096, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "intrinsic_fix_pipe_l0c2out_f322bf16": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
        "Intrinsic_fix_pipe_pre_conv_cast": true,
        "Intrinsic_data_move_l12bt": true,
        "UB_SIZE": 245760, "L2_SIZE": 134217728, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"
    }
})";

struct QuantMatmulActivationQuantTilingParam {
    std::string socVersion;
    std::string caseName;
    gert::Shape x1Shape;
    gert::Shape x2OriginShape;
    gert::Shape x2StorageShape;
    gert::Shape x1ScaleShape;
    gert::Shape x2ScaleShape;
    gert::Shape biasShape;
    bool hasBias = false;
    gert::Shape yShape;
    gert::Shape yScaleShape;
    bool x2IsNz = true;
    bool transposeX1 = false;
    bool transposeX2 = false;
    ge::DataType x1Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType x2Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType x1ScaleDtype = ge::DT_FLOAT8_E8M0;
    ge::DataType x2ScaleDtype = ge::DT_FLOAT8_E8M0;
    ge::DataType biasDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT8_E4M3FN;
    int64_t groupSize = 0;
    std::string activationType = "gelu_tanh";
    std::string quantMode = "mx";
    std::string roundMode = "rint";
    int64_t scaleAlg = 0;
    float dstTypeMax = 6.0f;
    bool expectResult = true;
    int64_t expectTilingKey = -1;
};

struct QuantMatmulActivationQuantTilingCsvLoadResult {
    std::vector<QuantMatmulActivationQuantTilingParam> params;
    std::vector<std::string> errors;
};

static QuantMatmulActivationQuantTilingCsvLoadResult LoadParams(const std::string& socVersion)
{
    QuantMatmulActivationQuantTilingCsvLoadResult result;
    std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    std::string casePath(rootPath + "matmul/quant_matmul_activation_quant/tests/ut/op_host/"
                                    "test_quant_matmul_activation_quant_tiling.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        result.errors.push_back("cannot open case file: " + casePath);
        return result;
    }

    std::string line;
    bool skipHeader = true;
    constexpr size_t kExpectedCols = 29UL;
    size_t lineNo = 0UL;
    while (std::getline(csvData, line)) {
        ++lineNo;
        const std::string trimLine = Trim(line);
        if (trimLine.empty() || trimLine[0] == '#') {
            continue;
        }
        if (skipHeader) {
            skipHeader = false;
            continue;
        }

        std::vector<std::string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != kExpectedCols) {
            result.errors.push_back("skip invalid csv line " + std::to_string(lineNo) + " in " + casePath +
                                    ": expected " + std::to_string(kExpectedCols) + " columns, got " +
                                    std::to_string(cols.size()));
            continue;
        }

        try {
            QuantMatmulActivationQuantTilingParam param;
            size_t idx = 0UL;
            param.socVersion = Trim(cols[idx++]);
            param.caseName = Trim(cols[idx++]);
            param.x1Shape = ParseShape(cols[idx++]);
            param.x2OriginShape = ParseShape(cols[idx++]);
            param.x2StorageShape = ParseShape(cols[idx++]);
            param.x1ScaleShape = ParseShape(cols[idx++]);
            param.x2ScaleShape = ParseShape(cols[idx++]);
            std::string biasShapeStr = Trim(cols[idx++]);
            if (!biasShapeStr.empty()) {
                param.biasShape = ParseShape(biasShapeStr);
            }
            param.yShape = ParseShape(cols[idx++]);
            param.yScaleShape = ParseShape(cols[idx++]);
            param.x2IsNz = (strcasecmp(Trim(cols[idx++]).c_str(), "NZ") == 0);
            param.hasBias = (strcasecmp(Trim(cols[idx++]).c_str(), "true") == 0);
            param.transposeX1 = (strcasecmp(Trim(cols[idx++]).c_str(), "true") == 0);
            param.transposeX2 = (strcasecmp(Trim(cols[idx++]).c_str(), "true") == 0);
            param.x1Dtype = ParseDtype(Trim(cols[idx++]));
            param.x2Dtype = ParseDtype(Trim(cols[idx++]));
            param.x1ScaleDtype = ParseDtype(Trim(cols[idx++]));
            param.x2ScaleDtype = ParseDtype(Trim(cols[idx++]));
            std::string biasDtypeStr = Trim(cols[idx++]);
            if (!biasDtypeStr.empty()) {
                param.biasDtype = ParseDtype(biasDtypeStr);
            }
            param.yDtype = ParseDtype(Trim(cols[idx++]));
            param.groupSize = stoll(Trim(cols[idx++]));
            param.activationType = Trim(cols[idx++]);
            param.quantMode = Trim(cols[idx++]);
            param.roundMode = Trim(cols[idx++]);
            param.scaleAlg = stoll(Trim(cols[idx++]));
            param.dstTypeMax = static_cast<float>(std::stod(Trim(cols[idx++])));
            param.expectResult = (strcasecmp(Trim(cols[idx++]).c_str(), "true") == 0);
            std::string tilingKeyStr = Trim(cols[idx++]);
            param.expectTilingKey = tilingKeyStr.empty() ? -1 : stoll(tilingKeyStr);

            if (param.socVersion != socVersion) {
                continue;
            }

            result.params.push_back(param);
        } catch (const std::exception& e) {
            result.errors.push_back("skip invalid csv line " + std::to_string(lineNo) + " in " + casePath + ": " +
                                    e.what());
        }
    }

    if (result.params.empty()) {
        result.errors.push_back("no valid tiling cases loaded from: " + casePath + " for socVersion: " + socVersion);
    }
    return result;
}

static const QuantMatmulActivationQuantTilingCsvLoadResult& GetParamsLoadResult()
{
    static const QuantMatmulActivationQuantTilingCsvLoadResult kCases = LoadParams("Ascend950");
    return kCases;
}

static std::vector<QuantMatmulActivationQuantTilingParam> GetParams() { return GetParamsLoadResult().params; }

using QuantMatmulActivationQuantMxHelper = QuantMatmulActivationQuantHelper<AdaptiveSlidingWindowMXBasicAPITiling>;

// These two tests are white-box checks of the tail-search ordering.  They inject an
// adaptive-window state that cannot be expressed by the public tiling CSV schema;
// ordinary operator cases remain data-driven by the CSV below.
static void SetupSwiGluTailSplitHelper(QuantMatmulActivationQuantMxHelper& helper, bool isAFullLoad)
{
    ResetQuantBatchMatmulV3InputParams();
    helper.activationType_ = QMMAQ::ActivationAlg::SWIGLU;
    helper.aicoreParams_.aicNum = 32UL;
    helper.inputParams_.mSize = isAFullLoad ? 256UL : 258UL;
    helper.inputParams_.nSize = isAFullLoad ? 4352UL : 256UL;
    helper.inputParams_.transA = false;
    helper.inputParams_.bFormat = ge::FORMAT_ND;
    helper.inputParams_.aDtype = ge::DT_FLOAT8_E4M3FN;
    helper.inputParams_.bDtype = ge::DT_FLOAT8_E4M3FN;
    helper.isAFullLoad_ = isAFullLoad;
    helper.adaptiveWin_.useTailWinLogic = true;
    helper.adaptiveWin_.baseM = 128UL;
    helper.adaptiveWin_.baseN = 256UL;
    helper.adaptiveWin_.mBlockCnt = isAFullLoad ? 2UL : 3UL;
    helper.adaptiveWin_.nBlockCnt = isAFullLoad ? 17UL : 1UL;
    helper.adaptiveWin_.totalBlockCnt = isAFullLoad ? 34UL : 3UL;
    helper.adaptiveWin_.totalWinCnt = isAFullLoad ? 2UL : 1UL;
    helper.adaptiveWin_.tailWinBlockCnt = isAFullLoad ? 2UL : 3UL;
    helper.adaptiveWin_.mTail = isAFullLoad ? 128UL : 2UL;
    helper.adaptiveWin_.nTail = 256UL;
    helper.adaptiveWin_.mTailTile = 7UL;
    helper.adaptiveWin_.nTailTile = 5UL;
    helper.adaptiveWin_.nBaseTailSplitCnt = 4UL;
    helper.adaptiveWin_.nTailMain = 64UL;
}

TEST(QuantMatmulActivationQuantTailSplit, SwiGluNonFullLoadSearchesMOnlyBeforeCoreCount)
{
    QuantMatmulActivationQuantMxHelper helper(nullptr);
    SetupSwiGluTailSplitHelper(helper, false);

    helper.CalcTailRoundBasicBlockSplit();

    // A coupled M/N search would spend cores on N=2 and stop M at 5 for this shape.
    EXPECT_EQ(helper.adaptiveWin_.mTailTile, 8UL);
    EXPECT_EQ(helper.adaptiveWin_.nTailTile, 1UL);
    EXPECT_EQ(helper.adaptiveWin_.nBaseTailSplitCnt, 1UL);
    EXPECT_EQ(helper.adaptiveWin_.nTailMain, 0UL);
    EXPECT_EQ(helper.CalUsedCoreNum(), 24U);
    ResetQuantBatchMatmulV3InputParams();
}

TEST(QuantMatmulActivationQuantTailSplit, SwiGluAFullLoadKeepsTailRoundUnsplit)
{
    QuantMatmulActivationQuantMxHelper helper(nullptr);
    SetupSwiGluTailSplitHelper(helper, true);

    helper.CalcTailRoundBasicBlockSplit();

    EXPECT_EQ(helper.adaptiveWin_.mTailTile, 1UL);
    EXPECT_EQ(helper.adaptiveWin_.nTailTile, 1UL);
    EXPECT_EQ(helper.adaptiveWin_.nBaseTailSplitCnt, 1UL);
    EXPECT_EQ(helper.adaptiveWin_.nTailMain, 0UL);
    EXPECT_EQ(helper.CalUsedCoreNum(), 32U);
    ResetQuantBatchMatmulV3InputParams();
}

static void SetupTilingParse(const string& opType, const string& compileInfoStr, fe::PlatFormInfos& platformInfo,
                             QuantBatchMatmulV3CompileInfo& compileInfo)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";

    platformInfo.Init();

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_NE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo(), nullptr);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    map<string, string> socVersionInfos = {
        {"SoC_version", "Ascend950"}, {"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

class TestQuantMatmulActivationQuantTiling : public testing::TestWithParam<QuantMatmulActivationQuantTilingParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

static uint64_t GetBatchCount(const gert::Shape& shape)
{
    uint64_t count = 1;
    for (size_t dim = 0; dim + 2 < shape.GetDimNum(); ++dim) {
        count *= static_cast<uint64_t>(shape.GetDim(dim));
    }
    return count;
}

static void CheckBatchDims(const std::array<uint32_t, 4>& dims, const gert::Shape& shape)
{
    const size_t batchRank = shape.GetDimNum() - 2;
    ASSERT_LE(batchRank, 4U);
    const size_t leading = 4U - batchRank;
    for (size_t dim = 0; dim < 4U; ++dim) {
        EXPECT_EQ(dims[dim], dim < leading ? 1 : shape.GetDim(dim - leading));
    }
}

TEST_P(TestQuantMatmulActivationQuantTiling, generalTest)
{
    const auto& param = GetParam();
    const string opType = "QuantMatmulActivationQuant";

    gert::StorageShape x1Shape;
    x1Shape.MutableOriginShape() = param.x1Shape;
    x1Shape.MutableStorageShape() = param.x1Shape;

    gert::StorageShape x2Shape;
    x2Shape.MutableOriginShape() = param.x2OriginShape;
    if (param.x2IsNz) {
        x2Shape.MutableStorageShape() = param.x2StorageShape;
    } else {
        x2Shape.MutableStorageShape() = param.x2OriginShape;
    }

    gert::StorageShape x1ScaleShape;
    x1ScaleShape.MutableOriginShape() = param.x1ScaleShape;
    x1ScaleShape.MutableStorageShape() = param.x1ScaleShape;

    gert::StorageShape x2ScaleShape;
    x2ScaleShape.MutableOriginShape() = param.x2ScaleShape;
    x2ScaleShape.MutableStorageShape() = param.x2ScaleShape;

    gert::StorageShape biasShape;
    biasShape.MutableOriginShape() = param.biasShape;
    biasShape.MutableStorageShape() = param.biasShape;

    gert::StorageShape yShape;
    yShape.MutableOriginShape() = param.yShape;
    yShape.MutableStorageShape() = param.yShape;

    gert::StorageShape yScaleShape;
    yScaleShape.MutableOriginShape() = param.yScaleShape;
    yScaleShape.MutableStorageShape() = param.yScaleShape;

    fe::PlatFormInfos platformInfo;
    QuantBatchMatmulV3CompileInfo compileInfo;
    SetupTilingParse(opType, kAscend950CompileInfo, platformInfo, compileInfo);

    auto tilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(tilingData, nullptr);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(kAscend950CompileInfo.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes(
                          {&x1Shape, &x2Shape, param.hasBias ? &biasShape : nullptr, &x1ScaleShape, &x2ScaleShape})
                      .OutputShapes({&yShape, &yScaleShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, param.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.x2Dtype, param.x2IsNz ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND,
                                   param.x2IsNz ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND)
                      .NodeInputTd(2, param.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, param.x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, param.x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX1)},
                           {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX2)},
                           {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groupSize)},
                           {"activation_type", Ops::NN::AnyValue::CreateFrom<string>(param.activationType)},
                           {"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.yDtype))},
                           {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>(param.quantMode)},
                           {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(param.roundMode)},
                           {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(param.scaleAlg)},
                           {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(param.dstTypeMax)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr) << "caseName=" << param.caseName;

    if (param.expectResult) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS) << "caseName=" << param.caseName;
        using namespace QuantMatmulActivationQuantArch35TilingKey;
        const bool withoutBatch = GetBatchCount(param.x1Shape) == 1 && GetBatchCount(param.x2OriginShape) == 1;
        const uint64_t batchMode = withoutBatch ? TPL_WITHOUT_BATCH : TPL_WITH_BATCH;
        auto* rawTiling = tilingContext->GetRawTilingData();
        const auto checkTilingFields = [&param](const auto& data) {
            EXPECT_EQ(static_cast<int64_t>(data.scaleAlg), param.scaleAlg);
            EXPECT_EQ(data.isBias, param.hasBias ? 1U : 0U);
            EXPECT_EQ(data.m, param.yShape.GetDim(param.yShape.GetDimNum() - 2));
            EXPECT_EQ(data.n,
                      param.x2OriginShape.GetDim(param.x2OriginShape.GetDimNum() - (param.transposeX2 ? 2 : 1)));
            EXPECT_EQ(data.k, param.x1Shape.GetDim(param.x1Shape.GetDimNum() - (param.transposeX1 ? 2 : 1)));
            if (param.activationType == "swiglu") {
                EXPECT_EQ(data.activationType, QMMAQ::ActivationAlg::SWIGLU);
                EXPECT_EQ(data.n % 64U, 0U);
                EXPECT_EQ(data.baseN % 128U, 0U);
                EXPECT_LE(static_cast<uint64_t>(data.baseM) * (data.baseN / 2U), 64UL * 256UL);
                EXPECT_EQ(data.nTailTile, 1U);
                EXPECT_EQ(data.nBaseTailSplitCnt, 1U);
                EXPECT_EQ(data.nTailMain, 0U);
                EXPECT_LE(ops::CeilAlign(static_cast<uint64_t>(data.baseM), 2UL) * data.baseN, 2UL * 64UL * 256UL);
                EXPECT_LE(ops::CeilAlign(static_cast<uint64_t>(data.baseK), 64UL) * data.baseN, 65536UL / 2UL);
            } else {
                EXPECT_EQ(data.activationType,
                          param.activationType == "gelu_erf" ? QMMAQ::ActivationAlg::ERF : QMMAQ::ActivationAlg::TANH);
            }
        };
        if (withoutBatch) {
            ASSERT_EQ(rawTiling->GetDataSize(), sizeof(QMMAQ::QMMAQWithoutBatchTilingData));
            const auto* data = static_cast<const QMMAQ::QMMAQWithoutBatchTilingData*>(rawTiling->GetData());
            checkTilingFields(*data);
        } else {
            ASSERT_EQ(rawTiling->GetDataSize(), sizeof(QMMAQ::QMMAQTilingData));
            const auto* batchData = static_cast<const QMMAQ::QMMAQTilingData*>(rawTiling->GetData());
            EXPECT_EQ(batchData->batchCount, GetBatchCount(param.yShape));
            CheckBatchDims({batchData->batchA1, batchData->batchA2, batchData->batchA3, batchData->batchA4},
                           param.x1Shape);
            CheckBatchDims({batchData->batchB1, batchData->batchB2, batchData->batchB3, batchData->batchB4},
                           param.x2OriginShape);
            CheckBatchDims({batchData->batchC1, batchData->batchC2, batchData->batchC3, batchData->batchC4},
                           param.yShape);
            checkTilingFields(*batchData);
        }
        const bool isSwiglu = param.activationType == "swiglu";
        const uint64_t noFullLoadKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(param.transposeX1), static_cast<uint64_t>(param.transposeX2), batchMode,
            static_cast<uint64_t>(isSwiglu ? TPL_SWIGLU_NO_FULLLOAD : TPL_GELU_NO_FULLLOAD));
        const uint64_t fullLoadKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(param.transposeX1), static_cast<uint64_t>(param.transposeX2), batchMode,
            static_cast<uint64_t>(isSwiglu ? TPL_SWIGLU_FULLLOAD : TPL_GELU_FULLLOAD));
        const uint64_t actualTilingKey = tilingContext->GetTilingKey();
        EXPECT_TRUE(actualTilingKey == noFullLoadKey || actualTilingKey == fullLoadKey)
            << "caseName=" << param.caseName
            << " activation/transpose/batch kernel dispatch mismatch, tilingKey=" << actualTilingKey;
        if (param.expectTilingKey >= 0) {
            EXPECT_EQ(static_cast<int64_t>(actualTilingKey), param.expectTilingKey)
                << "caseName=" << param.caseName << " tilingKey mismatch, expected=" << param.expectTilingKey
                << " actual=" << actualTilingKey;
        }
    } else {
        EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED) << "caseName=" << param.caseName;
    }
}

TEST(QuantMatmulActivationQuantTilingCsv, ShouldLoadValidCases)
{
    const auto& loadResult = GetParamsLoadResult();
    for (const auto& error : loadResult.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(loadResult.params.empty());
}

INSTANTIATE_TEST_CASE_P(QuantMatmulActivationQuantTilingCsv, TestQuantMatmulActivationQuantTiling,
                        testing::ValuesIn(GetParams()));
