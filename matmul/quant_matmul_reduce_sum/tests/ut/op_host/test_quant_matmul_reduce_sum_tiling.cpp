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
 * \file test_quant_matmul_reduce_sum_tiling.cpp
 * \brief
 */
#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_string_utils.h"
#include "../../../op_host/op_tiling/quant_matmul_reduce_sum_tiling.h"

using namespace std;
using namespace ge;

class QuantMatmulReduceSumTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QuantMatmulReduceSum SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantMatmulReduceSum TearDown" << std::endl; }
};

struct QuantMatmulReduceSumTilingTestData {
    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape x1ScaleShape;
    gert::StorageShape x2ScaleShape;
    gert::StorageShape outShape;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType x1ScaleDtype;
    ge::DataType x2ScaleDtype;
    ge::DataType outDType;
};

namespace {
using TilingData = QUANT_MATMUL_REDUCE_SUM::QuantMatmulReduceSumTilingData;

QuantMatmulReduceSumTilingTestData ValidInput()
{
    return {{{8, 2048, 1024}, {8, 2048, 1024}},
            {{8, 1024, 7168}, {8, 1024, 7168}},
            {{8, 2048}, {8, 2048}},
            {{7168}, {7168}},
            {{2048, 7168}, {2048, 7168}},
            ge::DT_INT8,
            ge::DT_INT8,
            ge::DT_FLOAT,
            ge::DT_BF16,
            ge::DT_BF16};
}

class QuantMatmulReduceSumTestContext {
public:
    explicit QuantMatmulReduceSumTestContext(const QuantMatmulReduceSumTilingTestData& testCase = ValidInput(),
                                             size_t capacity = sizeof(TilingData), uint64_t l2Size = 33554432,
                                             uint32_t coreNum = 20)
        : data(testCase),
          rawTiling(gert::TilingData::CreateCap(capacity)),
          workspace(gert::ContinuousVector::Create<size_t>(1))
    {
        string compileInfoString = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 1024,
                                            "load3d_constraints": "0",
                                            "Intrinsic_fix_pipe_l0c2out": true,
                                            "Intrinsic_data_move_l12ub": false,
                                            "Intrinsic_data_move_l0c2ub": false,
                                            "Intrinsic_data_move_out2l1_nd2nz": true,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 20
                                        }
                                    })";
        auto hardware = nlohmann::json::parse(compileInfoString);
        hardware["hardware_info"]["L2_SIZE"] = l2Size;
        hardware["hardware_info"]["CORE_NUM"] = coreNum;
        map<string, string> socInfos;
        map<string, string> aicoreSpec;
        map<string, string> intrinsics;
        GetPlatFormInfos(hardware.dump().c_str(), socInfos, aicoreSpec, intrinsics);
        platformInfo.Init();
        platformInfo.SetPlatformRes("SoCInfo", socInfos);
        platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
        platformInfo.SetCoreNumByCoreType("AICore");
        platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    }

    gert::TilingContext* Build()
    {
        gert::StorageShape dimsShape = {{1}, {1}};
        gert::StorageShape biasShape = {{}, {}};
        holder = gert::TilingContextFaker()
                     .NodeIoNum(6, 1)
                     .IrInstanceNum({1, 1, 1, 1, 1, 1})
                     .InputShapes({&data.x1Shape, &data.x2Shape, &dimsShape, &biasShape,
                                   missingScale == 4 ? nullptr : &data.x1ScaleShape,
                                   missingScale == 5 ? nullptr : &data.x2ScaleShape})
                     .OutputShapes({&data.outShape})
                     .CompileInfo(&compileInfo)
                     .PlatformInfo(&platformInfo)
                     .NodeInputTd(0, data.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, data.x2Dtype, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                     .NodeInputTd(4, data.x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, data.x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, data.outDType, ge::FORMAT_ND, ge::FORMAT_ND)
                     .DeterministicInfo(deterministic)
                     .TilingData(rawTiling.get())
                     .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                     .SetOpType("QuantMatmulReduceSum")
                     .Build();
        return holder.GetContext<gert::TilingContext>();
    }

    QuantMatmulReduceSumTilingTestData data;
    int32_t missingScale = -1;
    int32_t deterministic = 0;

private:
    fe::PlatFormInfos platformInfo;
    optiling::QuantMatmulReduceSumCompileInfo compileInfo;
    decltype(gert::TilingData::CreateCap(0)) rawTiling;
    decltype(gert::ContinuousVector::Create<size_t>(0)) workspace;
    gert::KernelRunContextHolder holder;
};

class QuantMatmulReduceSumTilingProbe : public optiling::QuantMatmulReduceSumTiling {
public:
    using QuantMatmulReduceSumTiling::AnalyzeAttrs;
    using QuantMatmulReduceSumTiling::CalUbSize;
    using QuantMatmulReduceSumTiling::CheckContext;
    using QuantMatmulReduceSumTiling::compileInfo_;
    using QuantMatmulReduceSumTiling::DoOpTiling;
    using QuantMatmulReduceSumTiling::GetShapeAttrsInfo;
    using QuantMatmulReduceSumTiling::GetWorkspaceSize;
    using QuantMatmulReduceSumTiling::inputParams_;
    using QuantMatmulReduceSumTiling::PostTiling;
    using QuantMatmulReduceSumTiling::QuantMatmulReduceSumTiling;
    using QuantMatmulReduceSumTiling::SetMatmulTiling;
    using QuantMatmulReduceSumTiling::SetQbmmTiling;
    using QuantMatmulReduceSumTiling::tilingData_;
    using QuantMatmulReduceSumTiling::tilingDataSize_;
    using TilingBaseClass::workspaceSize_;
};
} // namespace

static void TestInputFunc(const QuantMatmulReduceSumTilingTestData& testCase, uint64_t expectKey,
                          ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS, int32_t deterministic = 0)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulReduceSum");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    QuantMatmulReduceSumTestContext holder(testCase);
    holder.deterministic = deterministic;
    auto context = holder.Build();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->tiling(context), expectedStatus);
    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    EXPECT_EQ(context->GetTilingKey(), expectKey);
    ASSERT_EQ(context->GetRawTilingData()->GetDataSize(), sizeof(TilingData));
    TilingData result{};
    std::memcpy(&result, context->GetRawTilingData()->GetData(), sizeof(result));
    EXPECT_EQ(context->GetBlockDim(), result.matmulTiling.usedCoreNum);
    EXPECT_GT(result.matmulTiling.usedCoreNum, 0);
    EXPECT_LE(result.matmulTiling.usedCoreNum, 20);
    EXPECT_EQ(result.qbmmReduceSumParams.batchNum, testCase.x1Shape.GetOriginShape().GetDim(0));
    EXPECT_EQ(result.qbmmReduceSumParams.coreNum, result.matmulTiling.usedCoreNum);
    EXPECT_EQ(result.qbmmReduceSumParams.isPertoken, 1U);
    EXPECT_EQ(result.qbmmReduceSumParams.isDetermine, deterministic);
    EXPECT_GT(result.qbmmReduceSumParams.ubCalSize, 0U);
    EXPECT_EQ(result.qbmmReduceSumParams.ubRestBytes, 16U * result.qbmmReduceSumParams.ubCalSize);
    EXPECT_EQ(
        context->GetWorkspaceSizes(1)[0],
        16777216UL + 16UL * result.matmulTiling.baseM * result.matmulTiling.baseN * result.matmulTiling.usedCoreNum);
}

TEST_F(QuantMatmulReduceSumTilingTest, obj_cases_01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulReduceSum"), nullptr);

    struct QuantMatmulReduceSumTilingTestData testCase = {
        {{8, 2048, 1024}, {8, 2048, 1024}},
        {{8, 1024, 7168}, {8, 1024, 7168}},
        {{8, 2048}, {8, 2048}},
        {{7168}, {7168}},
        {{2048, 7168}, {2048, 7168}},
        ge::DT_INT8,
        ge::DT_INT8,
        ge::DT_FLOAT,
        ge::DT_BF16,
        ge::DT_BF16,
    };

    TestInputFunc(testCase, 0);
}

TEST_F(QuantMatmulReduceSumTilingTest, obj_cases_02)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulReduceSum"), nullptr);

    struct QuantMatmulReduceSumTilingTestData testCase = {
        {{8, 2048, 2048}, {8, 2048, 2048}},
        {{8, 2048, 7168}, {8, 2048, 7168}},
        {{8, 2048}, {8, 2048}},
        {{7168}, {7168}},
        {{2048, 7168}, {2048, 7168}},
        ge::DT_INT8,
        ge::DT_INT8,
        ge::DT_FLOAT,
        ge::DT_BF16,
        ge::DT_BF16,
    };

    TestInputFunc(testCase, 1);
}

namespace {
struct TilingCsvCase {
    std::string name;
    std::vector<std::string> fields;
};

std::vector<TilingCsvCase> LoadTilingCsvCases()
{
    const std::string
        path = ut_str::GetExeDirPath() +
               "../../../../matmul/quant_matmul_reduce_sum/tests/ut/op_host/test_quant_matmul_reduce_sum_tiling.csv";
    std::ifstream input(path);
    if (!input.is_open()) {
        return {{"MissingCsvFile", {path}}};
    }
    std::vector<TilingCsvCase> cases;
    std::set<std::string> names;
    std::string line;
    bool header = true;
    while (std::getline(input, line)) {
        line = ut_str::Trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }
        if (header) {
            header = false;
            continue;
        }
        std::vector<std::string> fields;
        ut_str::SplitStr2Vec(line, ",", fields);
        const auto name = ut_str::Trim(fields.front());
        if (name.empty() ||
            name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
                std::string::npos ||
            !names.insert(name).second) {
            return {{"InvalidOrDuplicateCaseName", {line}}};
        }
        cases.push_back({name, std::move(fields)});
    }
    return cases.empty() ? std::vector<TilingCsvCase>{{"EmptyCsvFile", {path}}} : cases;
}
} // namespace

class QuantMatmulReduceSumTilingCsvTest : public testing::TestWithParam<TilingCsvCase> {};

TEST_P(QuantMatmulReduceSumTilingCsvTest, ValidateInputAndTiling)
{
    const auto& fields = GetParam().fields;
    ASSERT_EQ(fields.size(), 14U) << "Missing or malformed tiling CSV: " << fields.front();
    auto parseShape = [](const std::string& value) {
        std::istringstream stream(value);
        std::vector<int64_t> dims;
        int64_t dim = 0;
        while (stream >> dim) {
            dims.push_back(dim);
        }
        EXPECT_TRUE(stream.eof()) << "Invalid shape: " << value;
        return ut_str::MakeStorageShape(dims);
    };
    auto parseDtype = [](const std::string& value) {
        auto dtype = ut_str::ParseDtype(value, ge::DT_UNDEFINED);
        EXPECT_NE(dtype, ge::DT_UNDEFINED) << "Invalid dtype: " << value;
        return dtype;
    };
    QuantMatmulReduceSumTilingTestData data = {parseShape(fields[1]), parseShape(fields[2]), parseShape(fields[3]),
                                               parseShape(fields[4]), parseShape(fields[5]), parseDtype(fields[6]),
                                               parseDtype(fields[7]), parseDtype(fields[8]), parseDtype(fields[9]),
                                               parseDtype(fields[10])};
    const auto status = ut_str::Trim(fields[11]);
    ASSERT_TRUE(status == "GRAPH_SUCCESS" || status == "GRAPH_FAILED");
    TestInputFunc(data, std::stoull(fields[12]), ut_str::ParseGraphStatus(status), std::stoi(fields[13]));
}

INSTANTIATE_TEST_SUITE_P(QuantMatmulReduceSum, QuantMatmulReduceSumTilingCsvTest,
                         testing::ValuesIn(LoadTilingCsvCases()),
                         [](const testing::TestParamInfo<TilingCsvCase>& info) { return info.param.name; });

class QuantMatmulReduceSumTilingStateTest : public testing::Test {
protected:
    void SetUp() override
    {
        context = holder.Build();
        ASSERT_NE(context, nullptr);
        tiling = std::make_unique<QuantMatmulReduceSumTilingProbe>(context);
        tiling->tilingData_ = {};
    }

    QuantMatmulReduceSumTestContext holder;
    gert::TilingContext* context = nullptr;
    std::unique_ptr<QuantMatmulReduceSumTilingProbe> tiling;
};

TEST_F(QuantMatmulReduceSumTilingStateTest, ReadsPlatformMemoryAndCoreCount)
{
    EXPECT_EQ(tiling->compileInfo_.ubSize, 196608U);
    EXPECT_EQ(tiling->compileInfo_.l1Size, 524288U);
    EXPECT_EQ(tiling->compileInfo_.l2Size, 33554432U);
    EXPECT_EQ(tiling->compileInfo_.l0ASize, 65536U);
    EXPECT_EQ(tiling->compileInfo_.l0BSize, 65536U);
    EXPECT_EQ(tiling->compileInfo_.l0CSize, 131072U);
    EXPECT_EQ(tiling->compileInfo_.aicNum, 20U);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, CorrectsReportedL2Size)
{
    QuantMatmulReduceSumTestContext corrected(ValidInput(), sizeof(TilingData), 96UL * 1024 * 1024);
    QuantMatmulReduceSumTilingProbe probe(corrected.Build());
    EXPECT_EQ(probe.compileInfo_.l2Size, 168UL * 1024 * 1024);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, PlatformWithNoCoresDoesNotInventCoreCount)
{
    QuantMatmulReduceSumTestContext empty(ValidInput(), sizeof(TilingData), 33554432, 0);
    QuantMatmulReduceSumTilingProbe probe(empty.Build());
    EXPECT_EQ(probe.compileInfo_.aicNum, 0U);
    EXPECT_EQ(probe.compileInfo_.ubSize, 196608U);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, ReusesParsedShapeInformation)
{
    ASSERT_EQ(tiling->GetShapeAttrsInfo(), ge::GRAPH_SUCCESS);
    ASSERT_TRUE(tiling->inputParams_.initFlag);
    EXPECT_EQ(tiling->inputParams_.batchNum, 8);
    EXPECT_EQ(tiling->inputParams_.mSize, 2048);
    EXPECT_EQ(tiling->inputParams_.kSize, 1024);
    EXPECT_EQ(tiling->inputParams_.nSize, 7168);
    EXPECT_TRUE(tiling->inputParams_.isPertoken);
    // The context is consulted only on the first parse; later calls retain the cached values.
    tiling->inputParams_.mSize = 64;
    ASSERT_EQ(tiling->GetShapeAttrsInfo(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->inputParams_.mSize, 64);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, AttributesResetTransposeFlags)
{
    tiling->inputParams_.transA = true;
    tiling->inputParams_.transB = true;
    EXPECT_EQ(tiling->AnalyzeAttrs(), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(tiling->inputParams_.transA);
    EXPECT_FALSE(tiling->inputParams_.transB);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, RejectsMissingQuantizationScales)
{
    for (int32_t index : {4, 5}) {
        SCOPED_TRACE(index);
        QuantMatmulReduceSumTestContext missing;
        missing.missingScale = index;
        auto missingContext = missing.Build();
        ASSERT_NE(missingContext, nullptr);
        ASSERT_EQ(missingContext->GetOptionalInputShape(index), nullptr);
        QuantMatmulReduceSumTilingProbe probe(missingContext);
        EXPECT_EQ(probe.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
        EXPECT_FALSE(probe.inputParams_.initFlag);
    }
}

TEST_F(QuantMatmulReduceSumTilingStateTest, RequiresFullTilingDataCapacity)
{
    tiling->tilingDataSize_ = sizeof(TilingData);
    EXPECT_EQ(tiling->CheckContext(), ge::GRAPH_SUCCESS);
    QuantMatmulReduceSumTestContext small(ValidInput(), sizeof(TilingData) - 1);
    QuantMatmulReduceSumTilingProbe probe(small.Build());
    EXPECT_EQ(probe.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
    EXPECT_FALSE(probe.inputParams_.initFlag);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, UbCapacityBoundaryReservesScaleBuffers)
{
    tiling->inputParams_.baseM = 16;
    tiling->inputParams_.baseN = 32;
    // Two FP32 M scales and two BF16 N scales occupy 256 bytes. One UB tile needs 26 * 32 more.
    tiling->compileInfo_.ubSize = 256 + 26 * 32;
    ASSERT_EQ(tiling->CalUbSize(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->tilingData_.qbmmReduceSumParams.ubCalSize, 32U);
    EXPECT_EQ(tiling->tilingData_.qbmmReduceSumParams.ubRestBytes, 512U);
    for (uint64_t capacity : {1087UL, 256UL, 0UL}) {
        SCOPED_TRACE(capacity);
        tiling->compileInfo_.ubSize = capacity;
        EXPECT_EQ(tiling->CalUbSize(), ge::GRAPH_FAILED);
        EXPECT_EQ(tiling->tilingData_.qbmmReduceSumParams.ubCalSize, 32U);
        EXPECT_EQ(tiling->tilingData_.qbmmReduceSumParams.ubRestBytes, 512U);
    }
}

TEST_F(QuantMatmulReduceSumTilingStateTest, UbCalculationHonorsTileAndAlignmentLimits)
{
    tiling->inputParams_.baseM = 16;
    tiling->inputParams_.baseN = 32;
    ASSERT_EQ(tiling->CalUbSize(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->tilingData_.qbmmReduceSumParams.ubCalSize, 512U);
    // Space for 63 elements must round down to one aligned 32-element tile.
    tiling->compileInfo_.ubSize = 256 + 26 * 63;
    ASSERT_EQ(tiling->CalUbSize(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->tilingData_.qbmmReduceSumParams.ubCalSize, 32U);
    tiling->inputParams_.baseM = 0;
    EXPECT_EQ(tiling->CalUbSize(), ge::GRAPH_FAILED);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, QbmmFieldsCarryDeterminismAndTileDimensions)
{
    for (int32_t deterministic : {0, 1}) {
        SCOPED_TRACE(deterministic);
        QuantMatmulReduceSumTestContext configured;
        configured.deterministic = deterministic;
        QuantMatmulReduceSumTilingProbe probe(configured.Build());
        probe.tilingData_ = {};
        probe.inputParams_.batchNum = 3;
        probe.inputParams_.usedCoreNum = 7;
        probe.inputParams_.baseM = 16;
        probe.inputParams_.baseN = 32;
        probe.inputParams_.baseK = 128;
        probe.inputParams_.isPertoken = true;
        ASSERT_TRUE(probe.SetQbmmTiling());
        const auto& params = probe.tilingData_.qbmmReduceSumParams;
        EXPECT_EQ(params.batchNum, 3U);
        EXPECT_EQ(params.coreNum, 7U);
        EXPECT_EQ(params.ubBaseK, 128U);
        EXPECT_EQ(params.ubBaseN, 32U);
        EXPECT_EQ(params.isPertoken, 1U);
        EXPECT_EQ(params.isDetermine, deterministic);
    }
}

TEST_F(QuantMatmulReduceSumTilingStateTest, L1CapacityBoundaryForDeepBuffers)
{
    ASSERT_EQ(tiling->GetShapeAttrsInfo(), ge::GRAPH_SUCCESS);
    tiling->compileInfo_.l1Size = 393216;
    ASSERT_TRUE(tiling->SetMatmulTiling());
    const auto& matmul = tiling->tilingData_.matmulTiling;
    ASSERT_EQ(matmul.baseM, 128);
    ASSERT_EQ(matmul.baseN, 256);
    ASSERT_EQ(matmul.baseK, 128);
    EXPECT_EQ(matmul.depthA1, 8);
    EXPECT_EQ(matmul.depthB1, 8);
    EXPECT_EQ(matmul.stepKa, 4);
    EXPECT_EQ(matmul.stepKb, 4);
    EXPECT_EQ(matmul.shareL1Size, 393216U);
    tiling->compileInfo_.l1Size = 393215;
    EXPECT_EQ(tiling->DoOpTiling(), ge::GRAPH_FAILED);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, PropagatesMatmulApiFailure)
{
    ASSERT_EQ(tiling->GetShapeAttrsInfo(), ge::GRAPH_SUCCESS);
    tiling->inputParams_.mSize = 0;
    EXPECT_EQ(tiling->DoOpTiling(), ge::GRAPH_FAILED);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, PropagatesInsufficientUbFailure)
{
    ASSERT_EQ(tiling->GetShapeAttrsInfo(), ge::GRAPH_SUCCESS);
    tiling->compileInfo_.ubSize = 0;
    EXPECT_EQ(tiling->DoOpTiling(), ge::GRAPH_FAILED);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, WritesTilingDataAndWorkspaceAtExactCapacity)
{
    tiling->inputParams_.baseM = 128;
    tiling->inputParams_.baseN = 256;
    tiling->inputParams_.usedCoreNum = 20;
    ASSERT_EQ(tiling->GetWorkspaceSize(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->workspaceSize_, 27262976U);
    tiling->tilingDataSize_ = sizeof(TilingData);
    tiling->tilingData_.matmulTiling.usedCoreNum = 20;
    tiling->tilingData_.qbmmReduceSumParams.batchNum = 8;
    ASSERT_EQ(tiling->PostTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetBlockDim(), 20U);
    EXPECT_EQ(context->GetScheduleMode(), 1U);
    EXPECT_EQ(context->GetWorkspaceSizes(1)[0], 27262976U);
    EXPECT_EQ(context->GetRawTilingData()->GetDataSize(), sizeof(TilingData));
    EXPECT_EQ(std::memcmp(context->GetRawTilingData()->GetData(), &tiling->tilingData_, sizeof(TilingData)), 0);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, FailedCopyDoesNotPublishTilingOrWorkspace)
{
    QuantMatmulReduceSumTestContext small(ValidInput(), sizeof(TilingData) - 1);
    auto smallContext = small.Build();
    QuantMatmulReduceSumTilingProbe probe(smallContext);
    probe.tilingData_ = {};
    probe.tilingDataSize_ = sizeof(TilingData);
    probe.workspaceSize_ = 1234;
    smallContext->GetRawTilingData()->SetDataSize(0);
    smallContext->GetWorkspaceSizes(1)[0] = 5678;
    EXPECT_EQ(probe.PostTiling(), ge::GRAPH_FAILED);
    EXPECT_EQ(smallContext->GetRawTilingData()->GetDataSize(), 0U);
    EXPECT_EQ(smallContext->GetWorkspaceSizes(1)[0], 5678U);
}

TEST_F(QuantMatmulReduceSumTilingStateTest, RegisteredCallbacksValidateContext)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulReduceSum");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);
    EXPECT_EQ(impl->tiling(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(impl->tiling_parse(nullptr), ge::GRAPH_FAILED);
    optiling::QuantMatmulReduceSumCompileInfo compileInfo;
    char compileJson[] = "{}";
    auto parseHolder = gert::TilingParseContextFaker()
                           .SetOpType("QuantMatmulReduceSum")
                           .KernelIONum(2, 1)
                           .Inputs({compileJson, context->GetPlatformInfo()})
                           .Outputs({&compileInfo})
                           .Build();
    auto parseContext = parseHolder.GetContext<gert::KernelContext>();
    ASSERT_NE(parseContext, nullptr);
    EXPECT_EQ(impl->tiling_parse(parseContext), ge::GRAPH_SUCCESS);
}
