/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <gtest/gtest.h>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/foreach_maximum_scalar_tiling_data.h"
#include "../../../../op_kernel/arch35/foreach_maximum_scalar_tiling_key.h"

using namespace ut_util;

class ForeachMaximumScalarTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachMaximumScalarTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachMaximumScalarTilingTest TearDown" << std::endl; }
};

struct TilingCase {
    gert::StorageShape xShape = {{4, 4}, {4, 4}};
    gert::StorageShape yShape = {{4, 4}, {4, 4}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ge::DataType inputDt = ge::DT_FLOAT;
    ge::DataType outputDt = ge::DT_FLOAT;
    ge::DataType scalarDt = ge::DT_FLOAT;
    ge::Format inputFormat = ge::FORMAT_ND;
    ge::Format outputFormat = ge::FORMAT_ND;
    ge::Format scalarFormat = ge::FORMAT_ND;
    ge::Format inputOriginFormat = ge::FORMAT_ND;
    ge::Format outputOriginFormat = ge::FORMAT_ND;
    ge::Format scalarOriginFormat = ge::FORMAT_ND;
    uint32_t tensorCount = 1;
    uint32_t outputCount = 1;
    uint32_t scalarCount = 1;
    uint64_t ubSize = 253952;
    uint32_t coreCount = 64;
};

static void RunCase(TilingCase test, ge::graphStatus expectedStatus, uint64_t expectKey = 0,
                    const char* expectedError = nullptr, const char* expectedParam = nullptr,
                    const std::map<std::string, std::string>& expectedArgs = {})
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    ai["ub_size"] = std::to_string(test.ubSize);
    soc["ai_core_cnt"] = std::to_string(test.coreCount);
    pf.Init();

    struct ForeachMaximumScalarCompileInfo {
    } ci;
    auto opType = std::string("ForeachMaximumScalar");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::vector<void*> inputShapes(test.tensorCount, &test.xShape);
    if (test.scalarCount != 0) {
        inputShapes.push_back(&test.scalarShape);
    }
    std::vector<gert::StorageShape*> outputShapes(test.outputCount, &test.yShape);
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    gert::TilingContextFaker faker;
    faker.NodeIoNum(2, 1)
        .IrInstanceNum({test.tensorCount, test.scalarCount}, {test.outputCount})
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&ci)
        .PlatformInfo(reinterpret_cast<char*>(&pf))
        .NodeInputTd(0, test.inputDt, test.inputOriginFormat, test.inputFormat)
        .TilingData(param.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()));
    if (test.scalarCount != 0) {
        faker.NodeInputTd(1, test.scalarDt, test.scalarOriginFormat, test.scalarFormat);
    }
    for (uint32_t i = 0; i < test.outputCount; ++i) {
        faker.NodeOutputTd(i, test.outputDt, test.outputOriginFormat, test.outputFormat);
    }
    auto th = faker.Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    auto& errorManager = ErrorManager::GetInstance();
    errorManager.GenWorkStreamIdDefault();
    (void)errorManager.GetRawErrorMessages();
    ASSERT_EQ(tilingFn(ctx), expectedStatus);
    const auto errors = errorManager.GetRawErrorMessages();
    if (expectedError != nullptr) {
        ASSERT_EQ(errors.size(), 1U);
        EXPECT_EQ(errors[0].error_id, expectedError);
        if (expectedParam != nullptr) {
            const char* paramKey = errors[0].args_map.count("param_name") != 0 ? "param_name" : "param_names";
            ASSERT_NE(errors[0].args_map.count(paramKey), 0U);
            EXPECT_EQ(errors[0].args_map.at(paramKey), expectedParam);
        }
        for (const auto& [key, value] : expectedArgs) {
            ASSERT_NE(errors[0].args_map.count(key), 0U) << key;
            EXPECT_EQ(errors[0].args_map.at(key), value) << key;
        }
    }
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        EXPECT_TRUE(errors.empty());
        EXPECT_EQ(ctx->GetTilingKey(), expectKey);
        EXPECT_GT(ctx->GetBlockDim(), 0U);
        auto* tiling = ctx->GetTilingData<ForeachMaximumScalarTilingData>();
        ASSERT_NE(tiling, nullptr);
        EXPECT_EQ(tiling->tensorCount, test.tensorCount);
        EXPECT_EQ(tiling->totalElements, test.xShape.GetStorageShape().GetShapeSize() * test.tensorCount);
    }
}

static void DoCase(std::initializer_list<int64_t> xShape, ge::DataType inputDt, uint64_t expectKey)
{
    TilingCase test;
    test.xShape = {xShape, xShape};
    test.yShape = {xShape, xShape};
    test.inputDt = inputDt;
    test.outputDt = inputDt;
    test.scalarDt = inputDt == ge::DT_BF16 ? ge::DT_FLOAT : inputDt;
    RunCase(test, ge::GRAPH_SUCCESS, expectKey);
}

TEST_F(ForeachMaximumScalarTilingTest, foreach_maximum_scalar_float32)
{
    DoCase({4, 4}, ge::DT_FLOAT, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
}

TEST_F(ForeachMaximumScalarTilingTest, foreach_maximum_scalar_float16)
{
    DoCase({4, 4}, ge::DT_FLOAT16, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT16));
}

TEST_F(ForeachMaximumScalarTilingTest, foreach_maximum_scalar_int32)
{
    DoCase({4, 4}, ge::DT_INT32, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_INT32));
}

TEST_F(ForeachMaximumScalarTilingTest, foreach_maximum_scalar_bfloat16)
{
    DoCase({4, 4}, ge::DT_BF16, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_BF16));
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_scalar_without_one_element)
{
    for (int64_t count : {0, 2}) {
        TilingCase test;
        test.scalarShape = {{count}, {count}};
        RunCase(test, ge::GRAPH_FAILED, 0, "EZ0014", "scalar",
                {{"incorrect_size", std::to_string(count)}, {"correct_size", "1"}});
    }
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_missing_scalar)
{
    TilingCase test;
    test.scalarCount = 0;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0023", "scalar", {{"incorrect_nums", "0"}});
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_mismatched_output_count)
{
    TilingCase test;
    test.tensorCount = 2;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0023", "x, y", {{"incorrect_nums", "2, 1"}});
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_missing_output)
{
    TilingCase test;
    test.outputCount = 0;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0023", "x, y", {{"incorrect_nums", "1, 0"}});
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_wrong_scalar_dtype)
{
    TilingCase test;
    test.scalarDt = ge::DT_FLOAT16;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0019", "scalar");
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_wrong_output_dtype)
{
    TilingCase test;
    test.outputDt = ge::DT_INT32;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0019", "y[0]");
}

TEST_F(ForeachMaximumScalarTilingTest, accepts_non_nd_storage_formats)
{
    for (auto format : {ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_HWCN, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC}) {
        SCOPED_TRACE(static_cast<int>(format));
        TilingCase test;
        test.inputFormat = format;
        RunCase(test, ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
        test = TilingCase{};
        test.scalarFormat = format;
        RunCase(test, ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
        test = TilingCase{};
        test.outputFormat = format;
        RunCase(test, ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
    }
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_small_output)
{
    TilingCase test;
    test.yShape = {{1}, {1}};
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0014", "y[0]", {{"incorrect_size", "1"}, {"correct_size", ">= 16"}});
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_shape_size_overflow)
{
    TilingCase test;
    test.xShape = {{std::numeric_limits<int64_t>::max(), 2}, {std::numeric_limits<int64_t>::max(), 2}};
    test.yShape = test.xShape;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0009", "x[0]");
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_total_elements_overflow)
{
    TilingCase test;
    test.tensorCount = test.outputCount = 2;
    test.xShape = {{int64_t{1} << 62}, {int64_t{1} << 62}};
    test.yShape = test.xShape;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0016");
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_negative_dimension)
{
    TilingCase test;
    test.xShape = {{-1}, {-1}};
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0009", "x[0]");
}

TEST_F(ForeachMaximumScalarTilingTest, accepts_rank_zero_scalar_and_empty_members)
{
    TilingCase test;
    test.scalarShape = {{}, {}};
    test.xShape = {{0, 4}, {0, 4}};
    test.yShape = test.xShape;
    RunCase(test, ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
}

TEST_F(ForeachMaximumScalarTilingTest, checks_tensorlist_limit)
{
    TilingCase test;
    test.tensorCount = test.outputCount = 256;
    RunCase(test, ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
    test.tensorCount = test.outputCount = 257;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0023", "x");
}

TEST_F(ForeachMaximumScalarTilingTest, accepts_non_nd_origin_formats)
{
    for (auto format : {ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_HWCN, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC}) {
        SCOPED_TRACE(static_cast<int>(format));
        TilingCase test;
        test.inputOriginFormat = format;
        RunCase(test, ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
        test = TilingCase{};
        test.scalarOriginFormat = format;
        RunCase(test, ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
        test = TilingCase{};
        test.outputOriginFormat = format;
        RunCase(test, ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(FOREACH_MAXIMUM_SCALAR_TPL_SCH_MODE_FLOAT));
    }
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_excessive_rank)
{
    TilingCase test;
    test.xShape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0011", "x[0]", {{"incorrect_dim", "9"}, {"correct_dim", "[0, 8]"}});
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_unsupported_input_dtype)
{
    TilingCase test;
    test.inputDt = test.outputDt = test.scalarDt = ge::DT_DOUBLE;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0019", "x[0]");
}

TEST_F(ForeachMaximumScalarTilingTest, rejects_empty_tensorlist)
{
    TilingCase test;
    test.tensorCount = test.outputCount = 0;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ0023", "x");
}

TEST_F(ForeachMaximumScalarTilingTest, reports_invalid_platform_resources)
{
    TilingCase test;
    test.coreCount = 0;
    RunCase(test, ge::GRAPH_FAILED, 0, "EZ1008", nullptr, {{"reason", "Available AIV core count is 0"}});
    test = TilingCase{};
    for (uint64_t ubSize : {uint64_t{0}, uint64_t{128 * 1024}, uint64_t{1} << 33}) {
        test.ubSize = ubSize;
        RunCase(test, ge::GRAPH_FAILED, 0, "EZ1008", nullptr,
                {{"reason", "Invalid UB size " + std::to_string(ubSize) +
                                ": must exceed reserved size 131072 and leave at most UINT32_MAX bytes available"}});
    }
}
