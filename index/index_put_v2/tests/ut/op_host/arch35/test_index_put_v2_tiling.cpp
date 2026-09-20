/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_index_put_v2_tiling.cpp
 * \brief
 */
#include <iostream>
#include <fstream>
#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../../index/op_host/arch35/index_tiling.h"
#include "../../../../op_host/arch35/index_put_v2_simd_tiling.h"
#include "any_value.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class IndexPutV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IndexPutV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "IndexPutV2Tiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }
    return result;
}

template <typename T>
void SetConstInput(size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
                   std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor({{data_size}, {data_size}},         // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        dtype,                              // dt
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

namespace {
constexpr uint64_t SIMD_TILING_KEY = 131072;
constexpr uint64_t SIMD_ACCUMULATE_TILING_KEY = 655360;
constexpr size_t EXPECTED_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int32_t DEFAULT_CORE_NUM = 64;
constexpr uint64_t DEFAULT_UB_SIZE = 196608;
constexpr const char* DEFAULT_COMPILE_INFO = R"({
    "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
    }
})";

struct SimdTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = std::numeric_limits<uint64_t>::max();
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    bool hasTilingData = false;
    IndexPutV2::IndexPutV2SimdTilingData tilingData{};
};

SimdTilingResult RunSimdTiling(std::initializer_list<int64_t> xDims, std::initializer_list<int64_t> valueDims,
                               const std::vector<int64_t>& indexedSizes, int64_t indexLength, ge::DataType valueDtype,
                               bool accumulate, bool provideTilingData = true, bool useRegisteredEntry = false,
                               bool provideCompileInfo = true)
{
    SimdTilingResult result;
    optiling::IndexCompileInfo compileInfo{};
    compileInfo.core_num = DEFAULT_CORE_NUM;
    compileInfo.ubSize = DEFAULT_UB_SIZE;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(DEFAULT_COMPILE_INFO, socInfos, aicoreSpec, intrinsics);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    auto rawTilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (rawTilingData == nullptr || workspace == nullptr) {
        return result;
    }

    gert::StorageShape x(xDims, xDims);
    gert::StorageShape value(valueDims, valueDims);
    gert::StorageShape indexedSizesShape({static_cast<int64_t>(indexedSizes.size())},
                                         {static_cast<int64_t>(indexedSizes.size())});
    gert::StorageShape indexedStridesShape({static_cast<int64_t>(indexedSizes.size())},
                                           {static_cast<int64_t>(indexedSizes.size())});
    gert::StorageShape indices({indexLength}, {indexLength});
    gert::StorageShape y(xDims, xDims);

    std::vector<int64_t> mutableIndexedSizes = indexedSizes;
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> constTensors;
    SetConstInput(2, ge::DT_INT64, mutableIndexedSizes.data(), static_cast<int64_t>(mutableIndexedSizes.size()),
                  constTensors);

    gert::TilingContextFaker faker;
    faker.SetOpType("IndexPutV2")
        .NodeIoNum(5, 1)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&x, &value, &indexedSizesShape, &indexedStridesShape, &indices})
        .OutputShapes({&y})
        .NodeInputTd(0, valueDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, valueDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, valueDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"accumulate", Ops::NN::AnyValue::CreateFrom<bool>(accumulate)}})
        .ConstInput(constTensors)
        .TilingData(provideTilingData ? rawTilingData.get() : nullptr)
        .Workspace(workspace)
        .CompileInfo(provideCompileInfo ? &compileInfo : nullptr)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo));
    auto holder = faker.Build();
    auto context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return result;
    }
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    if (useRegisteredEntry) {
        auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexPutV2");
        if (opImpl == nullptr || opImpl->tiling == nullptr) {
            return result;
        }
        result.status = opImpl->tiling(context);
    } else {
        optiling::IndexPutV2SimdTiling tiling(context);
        result.status = tiling.DoTiling();
    }
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }

    result.tilingKey = context->GetTilingKey();
    result.blockDim = context->GetBlockDim();
    auto workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        result.workspaceSize = workspaceSizes[0];
    }
    auto raw = context->GetRawTilingData();
    if (raw != nullptr && raw->GetData() != nullptr &&
        raw->GetDataSize() >= sizeof(IndexPutV2::IndexPutV2SimdTilingData)) {
        result.tilingData = *reinterpret_cast<const IndexPutV2::IndexPutV2SimdTilingData*>(raw->GetData());
        result.hasTilingData = true;
    }
    return result;
}

struct PlatformResources {
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
};

ge::graphStatus RunTilingParse(const string& compileInfoString, optiling::IndexCompileInfo* compileInfo)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexPutV2");
    if (opImpl == nullptr || opImpl->tiling_parse == nullptr) {
        return ge::GRAPH_FAILED;
    }

    PlatformResources resources;
    GetPlatFormInfos(compileInfoString.c_str(), resources.socInfos, resources.aicoreSpec, resources.intrinsics);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto holder = gert::KernelRunContextFaker()
                      .KernelIONum(2, 1)
                      .Inputs({const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                      .Outputs({compileInfo})
                      .Build();
    auto parseContext = holder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || parseContext->GetPlatformInfo() == nullptr ||
        !parseContext->GetPlatformInfo()->Init()) {
        return ge::GRAPH_FAILED;
    }
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", resources.socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", resources.aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", resources.intrinsics);
    return opImpl->tiling_parse(holder.GetContext<gert::KernelContext>());
}

string MakeCompileInfoString(int32_t coreNum, uint64_t ubSize)
{
    return "{\"hardware_info\":{\"BT_SIZE\":0,\"load3d_constraints\":\"1\","
           "\"Intrinsic_fix_pipe_l0c2out\":false,\"Intrinsic_data_move_l12ub\":true,"
           "\"Intrinsic_data_move_l0c2ub\":true,\"Intrinsic_data_move_out2l1_nd2nz\":false,"
           "\"UB_SIZE\":" +
           std::to_string(ubSize) +
           ",\"L2_SIZE\":33554432,\"L1_SIZE\":524288,\"L0A_SIZE\":65536,\"L0B_SIZE\":65536,"
           "\"L0C_SIZE\":131072,\"CORE_NUM\":" +
           std::to_string(coreNum) + "}}";
}
} // namespace

TEST_F(IndexPutV2Tiling, SimdAccumulateUsesRowTilingAndFullColumnUbTile)
{
    auto result = RunSimdTiling({2048, 128}, {1024, 128}, {1, 0}, 1024, ge::DT_FLOAT16, true);

    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasTilingData);
    EXPECT_EQ(result.tilingKey, SIMD_ACCUMULATE_TILING_KEY);
    EXPECT_EQ(result.workspaceSize, EXPECTED_WORKSPACE_SIZE);
    EXPECT_EQ(result.blockDim, result.tilingData.coreNum);
    EXPECT_EQ(result.tilingData.inputLength, 2048 * 128);
    EXPECT_EQ(result.tilingData.valueLength, 1024 * 128);
    EXPECT_EQ(result.tilingData.indexedLength, 1024);
    EXPECT_EQ(result.tilingData.nonIndexedLength, 128);
    EXPECT_EQ(result.tilingData.indexedDimNum, 1);
    EXPECT_EQ(result.tilingData.nonIndexedDimNum, 1);
    EXPECT_EQ(result.tilingData.accumulateMode, 1);
    EXPECT_EQ(result.tilingData.blockNumInCol, 1);
    EXPECT_EQ(result.tilingData.colsFactor, result.tilingData.normalCoreColsNum);
    EXPECT_EQ(result.tilingData.inputShapes[0], 2048);
    EXPECT_EQ(result.tilingData.inputShapes[1], 128);
    EXPECT_EQ(result.tilingData.indexedStrides[0], 128);
    EXPECT_EQ(result.tilingData.indexedStrides[1], 1);
}

TEST_F(IndexPutV2Tiling, SimdNonAccumulateSplitsLargeColumnInUb)
{
    auto result = RunSimdTiling({1, 1000000}, {1, 1000000}, {1, 0}, 1, ge::DT_INT64, false);

    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasTilingData);
    EXPECT_EQ(result.tilingKey, SIMD_TILING_KEY);
    EXPECT_EQ(result.blockDim, result.tilingData.coreNum);
    EXPECT_EQ(result.tilingData.accumulateMode, 0);
    EXPECT_EQ(result.tilingData.blockNumInRow, 1);
    EXPECT_GT(result.tilingData.blockNumInCol, 1);
    EXPECT_EQ(result.tilingData.rowsFactor, 1);
    EXPECT_GT(result.tilingData.normalCoreColsNum, result.tilingData.colsFactor);
    EXPECT_GT(result.tilingData.colsFactor, 0);
}

TEST_F(IndexPutV2Tiling, SimdSupportsMultipleTrailingNonIndexedDimensions)
{
    auto result = RunSimdTiling({16, 32, 32}, {8, 32, 32}, {1, 0, 0}, 8, ge::DT_FLOAT, false);

    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasTilingData);
    EXPECT_EQ(result.tilingKey, SIMD_TILING_KEY);
    EXPECT_EQ(result.tilingData.indexedSizesNum, 3);
    EXPECT_EQ(result.tilingData.indexedDimNum, 1);
    EXPECT_EQ(result.tilingData.nonIndexedDimNum, 2);
    EXPECT_EQ(result.tilingData.nonIndexedLength, 32 * 32);
    EXPECT_EQ(result.tilingData.indexedStrides[0], 32 * 32);
    EXPECT_EQ(result.tilingData.indexedStrides[1], 32);
    EXPECT_EQ(result.tilingData.indexedStrides[2], 1);
}

TEST_F(IndexPutV2Tiling, SimdRejectsNonContinuousIndexedSizePatterns)
{
    auto nonOneBeforeZero = RunSimdTiling({8, 128}, {4, 128}, {2, 0}, 4, ge::DT_FLOAT16, false);
    auto oneAfterZero = RunSimdTiling({8, 128, 2}, {4, 128, 2}, {1, 0, 1}, 4, ge::DT_FLOAT16, false);

    EXPECT_EQ(nonOneBeforeZero.status, ge::GRAPH_PARAM_INVALID);
    EXPECT_EQ(oneAfterZero.status, ge::GRAPH_PARAM_INVALID);
}

TEST_F(IndexPutV2Tiling, SimdRejectsSmallNonIndexedRegion)
{
    auto allDimensionsIndexed = RunSimdTiling({8, 32}, {4}, {1, 1}, 4, ge::DT_FLOAT, false);
    auto lessThan256Bytes = RunSimdTiling({8, 32}, {4, 32}, {1, 0}, 4, ge::DT_FLOAT, false);

    EXPECT_EQ(allDimensionsIndexed.status, ge::GRAPH_PARAM_INVALID);
    EXPECT_EQ(lessThan256Bytes.status, ge::GRAPH_PARAM_INVALID);
}

TEST_F(IndexPutV2Tiling, SimdChecksDtypeForAccumulateMode)
{
    auto unsupportedAtomicAdd = RunSimdTiling({8, 32}, {4, 32}, {1, 0}, 4, ge::DT_INT64, true);
    auto unsupportedValueType = RunSimdTiling({8, 32}, {4, 32}, {1, 0}, 4, ge::DT_DOUBLE, false);

    EXPECT_EQ(unsupportedAtomicAdd.status, ge::GRAPH_PARAM_INVALID);
    EXPECT_EQ(unsupportedValueType.status, ge::GRAPH_PARAM_INVALID);
}

TEST_F(IndexPutV2Tiling, SimdFailsWhenTilingBufferIsMissing)
{
    auto result = RunSimdTiling({8, 128}, {4, 128}, {1, 0}, 4, ge::DT_FLOAT16, false, false);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(IndexPutV2Tiling, RegisteredTilingRejectsMissingCompileInfo)
{
    auto result = RunSimdTiling({8, 128}, {4, 128}, {1, 0}, 4, ge::DT_FLOAT16, false, true, true, false);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(IndexPutV2Tiling, TilingParsePopulatesCompileInfo)
{
    optiling::IndexCompileInfo compileInfo{};
    auto status = RunTilingParse(MakeCompileInfoString(DEFAULT_CORE_NUM, DEFAULT_UB_SIZE), &compileInfo);

    ASSERT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(compileInfo.core_num, DEFAULT_CORE_NUM);
    EXPECT_EQ(compileInfo.ubSize, DEFAULT_UB_SIZE);
}

TEST_F(IndexPutV2Tiling, TilingParseRejectsInvalidPlatformResources)
{
    optiling::IndexCompileInfo zeroCoreCompileInfo{};
    optiling::IndexCompileInfo zeroUbCompileInfo{};

    EXPECT_EQ(RunTilingParse(MakeCompileInfoString(0, DEFAULT_UB_SIZE), &zeroCoreCompileInfo), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTilingParse(MakeCompileInfoString(DEFAULT_CORE_NUM, 0), &zeroUbCompileInfo), ge::GRAPH_FAILED);
}

TEST_F(IndexPutV2Tiling, TilingParseRejectsMissingCompileInfo)
{
    EXPECT_EQ(RunTilingParse(MakeCompileInfoString(DEFAULT_CORE_NUM, DEFAULT_UB_SIZE), nullptr), ge::GRAPH_FAILED);
}

TEST_F(IndexPutV2Tiling, IndexPutV2_AC_tiling_fp16_continue_0)
{
    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::IndexCompileInfo compile_info;

    std::string op_type("IndexPutV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{4096, 6400}, {4096, 6400}};
    gert::StorageShape value = {{4096}, {4096}};
    gert::StorageShape indexedSizes = {{2}, {2}};
    gert::StorageShape indexedStrides = {{1}, {1}};
    gert::StorageShape indices = {{4096}, {4096}};
    gert::StorageShape y = {{4096}, {4096}};
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};

    int64_t mask[2] = {1, 1};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(2, DT_INT64, mask, 2, const_tensors);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&x, &value, &indexedSizes, &indexedStrides, &indices})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"accumulate", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .ConstInput(const_tensors)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 524290);
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::string expect_tiling = "26214400 0 4096 0 4096 0 2 1 2 1 4096 0 6400 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    ASSERT_EQ(expect_tiling, tiling_data_result);
}
