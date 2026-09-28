/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scatter_elements_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "log/log.h"
#include "graph/graph.h"
#include "kernel_run_context_facker.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch22/scatter_elements_v2_tiling.h"

using namespace std;
using namespace ge;

class ScatterElementsV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScatterElementsV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScatterElementsV2Tiling TearDown" << std::endl; }
};

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

template <typename T>
static string to_string(void* buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

struct Arch22TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    uint64_t inputCount = 0;
    uint64_t inputOneTime = 0;
    uint64_t indicesLoop = 0;
    uint64_t modeFlag = 0;
    uint64_t stableBucket = 0;
    // 分桶散射分支的 tiling 结果；bktMode == 0 表示未走该分支（走既有主路径）
    uint64_t bktMode = 0;
    uint64_t bktRows = 0;
    uint64_t bktVarN = 0;
    uint64_t bktIndicesN = 0;
    uint64_t bktTileLen = 0;
    uint64_t bktNumTiles = 0;
    uint64_t bktShift = 0;
    uint64_t bktFifoDepth = 0;
    uint64_t bktStride = 0;
    uint64_t bktRowsPerCore = 0;
    uint64_t bktFrontCore = 0;
};

static Arch22TilingResult ExecuteArch22DeterministicCase(ge::DataType inputDtype, ge::DataType indicesDtype,
                                                         ge::DataType updatesDtype, gert::StorageShape& inputShape,
                                                         gert::StorageShape& indicesShape,
                                                         gert::StorageShape& updatesShape, int64_t axis,
                                                         const std::string& reduction, int32_t deterministic)
{
    std::string op_type("ScatterElementsV2");
    EXPECT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "220"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::ScatterElementsV2CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    EXPECT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    EXPECT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    EXPECT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    auto outputShape = inputShape;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &indicesShape, &updatesShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compile_info)
                      .DeterministicInfo(reinterpret_cast<int32_t*>(deterministic))
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, updatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    Arch22TilingResult result;
    result.status = tiling_func(tiling_context);
    result.tilingKey = tiling_context->GetTilingKey();
    result.blockDim = tiling_context->GetBlockDim();
    if (result.status == ge::GRAPH_SUCCESS) {
        // The first 29 serialized fields in ScatterElementsV2TilingData are
        // uint64_t. Inspect the actual device payload, not the host defaults.
        const auto* fields = static_cast<const uint64_t*>(tiling_context->GetRawTilingData()->GetData());
        result.inputCount = fields[5];
        result.inputOneTime = fields[8];
        result.indicesLoop = fields[16];
        result.modeFlag = fields[25];
        result.stableBucket = fields[28];
        // 分桶字段追加在 realDim 之后，按同一 uint64 下标口径读取：
        // 前 29 个 uint64（usedCoreNum..M）占 0..231 字节；int32 coreNums 落在 232，
        // 其后补 4 字节自然对齐；xDim0 起 8 个 uint64（240..303）止于 realDim(296)。
        // 因此 bktMode 起始字节 304，即 uint64 下标 38。
        constexpr size_t BKT_FIELD_BASE = 38U;
        EXPECT_GE(tiling_context->GetRawTilingData()->GetDataSize(), (BKT_FIELD_BASE + 11U) * sizeof(uint64_t));
        result.bktMode = fields[BKT_FIELD_BASE + 0U];
        result.bktRows = fields[BKT_FIELD_BASE + 1U];
        result.bktVarN = fields[BKT_FIELD_BASE + 2U];
        result.bktIndicesN = fields[BKT_FIELD_BASE + 3U];
        result.bktTileLen = fields[BKT_FIELD_BASE + 4U];
        result.bktNumTiles = fields[BKT_FIELD_BASE + 5U];
        result.bktShift = fields[BKT_FIELD_BASE + 6U];
        result.bktFifoDepth = fields[BKT_FIELD_BASE + 7U];
        result.bktStride = fields[BKT_FIELD_BASE + 8U];
        result.bktRowsPerCore = fields[BKT_FIELD_BASE + 9U];
        result.bktFrontCore = fields[BKT_FIELD_BASE + 10U];
    }
    return result;
}

TEST_F(ScatterElementsV2Tiling, experimental_none_uses_implemented_row_kernel)
{
    const std::vector<ge::DataType> valueTypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                  ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT16,
                                                  ge::DT_INT32, ge::DT_INT64,   ge::DT_DOUBLE};
    for (auto valueType : valueTypes) {
        for (auto indexType : {ge::DT_INT32, ge::DT_INT64}) {
            gert::StorageShape inputShape = {{23, 1691}, {23, 1691}};
            gert::StorageShape indicesShape = {{23, 845}, {23, 845}};
            gert::StorageShape updatesShape = indicesShape;
            auto result = ExecuteArch22DeterministicCase(valueType, indexType, valueType, inputShape, indicesShape,
                                                         updatesShape, -1, "none", 0);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            EXPECT_GT(result.blockDim, 0U);
            EXPECT_EQ(result.inputCount, 23U * 1691U);
            EXPECT_EQ(result.inputOneTime, 1691U);
            EXPECT_GT(result.indicesLoop, 0U);
            EXPECT_EQ(result.modeFlag, 1U);
            EXPECT_EQ(result.stableBucket, 0U);
        }
    }
}

TEST_F(ScatterElementsV2Tiling, experimental_large_row_has_multiple_stable_source_chunks)
{
    gert::StorageShape inputShape = {{1, 16438026}, {1, 16438026}};
    gert::StorageShape indicesShape = {{1, 4210646}, {1, 4210646}};
    gert::StorageShape updatesShape = indicesShape;
    auto result = ExecuteArch22DeterministicCase(ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, inputShape, indicesShape,
                                                 updatesShape, -1, "none", 0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.inputCount, 16438026U);
    EXPECT_EQ(result.modeFlag, 0U);
    EXPECT_EQ(result.stableBucket, 1U);
    EXPECT_GT(result.indicesLoop, 1U);
}

TEST_F(ScatterElementsV2Tiling, experimental_small_row_budget_uses_update_value_width)
{
    // 8000*8 + 14000*(8+4) bytes does not fit in UB. Counting the int64
    // updates as int32 indices would incorrectly select the small-row path.
    gert::StorageShape inputShape = {{1, 8000}, {1, 8000}};
    gert::StorageShape indicesShape = {{1, 14000}, {1, 14000}};
    gert::StorageShape updatesShape = indicesShape;
    auto result = ExecuteArch22DeterministicCase(ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, inputShape, indicesShape,
                                                 updatesShape, -1, "none", 0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.inputCount, 8000U);
    EXPECT_EQ(result.modeFlag, 0U);
    EXPECT_GT(result.indicesLoop, 0U);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_float32)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ScatterElementsV2CompileInfo compile_info;
    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{4096, 5933}, {4096, 5933}};
    gert::StorageShape indic_shape = {{4096, 5933}, {4096, 5933}};
    gert::StorageShape src_shape = {{4096, 5933}, {4096, 5933}};
    gert::StorageShape output_shape = {{4096, 5933}, {4096, 5933}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("add")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 120);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_float32_few)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ScatterElementsV2CompileInfo compile_info;
    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{10, 5933}, {10, 5933}};
    gert::StorageShape indic_shape = {{10, 5933}, {10, 5933}};
    gert::StorageShape src_shape = {{10, 5933}, {10, 5933}};
    gert::StorageShape output_shape = {{10, 5933}, {10, 5933}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("add")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 120);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_float16)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ScatterElementsV2CompileInfo compile_info;
    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{4096, 20000}, {4096, 20000}};
    gert::StorageShape indic_shape = {{4096, 2000}, {4096, 2000}};
    gert::StorageShape src_shape = {{4096, 2000}, {4096, 2000}};
    gert::StorageShape output_shape = {{4096, 20000}, {4096, 20000}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("none")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 210);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_bfloat16)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ScatterElementsV2CompileInfo compile_info;
    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("add")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 620);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_mul_error)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ScatterElementsV2CompileInfo compile_info;
    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mul")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 120);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_min)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::ScatterElementsV2CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("min")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 120);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_max)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::ScatterElementsV2CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("max")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 210);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_mean)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::ScatterElementsV2CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 620);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_none_deterministic_multi_core)
{
    gert::StorageShape input_shape = {{4096, 20000}, {4096, 20000}};
    gert::StorageShape indic_shape = {{4096, 2000}, {4096, 2000}};
    gert::StorageShape src_shape = {{4096, 2000}, {4096, 2000}};

    auto normalResult = ExecuteArch22DeterministicCase(ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, input_shape,
                                                       indic_shape, src_shape, -1, "none", 0);
    EXPECT_EQ(normalResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(normalResult.tilingKey, 210);
    EXPECT_GT(normalResult.blockDim, 1U);

    auto deterministicResult = ExecuteArch22DeterministicCase(ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, input_shape,
                                                              indic_shape, src_shape, -1, "none", 1);
    EXPECT_EQ(deterministicResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(deterministicResult.tilingKey, 210);
    EXPECT_EQ(deterministicResult.blockDim, normalResult.blockDim);
    EXPECT_GT(deterministicResult.blockDim, 1U);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_add_deterministic_multi_core)
{
    gert::StorageShape input_shape = {{4096, 5933}, {4096, 5933}};
    gert::StorageShape indic_shape = {{4096, 5933}, {4096, 5933}};
    gert::StorageShape src_shape = {{4096, 5933}, {4096, 5933}};

    auto normalResult = ExecuteArch22DeterministicCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, input_shape,
                                                       indic_shape, src_shape, -1, "add", 0);
    EXPECT_EQ(normalResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(normalResult.tilingKey, 120);
    EXPECT_GT(normalResult.blockDim, 1U);

    auto deterministicResult = ExecuteArch22DeterministicCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, input_shape,
                                                              indic_shape, src_shape, -1, "add", 1);
    EXPECT_EQ(deterministicResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(deterministicResult.tilingKey, 120);
    EXPECT_EQ(deterministicResult.blockDim, normalResult.blockDim);
    EXPECT_GT(deterministicResult.blockDim, 1U);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_mul_deterministic_single_core)
{
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};

    auto normalResult = ExecuteArch22DeterministicCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, input_shape,
                                                       indic_shape, src_shape, -1, "mul", 0);
    EXPECT_EQ(normalResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(normalResult.tilingKey, 120);
    EXPECT_GT(normalResult.blockDim, 1U);

    auto deterministicResult = ExecuteArch22DeterministicCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, input_shape,
                                                              indic_shape, src_shape, -1, "mul", 1);
    EXPECT_EQ(deterministicResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(deterministicResult.tilingKey, 120);
    EXPECT_EQ(deterministicResult.blockDim, 1U);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_mean_deterministic_single_core)
{
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};

    auto normalResult = ExecuteArch22DeterministicCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, input_shape,
                                                       indic_shape, src_shape, -1, "mean", 0);
    EXPECT_EQ(normalResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(normalResult.tilingKey, 120);
    EXPECT_GT(normalResult.blockDim, 1U);

    auto deterministicResult = ExecuteArch22DeterministicCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, input_shape,
                                                              indic_shape, src_shape, -1, "mean", 1);
    EXPECT_EQ(deterministicResult.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(deterministicResult.tilingKey, 120);
    EXPECT_EQ(deterministicResult.blockDim, 1U);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_other_error)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ScatterElementsV2CompileInfo compile_info;
    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 3192}, {4096, 3192}};
    gert::StorageShape src_shape = {{4096, 3192}, {4096, 3192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("other")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_NE(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

// ==================== 分桶散射分支（BFLOAT16 稀疏大 var）====================
// BucketScatterSupport() 有四道收窄闸：dtype 必须 BFLOAT16、reduction 必须 none、
// var 末轴 >= BUCKET_MIN_VAR_N(65536)、indicesN * BUCKET_SPARSE_RATIO(8) <= varN。
// 下面 2 个正向用例断言分桶分支被选中且各 bkt* 字段自洽；4 个反向用例逐条验证四道闸，
// 用来证明既有 dtype / reduction 的选路未被本次改动影响。

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_bucket_bf16_int32_2d)
{
    gert::StorageShape inputShape = {{4, 131072}, {4, 131072}};
    gert::StorageShape indicesShape = {{4, 1024}, {4, 1024}};
    gert::StorageShape updatesShape = {{4, 1024}, {4, 1024}};
    auto result = ExecuteArch22DeterministicCase(ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, inputShape, indicesShape,
                                                 updatesShape, -1, "none", 0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    // 分桶分支必须被选中：修复前 xDim1 恒为成员默认值 1，xDim1 < 65536 恒成立，该分支永不生效
    EXPECT_EQ(result.bktMode, 1UL);
    EXPECT_EQ(result.bktRows, 4UL);
    EXPECT_EQ(result.bktVarN, 131072UL);
    EXPECT_EQ(result.bktIndicesN, 1024UL);
    // 结构性关系：tileLen 为 2 的幂、numTiles = ceil(varN/tileLen)、fifoDepth 按 16 对齐
    EXPECT_EQ(result.bktTileLen, 1UL << result.bktShift);
    EXPECT_EQ(result.bktNumTiles, (result.bktVarN + result.bktTileLen - 1) / result.bktTileLen);
    EXPECT_EQ(result.bktFifoDepth % 16UL, 0UL);
    // ★ 桶区容量下界：kernel 侧每桶预留 ceil(cnt,16)*16 + 16 <= cnt + 31，
    //   故 Σ padded <= bktIndicesN + 31 * bktNumTiles；原实现只给 +16*numTiles，会越界写邻核
    EXPECT_GE(result.bktStride, result.bktIndicesN + 31UL * result.bktNumTiles);
    // 行分核：usedCore = min(rows, coreNum)，rows=4 远小于核数，故每核 1 行、无前置核多分
    EXPECT_EQ(result.blockDim, 4U);
    EXPECT_EQ(result.bktRowsPerCore, 1UL);
    EXPECT_EQ(result.bktFrontCore, 0UL);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_bucket_bf16_int64_3d)
{
    gert::StorageShape inputShape = {{2, 3, 98304}, {2, 3, 98304}};
    gert::StorageShape indicesShape = {{2, 3, 512}, {2, 3, 512}};
    gert::StorageShape updatesShape = {{2, 3, 512}, {2, 3, 512}};
    auto result = ExecuteArch22DeterministicCase(ge::DT_BF16, ge::DT_INT64, ge::DT_BF16, inputShape, indicesShape,
                                                 updatesShape, 2, "none", 0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.bktMode, 1UL);
    // 末轴之前的维度折叠为行数：2 * 3 = 6
    EXPECT_EQ(result.bktRows, 6UL);
    EXPECT_EQ(result.bktVarN, 98304UL);
    EXPECT_EQ(result.bktIndicesN, 512UL);
    EXPECT_EQ(result.bktTileLen, 1UL << result.bktShift);
    EXPECT_EQ(result.bktNumTiles, (result.bktVarN + result.bktTileLen - 1) / result.bktTileLen);
    EXPECT_GE(result.bktStride, result.bktIndicesN + 31UL * result.bktNumTiles);
    EXPECT_EQ(result.blockDim, 6U);
}

// 反向闸 1：dtype 非 BFLOAT16（FLOAT16 同形状）
TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_bucket_reject_dtype)
{
    gert::StorageShape inputShape = {{4, 131072}, {4, 131072}};
    gert::StorageShape indicesShape = {{4, 1024}, {4, 1024}};
    gert::StorageShape updatesShape = {{4, 1024}, {4, 1024}};
    auto result = ExecuteArch22DeterministicCase(ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, inputShape, indicesShape,
                                                 updatesShape, -1, "none", 0);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.bktMode, 0UL);
}

// 反向闸 2：reduction 非 none（累积语义不适用"末次写赢"）
TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_bucket_reject_reduction)
{
    gert::StorageShape inputShape = {{4, 131072}, {4, 131072}};
    gert::StorageShape indicesShape = {{4, 1024}, {4, 1024}};
    gert::StorageShape updatesShape = {{4, 1024}, {4, 1024}};
    auto result = ExecuteArch22DeterministicCase(ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, inputShape, indicesShape,
                                                 updatesShape, -1, "add", 0);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.bktMode, 0UL);
}

// 反向闸 3：var 末轴小于 BUCKET_MIN_VAR_N(65536)
TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_bucket_reject_small_var)
{
    gert::StorageShape inputShape = {{4, 32768}, {4, 32768}};
    gert::StorageShape indicesShape = {{4, 256}, {4, 256}};
    gert::StorageShape updatesShape = {{4, 256}, {4, 256}};
    auto result = ExecuteArch22DeterministicCase(ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, inputShape, indicesShape,
                                                 updatesShape, -1, "none", 0);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.bktMode, 0UL);
}

// 反向闸 4：不够稀疏（indicesN * 8 > varN）
TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_bucket_reject_dense)
{
    gert::StorageShape inputShape = {{4, 65536}, {4, 65536}};
    gert::StorageShape indicesShape = {{4, 16384}, {4, 16384}};
    gert::StorageShape updatesShape = {{4, 16384}, {4, 16384}};
    auto result = ExecuteArch22DeterministicCase(ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, inputShape, indicesShape,
                                                 updatesShape, -1, "none", 0);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.bktMode, 0UL);
}
