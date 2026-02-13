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
 * \file test_embedding_dense_grad_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/embedding_dense_grad_tiling.h"

using namespace std;
using namespace ge;

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

class EmbeddingDenseGradRegBaseTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EmbeddingDenseGradRegBaseTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EmbeddingDenseGradRegBaseTiling TearDown" << std::endl;
    }
};

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_full_load_bf16_scale)
{
    std::string op_type("EmbeddingDenseGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64, "socVersion": "Ascend950"}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct EmbeddingDenseGradCompileInfo {
        int32_t core_num{0};
        int32_t scale_grad_by_freq{0};
        bool isAscendc{false};
        uint32_t maxCoreNum{0};
        uint32_t ubSizePlatform{0};
        uint32_t maxThreadNum{0};
    } compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = {{1024, 6000}, {1024, 4096}};
    gert::StorageShape input_1 = {{512, 1, 2, 1, 1, 1}, {512, 1, 2, 1, 1, 1}};
    gert::StorageShape output_shape = {{1667, 6000}, {1667, 6000}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, &input_1})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(1667)},
                           {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                           {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
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
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    uint64_t tilingKeyValue = 801;
    ASSERT_EQ(tiling_key, tilingKeyValue);
    std::string expectTilingData =
        "1667 0 4 0 1 0 4096 0 579 37056 2 0 1 0 4 0 1 0 64 0 64 0 64 64 445 472 251 64 64 0 0 0 0 0 0 0 0 0 64 64 ";
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_full_load_fp16_no_scale)
{
    std::string op_type("EmbeddingDenseGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct EmbeddingDenseGradCompileInfo {
        int32_t core_num{0};
        int32_t scale_grad_by_freq{0};
        bool isAscendc{false};
        uint32_t maxCoreNum{0};
        uint32_t ubSizePlatform{0};
        uint32_t maxThreadNum{0};
    } compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = {{83, 7078}, {83, 7078}};
    gert::StorageShape input_1 = {{83, 1, 1, 1, 1, 1}, {83, 1, 1, 1, 1, 1}};
    gert::StorageShape output_shape = {{10145, 7078}, {10145, 7078}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, &input_1})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(10145)},
                           {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(-9)},
                           {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
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
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    uint64_t tilingKeyValue = 102;
    ASSERT_EQ(tiling_key, tilingKeyValue);
    std::string expectTilingData =
        "10145 0 -9 -1 0 0 7078 0 83 59760 1 0 1 0 0 0 0 0 111 0 85 0 720 111 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 64 ";
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_no_full_load_fp16_int32_no_scale)
{
    std::string op_type("EmbeddingDenseGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct EmbeddingDenseGradCompileInfo {
        int32_t core_num{0};
        int32_t scale_grad_by_freq{0};
        bool isAscendc{false};
        uint32_t maxCoreNum{0};
        uint32_t ubSizePlatform{0};
        uint32_t maxThreadNum{0};
    } compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = {{2011, 5123}, {2011, 5123}};
    gert::StorageShape input_1 = {{2011, 1, 1, 1, 1, 1}, {2011, 1, 1, 1, 1, 1}};
    gert::StorageShape output_shape = {{3000, 5123}, {3000, 5123}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, &input_1})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(3000)},
                           {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(10)},
                           {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
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
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    uint64_t tilingKeyValue = 402;
    ASSERT_EQ(tiling_key, tilingKeyValue);
    std::string expectTilingData =
        "3000 0 10 0 0 0 5123 0 435 41760 5 0 1 0 0 0 0 0 81 0 20 0 81 81 271 0 0 0 0 0 81 426 1 0 8 0 81 18 64 64 ";
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_no_full_load_fp16_int32_scale)
{
    std::string op_type("EmbeddingDenseGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct EmbeddingDenseGradCompileInfo {
        int32_t core_num{0};
        int32_t scale_grad_by_freq{0};
        bool isAscendc{false};
        uint32_t maxCoreNum{0};
        uint32_t ubSizePlatform{0};
        uint32_t maxThreadNum{0};
    } compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape input_1 = {{4096, 1, 1, 1, 1, 1}, {4096, 1, 1, 1, 1, 1}};
    gert::StorageShape output_shape = {{3000, 4096}, {3000, 4096}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, &input_1})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(3000)},
                           {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(10)},
                           {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
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
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    uint64_t tilingKeyValue = 402;
    ASSERT_EQ(tiling_key, tilingKeyValue);
    std::string expectTilingData =
        "3000 0 10 0 1 0 4096 0 608 38912 7 0 1 0 7 0 1 0 64 0 64 0 64 64 448 476 144 64 64 0 0 0 0 0 0 0 0 0 64 64 ";
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_no_full_load_fp32_int64_scale)
{
    std::string op_type("EmbeddingDenseGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct EmbeddingDenseGradCompileInfo {
        int32_t core_num{0};
        int32_t scale_grad_by_freq{0};
        bool isAscendc{false};
        uint32_t maxCoreNum{0};
        uint32_t ubSizePlatform{0};
        uint32_t maxThreadNum{0};
    } compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = {{2110, 4096}, {2110, 4096}};
    gert::StorageShape input_1 = {{2110, 1, 1, 1, 1, 1}, {2110, 1, 1, 1, 1, 1}};
    gert::StorageShape output_shape = {{3000, 4096}, {3000, 4096}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, &input_1})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(3000)},
                           {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(10)},
                           {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
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
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    uint64_t tilingKeyValue = 804;
    ASSERT_EQ(tiling_key, tilingKeyValue);
    std::string expectTilingData =
        "3000 0 10 0 1 0 4096 0 444 28416 5 0 1 0 7 0 1 0 64 0 64 0 64 64 334 472 168 64 64 0 0 0 0 0 0 0 0 0 64 64 ";
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

static void runCase(
    int32_t freq, gert::StorageShape input0, gert::StorageShape input1, gert::StorageShape output, int64_t numWeights,
    int64_t paddingIdx, const ge::graphStatus& expectStatus)
{
    std::string op_type("EmbeddingDenseGrad");
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
    struct EmbeddingDenseGradCompileInfo {
        int32_t core_num = 32;
        int32_t scale_grad_by_freq = 0;
    } compile_info;
    compile_info.scale_grad_by_freq = freq;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_0 = input0;
    gert::StorageShape input_1 = input1;
    gert::StorageShape output_shape = output;

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, &input_1})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"num_weights", Ops::NN::AnyValue::CreateFrom<int64_t>(numWeights)},
                           {"padding_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(paddingIdx)},
                           {"scale_grad_by_freq", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_EQ(tiling_func(tiling_context), expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_0)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{20000, 512}, {20000, 512}};
    gert::StorageShape input1 = {{20000}, {20000}};
    gert::StorageShape output = {{20000, 512}, {20000, 512}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 20;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_1)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{30000, 1024}, {30000, 1024}};
    gert::StorageShape input1 = {{30000}, {30000}};
    gert::StorageShape output = {{20000, 1024}, {20000, 1024}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 20;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_2)
{
    int32_t freq = 0;
    gert::StorageShape input0 = {{10000, 768}, {10000, 768}};
    gert::StorageShape input1 = {{10000}, {10000}};
    gert::StorageShape output = {{20000, 768}, {20000, 768}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 10;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_3)
{
    int32_t freq = 0;
    gert::StorageShape input0 = {{1000, 6}, {1000, 6}};
    gert::StorageShape input1 = {{1000}, {1000}};
    gert::StorageShape output = {{100, 6}, {100, 6}};
    int64_t numWeights = 100;
    int64_t paddingIdx = 10;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_4)
{
    int32_t freq = 0;
    gert::StorageShape input0 = {{100000, 100}, {100000, 100}};
    gert::StorageShape input1 = {{100000}, {100000}};
    gert::StorageShape output = {{20000, 100}, {20000, 100}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 540;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_5)
{
    int32_t freq = 0;
    gert::StorageShape input0 = {{100000, 100}, {100000, 100}};
    gert::StorageShape input1 = {{100000}, {100000}};
    gert::StorageShape output = {{20000, 100}, {20000, 100}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 1000;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_6)
{
    int32_t freq = 0;
    gert::StorageShape input0 = {{7, 10}, {7, 10}};
    gert::StorageShape input1 = {{7}, {7}};
    gert::StorageShape output = {{20000, 100}, {20000, 100}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = -1;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_7)
{
    int32_t freq = 0;
    gert::StorageShape input0 = {{1024, 2048}, {1024, 2048}};
    gert::StorageShape input1 = {{1024}, {1024}};
    gert::StorageShape output = {{20000, 2048}, {20000, 2048}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = -1;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_8)
{
    int32_t freq = 0;
    gert::StorageShape input0 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape input1 = {{1024}, {1024}};
    gert::StorageShape output = {{20000, 4096}, {20000, 4096}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 1000;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_9)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{1000, 6}, {1000, 6}};
    gert::StorageShape input1 = {{1000}, {1000}};
    gert::StorageShape output = {{100, 6}, {100, 6}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 10;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_10)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{100000, 100}, {1000, 100}};
    gert::StorageShape input1 = {{1000}, {1000}};
    gert::StorageShape output = {{20000, 100}, {20000, 100}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 540;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_11)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape input1 = {{1024}, {1024}};
    gert::StorageShape output = {{20000, 4096}, {20000, 4096}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 1000;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_12)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape input1 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape output = {{20000, 4096}, {20000, 4096}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 1000;
    const ge::graphStatus expectStatus = ge::GRAPH_FAILED;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_13)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape input1 = {{2000}, {2000}};
    gert::StorageShape output = {{20000, 4096}, {20000, 4096}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 1000;
    const ge::graphStatus expectStatus = ge::GRAPH_FAILED;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_14)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape input1 = {{2000}, {2000}};
    gert::StorageShape output = {{20000, 4096}, {20000, 4096}};
    int64_t numWeights = 20000;
    int64_t paddingIdx = 1000;
    const ge::graphStatus expectStatus = ge::GRAPH_FAILED;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_scale_huge_int64)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{1, 100000}, {1, 100000}};
    gert::StorageShape input1 = {{100000}, {100000}};
    gert::StorageShape output = {{1, 100000}, {1, 100000}};
    int64_t numWeights = 1;
    int64_t paddingIdx = 540;
    const ge::graphStatus expectStatus = ge::GRAPH_FAILED;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}

TEST_F(EmbeddingDenseGradRegBaseTiling, embedding_dense_grad_tiling_scale_huge_int32)
{
    int32_t freq = 1;
    gert::StorageShape input0 = {{1, 100000}, {1, 100000}};
    gert::StorageShape input1 = {{100000}, {100000}};
    gert::StorageShape output = {{1, 100000}, {1, 100000}};
    int64_t numWeights = 1;
    int64_t paddingIdx = 540;
    const ge::graphStatus expectStatus = ge::GRAPH_FAILED;
    runCase(freq, input0, input1, output, numWeights, paddingIdx, expectStatus);
}