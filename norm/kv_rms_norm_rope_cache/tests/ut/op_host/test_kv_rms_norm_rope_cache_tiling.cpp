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
 * \file repeat_interleave_grad_proto.h
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
#include "../../../op_host/kv_rms_norm_rope_cache_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class KvRmsNormRopeCacheTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvRmsNormRopeCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvRmsNormRopeCacheTiling TearDown" << std::endl;
    }
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
static string to_string(void *buf, size_t size)
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

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5011A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape c_kv_scale_shape = {{512}, {512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BNSD");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 5011);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 65792 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5011B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{32, 1, 15, 192}, {32, 1, 15, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape cos_shape = {{32, 1, 15, 64}, {32, 1, 15, 64}};
    gert::StorageShape sin_shape = {{32, 1, 15, 64}, {32, 1, 15, 64}};
    gert::StorageShape index_shape = {{480}, {480}};
    gert::StorageShape k_cache_shape = {{125, 10, 1, 192}, {125, 10, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{125, 10, 1, 128}, {125, 10, 1, 128}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{32, 1, 15, 128}, {32, 1, 15, 128}};

    gert::StorageShape k_cache_shape_out = {{125, 10, 1, 192}, {125, 10, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{125, 10, 1, 128}, {125, 10, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{32, 1, 15, 192}, {32, 1, 15, 192}};
    gert::StorageShape c_kv_shape_out = {{32, 1, 15, 128}, {32, 1, 15, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BNSD");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 5011);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "48 0 0 0 1 0 32 0 1 0 15 0 10 0 10 0 32 0 925353388 1001040555 65536 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5010A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape c_kv_scale_shape = {{512}, {512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BLK_BNSD");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 5010);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 65792 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5010B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{9, 1, 14, 192}, {9, 1, 14, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape cos_shape = {{9, 1, 14, 64}, {9, 1, 14, 64}};
    gert::StorageShape sin_shape = {{9, 1, 14, 64}, {9, 1, 14, 64}};
    gert::StorageShape index_shape = {{9}, {9}};
    gert::StorageShape k_cache_shape = {{21, 16, 1, 192}, {21, 16, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{21, 16, 1, 128}, {21, 16, 1, 128}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{9, 1, 14, 128}, {9, 1, 14, 128}};

    gert::StorageShape k_cache_shape_out = {{21, 16, 1, 192}, {21, 16, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{21, 16, 1, 128}, {21, 16, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{9, 1, 14, 192}, {9, 1, 14, 192}};
    gert::StorageShape c_kv_shape_out = {{9, 1, 14, 128}, {9, 1, 14, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    printf("ws_size = %ld\n", ws_size);
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BLK_BNSD");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 5010);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 42);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "42 0 0 0 1 0 9 0 1 0 14 0 16 0 3 0 32 0 925353388 1001040555 65536 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5001A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BNSD");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 5001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 1 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5001B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{30, 1, 10, 192}, {30, 1, 10, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape cos_shape = {{30, 1, 10, 64}, {30, 1, 10, 64}};
    gert::StorageShape sin_shape = {{30, 1, 10, 64}, {30, 1, 10, 64}};
    gert::StorageShape index_shape = {{300}, {300}};
    gert::StorageShape k_cache_shape = {{42, 12, 1, 192}, {42, 12, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{42, 12, 1, 128}, {42, 12, 1, 128}};
    gert::StorageShape v_shape = {{30, 1, 10, 128}, {30, 1, 10, 128}};

    gert::StorageShape k_cache_shape_out = {{42, 12, 1, 192}, {42, 12, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{42, 12, 1, 128}, {42, 12, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{30, 1, 10, 192}, {30, 1, 10, 192}};
    gert::StorageShape c_kv_shape_out = {{30, 1, 10, 128}, {30, 1, 10, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BNSD");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 5001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 43);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "43 0 0 0 1 0 30 0 1 0 10 0 12 0 7 0 32 0 925353388 1001040555 1 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5000A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{38, 1, 3809, 576}, {38, 1, 3809, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{228}, {228}};
    gert::StorageShape cos_shape = {{38, 1, 3809, 64}, {38, 1, 3809, 64}};
    gert::StorageShape sin_shape = {{38, 1, 3809, 64}, {38, 1, 3809, 64}};
    gert::StorageShape k_cache_shape = {{230, 745, 1, 64}, {230, 745, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{230, 745, 1, 512}, {230, 745, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{230, 745, 1, 64}, {230, 745, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{230, 745, 1, 512}, {230, 745, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{38, 1, 3809, 64}, {38, 1, 3809, 64}};
    gert::StorageShape c_kv_shape_out = {{38, 1, 3809, 512}, {38, 1, 3809, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BLK_BNSD");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 5000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "48 0 0 0 1 0 38 0 1 0 3809 0 745 0 3016 0 16 0 925353388 989855744 0 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5000B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{9, 1, 14, 192}, {9, 1, 14, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape cos_shape = {{9, 1, 14, 64}, {9, 1, 14, 64}};
    gert::StorageShape sin_shape = {{9, 1, 14, 64}, {9, 1, 14, 64}};
    gert::StorageShape index_shape = {{9}, {9}};
    gert::StorageShape k_cache_shape = {{21, 16, 1, 192}, {21, 16, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{21, 16, 1, 128}, {21, 16, 1, 128}};
    gert::StorageShape v_shape = {{9, 1, 14, 128}, {9, 1, 14, 128}};

    gert::StorageShape k_cache_shape_out = {{21, 16, 1, 192}, {21, 16, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{21, 16, 1, 128}, {21, 16, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{9, 1, 14, 192}, {9, 1, 14, 192}};
    gert::StorageShape c_kv_shape_out = {{9, 1, 14, 128}, {9, 1, 14, 128}};


    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BLK_BNSD");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 5000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 42);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "42 0 0 0 1 0 9 0 1 0 14 0 16 0 3 0 32 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4011A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape c_kv_scale_shape = {{512}, {512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 4011);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 65792 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4011B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_scale_shape = {{192}, {192}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 4011);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 32 0 925353388 1001040555 65792 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4010A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape c_kv_scale_shape = {{512}, {512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BLK_NZ");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 4010);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 65792 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4010B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_scale_shape = {{192}, {192}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BLK_NZ");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 4010);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 32 0 925353388 1001040555 65792 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4001A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 4001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 0 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4001B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64}, {64}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_NZ");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 4001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 32);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 32 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4000A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{38, 1, 3809, 576}, {38, 1, 3809, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{228}, {228}};
    gert::StorageShape cos_shape = {{38, 1, 3809, 64}, {38, 1, 3809, 64}};
    gert::StorageShape sin_shape = {{38, 1, 3809, 64}, {38, 1, 3809, 64}};
    gert::StorageShape k_cache_shape = {{230, 745, 1, 64}, {230, 745, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{230, 745, 1, 512}, {230, 745, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{230, 745, 1, 64}, {230, 745, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{230, 745, 1, 512}, {230, 745, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{38, 1, 3809, 64}, {38, 1, 3809, 64}};
    gert::StorageShape c_kv_shape_out = {{38, 1, 3809, 512}, {38, 1, 3809, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BLK_NZ");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 4000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "48 0 0 0 1 0 38 0 1 0 3809 0 745 0 3016 0 16 0 925353388 989855744 0 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4000B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{38, 1, 3809, 192}, {38, 1, 3809, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{228}, {228}};
    gert::StorageShape cos_shape = {{38, 1, 3809, 64}, {38, 1, 3809, 64}};
    gert::StorageShape sin_shape = {{38, 1, 3809, 64}, {38, 1, 3809, 64}};
    gert::StorageShape k_cache_shape = {{230, 745, 1, 192}, {230, 745, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{230, 745, 1, 128}, {230, 745, 1, 128}};
    gert::StorageShape v_shape = {{38, 1, 3809, 128}, {38, 1, 3809, 128}};

    gert::StorageShape k_cache_shape_out = {{230, 745, 1, 192}, {230, 745, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{230, 745, 1, 128}, {230, 745, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{38, 1, 3809, 192}, {38, 1, 3809, 192}};
    gert::StorageShape c_kv_shape_out = {{38, 1, 3809, 128}, {38, 1, 3809, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA_BLK_NZ");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 4000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "48 0 0 0 1 0 38 0 1 0 3809 0 745 0 3016 0 32 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3010_SYMMETRIC_QUANT)   //对称量化
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape sin_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 3010);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 45);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "45 0 0 0 1 0 64 0 1 0 7 0 128 0 10 0 32 0 925353388 1001040555 65537 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3010_ASYMMETRIC_QUANT)   //非对称量化
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape sin_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape c_kv_offset_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, &c_kv_scale_shape, nullptr, &c_kv_offset_shape, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 3010);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 45);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "45 0 0 0 1 0 64 0 1 0 7 0 128 0 10 0 32 0 925353388 1001040555 65537 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3001A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 576}, {64, 1, 7, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape sin_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 512}, {64, 1, 7, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("PA")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 3001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 45);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "45 0 0 0 1 0 64 0 1 0 7 0 128 0 10 0 16 0 925353388 989855744 0 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3001B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape sin_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape v_shape = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("PA")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 3001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 45);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "45 0 0 0 1 0 64 0 1 0 7 0 128 0 10 0 32 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3000A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 576}, {64, 1, 7, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape sin_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 512}, {64, 1, 7, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 3000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 45);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "45 0 0 0 1 0 64 0 1 0 7 0 128 0 10 0 16 0 925353388 989855744 0 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3000B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape sin_shape = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape v_shape = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 3000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 45);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "45 0 0 0 1 0 64 0 1 0 7 0 128 0 10 0 32 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2001A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 576}, {64, 1, 7, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 7, 64}, {64, 1, 7, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 512}, {64, 1, 7, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 2001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 7 0 128 0 0 0 0 0 925353388 989855744 0 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2001B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape v_shape = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 2001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 7 0 128 0 0 0 0 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2000A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 64}, {192, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 512}, {192, 128, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 64}, {192, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 512}, {192, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 2000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 1 0 128 0 0 0 0 0 925353388 989855744 0 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2000B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 2000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 1 0 128 0 0 0 0 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1011B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 1011);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 7 0 128 0 0 0 0 0 925353388 1001040555 65537 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1010B)   
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 1010);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 1 0 128 0 0 0 0 0 925353388 1001040555 65537 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1000A)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 1000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 1 0 128 0 0 0 0 0 925353388 989855744 1 0 0 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1000B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 1000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 1 0 128 0 0 0 0 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1001B)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 7, 192}, {64, 1, 7, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 7}, {64, 7}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 192}, {192, 128, 1, 192}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape v_shape = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 7, 192}, {192, 128, 7, 192}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 7, 128}, {64, 1, 7, 128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;

    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 1001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 16);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "16 0 4 0 1 0 64 0 1 0 7 0 128 0 0 0 0 0 925353388 1001040555 0 0 1 0 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}


TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_tilingA)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{64, 1, 1, 512}, {64, 1, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape c_kv_scale_shape = {{512}, {512}};

    gert::StorageShape k_cache_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;
    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("Norm");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 10000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "64 0 1 0 1 0 1 0 1 0 0 0 2 0 0 0 2 0 0 0 0 0 0 0 64 0 64 0 64 0 32 0 32 0 512 0 512 0 512 0 1 0 1 0 1024 0 1536 0 2048 0 925353388 989855744 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_tilingB)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{192}, {192}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_scale_shape = {{192}, {192}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape ckv_cache_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;
    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("Norm");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    ASSERT_EQ(tiling_key, 10000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "64 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 64 0 64 0 64 0 32 0 32 0 192 0 192 0 192 0 1 0 1 0 384 0 576 0 768 0 925353388 1001040555 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_tilingA)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{64, 1, 1, 512}, {64, 1, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape c_kv_scale_shape = {{512}, {512}};

    gert::StorageShape k_cache_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;
    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("Norm");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 10000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "64 0 1 0 1 0 1 0 1 0 0 0 2 0 0 0 2 0 0 0 1 0 0 0 64 0 64 0 64 0 32 0 32 0 512 0 512 0 512 0 1 0 1 0 1024 0 1536 0 2048 0 925353388 989855744 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_tilingB)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;
    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("Norm");

    auto holder = gert::TilingContextFaker()
                      .SetOpType("KvRmsNormRopeCache")
                      .NodeIoNum(12, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, &c_kv_scale_shape, nullptr, nullptr, &v_shape})
                      .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 10000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "64 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 2 0 0 0 1 0 0 0 64 0 64 0 64 0 32 0 32 0 128 0 128 0 128 0 1 0 1 0 256 0 384 0 512 0 925353388 1006632960 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_with_PA_tilingA)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 576}, {64, 1, 1, 576}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_scale_shape = {{64}, {64}};
    gert::StorageShape c_kv_scale_shape = {{512}, {512}};
    gert::StorageShape k_rope_offset_shape = {{64}, {64}};
    gert::StorageShape c_kv_offset_shape = {{512}, {512}};

    gert::StorageShape k_cache_shape_out = {{576, 128, 1, 64}, {576, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{576, 128, 1, 512}, {576, 128, 1, 512}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 512}, {64, 1, 1, 512}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;
    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA");

    auto holder =
        gert::TilingContextFaker()
            .SetOpType("KvRmsNormRopeCache")
            .NodeIoNum(12, 4)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape, &ckv_cache_shape,
                 &k_rope_scale_shape, &c_kv_scale_shape, &k_rope_offset_shape, &c_kv_offset_shape, nullptr})
            .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

            .NodeAttrs(
                {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                 {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                 {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 10000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "64 0 1 0 1 0 1 0 128 0 0 0 2 0 2 0 2 0 2 0 1 0 1 0 64 0 64 0 64 0 32 0 32 0 512 0 512 0 512 0 1 0 1 0 1024 0 1536 0 2048 0 925353388 989855744 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_with_PA_tilingB)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape kv_shape = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape index_shape = {{64, 1}, {64, 1}};
    gert::StorageShape cos_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape sin_shape = {{64, 1, 1, 64}, {64, 1, 1, 64}};
    gert::StorageShape k_cache_shape = {{192, 128, 1, 64}, {192, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape c_kv_scale_shape = {{128}, {128}};
    gert::StorageShape c_kv_offset_shape = {{128}, {128}};
    gert::StorageShape v_shape = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    gert::StorageShape k_cache_shape_out = {{192, 128, 1, 64}, {192, 128, 1, 64}};
    gert::StorageShape ckv_cache_shape_out = {{192, 128, 1, 128}, {192, 128, 1, 128}};
    gert::StorageShape k_rope_shape_out = {{64, 1, 1, 192}, {64, 1, 1, 192}};
    gert::StorageShape c_kv_shape_out = {{64, 1, 1, 128}, {64, 1, 1, 128}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::KvRmsNormRopeCacheCompileInfo compile_info;
    std::string op_type("KvRmsNormRopeCache");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string cache_mode("PA");

    auto holder =
        gert::TilingContextFaker()
            .SetOpType("KvRmsNormRopeCache")
            .NodeIoNum(12, 4)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape, &ckv_cache_shape,
                 nullptr, &c_kv_scale_shape, nullptr, &c_kv_offset_shape, &v_shape})
            .OutputShapes({&k_cache_shape_out, &ckv_cache_shape_out, &k_rope_shape_out, &c_kv_shape_out})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

            .NodeAttrs(
                {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)},
                 {"cache_mode", Ops::NN::AnyValue::CreateFrom<std::string>(cache_mode)},
                 {"is_output_kv", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 10000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "64 0 1 0 1 0 1 0 128 0 0 0 0 0 0 0 2 0 2 0 1 0 1 0 64 0 64 0 64 0 32 0 32 0 128 0 128 0 128 0 1 0 1 0 256 0 384 0 512 0 925353388 1006632960 ");
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}