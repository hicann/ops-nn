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
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "op_log.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "array_ops.h"
#include "experiment_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "common_unittest.h"
#include "../../../op_host/kv_rms_norm_rope_cache_tiling.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "tiling//tiling_base.h"
#include "tiling/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "test_cube_util.h"

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

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5011)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(false)}})
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
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 65792 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5010)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(false)}})
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
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 65792 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4011)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(false)}})
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
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 65792 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4010)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(false)}})
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
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 65792 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5001)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
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
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 1 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5000)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(false)}})
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
        "48 0 0 0 1 0 38 0 1 0 3809 0 745 0 3016 0 16 0 925353388 989855744 0 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4001)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(false)}})
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
        "32 0 0 0 1 0 64 0 1 0 1 0 128 0 2 0 16 0 925353388 989855744 0 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4000)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(false)}})
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
        "48 0 0 0 1 0 38 0 1 0 3809 0 745 0 3016 0 16 0 925353388 989855744 0 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1000)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, nullptr, nullptr, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>("")},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
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
        "16 0 4 0 1 0 64 0 1 0 1 0 128 0 0 0 0 0 925353388 989855744 1 0 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_tiling)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(false)}})
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
        "64 0 1 0 1 0 1 0 1 0 0 0 2 0 0 0 2 0 0 0 0 0 0 0 64 0 64 0 64 0 32 0 32 0 512 0 512 0 512 0 1 0 1 0 1024 0 "
        "1536 0 2048 0 925353388 989855744 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_tiling)
{
    dlog_setlevel(0, 0, 0);
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
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape,
                           &ckv_cache_shape, &k_rope_scale_shape, &c_kv_scale_shape, nullptr, nullptr})
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
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

                      .NodeAttrs(
                          {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                           {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
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
        "64 0 1 0 1 0 1 0 1 0 0 0 2 0 0 0 2 0 0 0 1 0 0 0 64 0 64 0 64 0 32 0 32 0 512 0 512 0 512 0 1 0 1 0 1024 0 "
        "1536 0 2048 0 925353388 989855744 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_with_PA_tiling)
{
    dlog_setlevel(0, 0, 0);
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
            .NodeIoNum(11, 4)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&kv_shape, &gamma_shape, &cos_shape, &sin_shape, &index_shape, &k_cache_shape, &ckv_cache_shape,
                 &k_rope_scale_shape, &c_kv_scale_shape, &k_rope_offset_shape, &c_kv_offset_shape})
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
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

            .NodeAttrs(
                {{"epsilon", ge::AnyValue::CreateFrom<float>(1e-05)},
                 {"cache_mode", ge::AnyValue::CreateFrom<std::string>(cache_mode)},
                 {"is_output_kv", ge::AnyValue::CreateFrom<bool>(true)}})
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
        "64 0 1 0 1 0 1 0 128 0 0 0 2 0 2 0 2 0 2 0 1 0 1 0 64 0 64 0 64 0 32 0 32 0 512 0 512 0 512 0 1 0 1 0 1024 0 "
        "1536 0 2048 0 925353388 989855744 ");
    dlog_setlevel(static_cast<int>(OP), 0, 1);
}