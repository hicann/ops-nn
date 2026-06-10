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
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"

class ReluV3Tiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReluV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReluV3Tiling TearDown" << std::endl;
    }
};

TEST_F(ReluV3Tiling, relu_v3_float16_success)
{
    gert::StorageShape x_shape = {{19, 8, 12, 37, 16}, {19, 8, 12, 37, 16}};
    gert::StorageShape y_shape = {{19, 8, 12, 37, 16}, {19, 8, 12, 37, 16}};
    gert::StorageShape mask_shape = {{19, 8, 12, 37, 16}, {19, 8, 12, 37, 16}};
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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

    struct ReluV3TilingCompileInfo {};
    ReluV3TilingCompileInfo compile_info;

    std::string op_type("ReluV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto kernel_holder =
            gert::KernelRunContextFaker()
            .KernelIONum(1, 2)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
            .SetOpType("ReluV3")
            .NodeIoNum(1, 2)
            .IrInstanceNum({1})
            .InputShapes({&x_shape})
            .OutputShapes({&y_shape, &mask_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ReluV3Tiling, relu_v3_float_success)
{
    gert::StorageShape x_shape = {{25, 7, 20, 8, 16}, {25, 7, 20, 8, 16}};
    gert::StorageShape y_shape = {{25, 7, 20, 8, 16}, {25, 7, 20, 8, 16}};
    gert::StorageShape mask_shape = {{25, 7, 20, 8, 16}, {25, 7, 20, 8, 16}};
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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

    struct ReluV3TilingCompileInfo {};
    ReluV3TilingCompileInfo compile_info;

    std::string op_type("ReluV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto kernel_holder =
            gert::KernelRunContextFaker()
            .KernelIONum(1, 2)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
            .SetOpType("ReluV3")
            .NodeIoNum(1, 2)
            .IrInstanceNum({1})
            .InputShapes({&x_shape})
            .OutputShapes({&y_shape, &mask_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ReluV3Tiling, relu_v3_int32_success)
{
    gert::StorageShape x_shape = {{3, 1, 3, 3, 16}, {3, 1, 3, 3, 16}};
    gert::StorageShape y_shape = {{3, 1, 3, 3, 16}, {3, 1, 3, 3, 16}};
    gert::StorageShape mask_shape = {{3, 1, 3, 3, 16}, {3, 1, 3, 3, 16}};
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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

    struct ReluV3TilingCompileInfo {};
    ReluV3TilingCompileInfo compile_info;

    std::string op_type("ReluV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto kernel_holder =
            gert::KernelRunContextFaker()
            .KernelIONum(1, 2)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
            .SetOpType("ReluV3")
            .NodeIoNum(1, 2)
            .IrInstanceNum({1})
            .InputShapes({&x_shape})
            .OutputShapes({&y_shape, &mask_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ReluV3Tiling, relu_v3_int8_success)
{
    gert::StorageShape x_shape = {{66, 9, 45, 275, 32}, {66, 9, 45, 275, 32}};
    gert::StorageShape y_shape = {{66, 9, 45, 275, 32}, {66, 9, 45, 275, 32}};
    gert::StorageShape mask_shape = {{66, 9, 45, 275, 32}, {66, 9, 45, 275, 32}};
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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

    struct ReluV3TilingCompileInfo {};
    ReluV3TilingCompileInfo compile_info;

    std::string op_type("ReluV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto kernel_holder =
            gert::KernelRunContextFaker()
            .KernelIONum(1, 2)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
            .SetOpType("ReluV3")
            .NodeIoNum(1, 2)
            .IrInstanceNum({1})
            .InputShapes({&x_shape})
            .OutputShapes({&y_shape, &mask_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ReluV3Tiling, relu_v3_uint8_success)
{
    gert::StorageShape x_shape = {{3, 1, 2, 3, 32}, {3, 1, 2, 3, 32}};
    gert::StorageShape y_shape = {{3, 1, 2, 3, 32}, {3, 1, 2, 3, 32}};
    gert::StorageShape mask_shape = {{3, 1, 2, 3, 32}, {3, 1, 2, 3, 32}};
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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

    struct ReluV3TilingCompileInfo {};
    ReluV3TilingCompileInfo compile_info;

    std::string op_type("ReluV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto kernel_holder =
            gert::KernelRunContextFaker()
            .KernelIONum(1, 2)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
            .SetOpType("ReluV3")
            .NodeIoNum(1, 2)
            .IrInstanceNum({1})
            .InputShapes({&x_shape})
            .OutputShapes({&y_shape, &mask_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ReluV3Tiling, relu_v3_bfloat16_success)
{
    gert::StorageShape x_shape = {{26, 6, 18, 16, 16}, {26, 6, 18, 16, 16}};
    gert::StorageShape y_shape = {{26, 6, 18, 16, 16}, {26, 6, 18, 16, 16}};
    gert::StorageShape mask_shape = {{26, 6, 18, 16, 16}, {26, 6, 18, 16, 16}};
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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

    struct ReluV3TilingCompileInfo {};
    ReluV3TilingCompileInfo compile_info;

    std::string op_type("ReluV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto kernel_holder =
            gert::KernelRunContextFaker()
            .KernelIONum(1, 2)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
            .SetOpType("ReluV3")
            .NodeIoNum(1, 2)
            .IrInstanceNum({1})
            .InputShapes({&x_shape})
            .OutputShapes({&y_shape, &mask_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
