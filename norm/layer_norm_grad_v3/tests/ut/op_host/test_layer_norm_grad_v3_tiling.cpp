/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "array_ops.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

struct LayerNormGradV3CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    bool isRegBase = false;
};

class LayerNormGradV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LayerNormGradV3Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LayerNormGradV3Tiling TearDown" << std::endl; }
};

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_000)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input1_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input2_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input3_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input4_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape output1_shape = {{4096}, {4096}};
    gert::StorageShape output2_shape = {{4096}, {4096}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 112);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{48, 16}, {48, 16}};
    gert::StorageShape input1_shape = {{48, 16}, {48, 16}};
    gert::StorageShape input2_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input3_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input4_shape = {{16}, {16}};
    gert::StorageShape output0_shape = {{48, 16}, {48, 16}};
    gert::StorageShape output1_shape = {{16}, {16}};
    gert::StorageShape output2_shape = {{16}, {16}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 412);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    //(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input1_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input2_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input3_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input4_shape = {{4095}, {4095}};
    gert::StorageShape output0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape output1_shape = {{4096}, {4096}};
    gert::StorageShape output2_shape = {{4096}, {4096}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_003)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input1_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input2_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input3_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input4_shape = {{48, 1, 4096}, {48, 1, 4096}};
    gert::StorageShape output0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape output1_shape = {{4096}, {4096}};
    gert::StorageShape output2_shape = {{4096}, {4096}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_004)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input1_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input2_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input3_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input4_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape output1_shape = {{4096}, {4096}};
    gert::StorageShape output2_shape = {{4096}, {4096}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 111);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_005)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input1_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input2_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input3_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input4_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape output1_shape = {{4096}, {4096}};
    gert::StorageShape output2_shape = {{4096}, {4096}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 113);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_006)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input1_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input2_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input3_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input4_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape output1_shape = {{4096}, {4096}};
    gert::StorageShape output2_shape = {{4096}, {4096}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 115);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_007)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input1_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape input2_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input3_shape = {{48, 1}, {48, 1}};
    gert::StorageShape input4_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{48, 4096}, {48, 4096}};
    gert::StorageShape output1_shape = {{4096}, {4096}};
    gert::StorageShape output2_shape = {{4096}, {4096}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 114);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_008)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{18, 20}, {18, 20}};
    gert::StorageShape input1_shape = {{18, 20}, {18, 20}};
    gert::StorageShape input2_shape = {{18, 1}, {18, 1}};
    gert::StorageShape input3_shape = {{18, 1}, {18, 1}};
    gert::StorageShape input4_shape = {{20}, {20}};
    gert::StorageShape output0_shape = {{18, 20}, {18, 20}};
    gert::StorageShape output1_shape = {{20}, {20}};
    gert::StorageShape output2_shape = {{20}, {20}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 312);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 18);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_009)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{18, 20000}, {18, 20000}};
    gert::StorageShape input1_shape = {{18, 20000}, {18, 20000}};
    gert::StorageShape input2_shape = {{18, 1}, {18, 1}};
    gert::StorageShape input3_shape = {{18, 1}, {18, 1}};
    gert::StorageShape input4_shape = {{20000}, {20000}};
    gert::StorageShape output0_shape = {{18, 20}, {18, 20000}};
    gert::StorageShape output1_shape = {{20000}, {20000}};
    gert::StorageShape output2_shape = {{20000}, {20000}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 212);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_010)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{18, 20000}, {18, 20000}};
    gert::StorageShape input1_shape = {{18, 20000}, {18, 20000}};
    gert::StorageShape input2_shape = {{18, 1}, {18, 1}};
    gert::StorageShape input3_shape = {{18, 1}, {18, 1}};
    gert::StorageShape input4_shape = {{20000}, {20000}};
    gert::StorageShape output0_shape = {{18, 20}, {18, 20000}};
    gert::StorageShape output1_shape = {{20000}, {20000}};
    gert::StorageShape output2_shape = {{20000}, {20000}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 212);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_011)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{18000, 20}, {18000, 20}};
    gert::StorageShape input1_shape = {{18000, 20}, {18000, 20}};
    gert::StorageShape input2_shape = {{18000, 1}, {18000, 1}};
    gert::StorageShape input3_shape = {{18000, 1}, {18000, 1}};
    gert::StorageShape input4_shape = {{20}, {20}};
    gert::StorageShape output0_shape = {{18000, 20}, {18000, 20}};
    gert::StorageShape output1_shape = {{20}, {20}};
    gert::StorageShape output2_shape = {{20}, {20}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 312);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_012)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input0_shape = {{100, 50}, {100, 50}};
    gert::StorageShape input1_shape = {{100, 50}, {100, 50}};
    gert::StorageShape input2_shape = {{100, 1}, {100, 1}};
    gert::StorageShape input3_shape = {{100, 1}, {100, 1}};
    gert::StorageShape input4_shape = {{50}, {50}};
    gert::StorageShape output0_shape = {{100, 50}, {100, 50}};
    gert::StorageShape output1_shape = {{50}, {50}};
    gert::StorageShape output2_shape = {{50}, {50}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    std::vector<bool> output_mask = {true, true, true};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)}})
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
    // auto tiling_key = tiling_context->GetTilingKey();
    // ASSERT_EQ(tiling_key, 500);
    // auto block_dim = tiling_context->GetBlockDim();
    // ASSERT_EQ(block_dim, 50);
    // dlog_setlevel(static_cast<int>(OP), 0, 1);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_bigm_01)
{
    gert::StorageShape input0_shape = {{10000, 50}, {10000, 50}};
    gert::StorageShape input1_shape = {{10000, 50}, {10000, 50}};
    gert::StorageShape input2_shape = {{10000, 1}, {10000, 1}};
    gert::StorageShape input3_shape = {{10000, 1}, {10000, 1}};
    gert::StorageShape input4_shape = {{50}, {50}};
    gert::StorageShape output0_shape = {{10000, 50}, {10000, 50}};
    gert::StorageShape output1_shape = {{50}, {50}};
    gert::StorageShape output2_shape = {{50}, {50}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    // map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    // kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    std::vector<bool> output_mask = {true, true, true};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 312);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
}

TEST_F(LayerNormGradV3Tiling, layer_norm_grad_v3_tiling_bign_01)
{
    gert::StorageShape input0_shape = {{100, 10000}, {100, 10000}};
    gert::StorageShape input1_shape = {{100, 10000}, {100, 10000}};
    gert::StorageShape input2_shape = {{100, 1}, {100, 1}};
    gert::StorageShape input3_shape = {{100, 1}, {100, 1}};
    gert::StorageShape input4_shape = {{10000}, {10000}};
    gert::StorageShape output0_shape = {{100, 10000}, {100, 10000}};
    gert::StorageShape output1_shape = {{10000}, {10000}};
    gert::StorageShape output2_shape = {{10000}, {10000}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    std::vector<bool> output_mask = {true, true, true};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)}})
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
    ASSERT_EQ(tiling_key, 112);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
}

static void RunTilingTest(gert::StorageShape& input0_shape, gert::StorageShape& input1_shape,
                          gert::StorageShape& input2_shape, gert::StorageShape& input3_shape,
                          gert::StorageShape& input4_shape, gert::StorageShape& output0_shape,
                          gert::StorageShape& output1_shape, gert::StorageShape& output2_shape, ge::DataType dy_dt,
                          ge::DataType x_dt, ge::DataType gamma_dt, ge::DataType pdx_dt, ge::DataType pdgamma_dt,
                          ge::DataType pdbeta_dt, const std::string& compile_info_str, ge::graphStatus expected_status,
                          int64_t expected_tiling_key = -1,
                          std::map<std::string, std::string>* soc_version_infos = nullptr)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_str.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    LayerNormGradV3CompileInfo compile_info;

    std::string op_type("LayerNormGradV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs(
                                 {const_cast<char*>(compile_info_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    if (soc_version_infos != nullptr) {
        kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                                *soc_version_infos);
    }

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dy_dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, x_dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, gamma_dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, pdx_dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, pdgamma_dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, pdbeta_dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), expected_status);
    if (expected_status == ge::GRAPH_SUCCESS && expected_tiling_key >= 0) {
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, expected_tiling_key);
    }
}

static const std::string COMPILE_INFO_910B = R"({
   "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                     "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                     "CORE_NUM": 48}
                     })";

static const std::string COMPILE_INFO_950 = R"({
   "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                     "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                     "CORE_NUM": 64}
                     })";

TEST_F(LayerNormGradV3Tiling, tiling_base_dy_x_shape_mismatch)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 2048}, {48, 2048}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_dy_pdx_shape_mismatch)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 2048}, {48, 2048}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_rstd_mean_shape_mismatch)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 2}, {48, 2}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_gamma_pdgamma_shape_mismatch)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{2048}, {2048}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_gamma_pdbeta_shape_mismatch)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{2048}, {2048}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_dy_dim_less_than_gamma_dim)
{
    gert::StorageShape s0 = {{4096}, {4096}};
    gert::StorageShape s1 = {{4096}, {4096}};
    gert::StorageShape s2 = {{1}, {1}};
    gert::StorageShape s3 = {{1}, {1}};
    gert::StorageShape s4 = {{2, 4096}, {2, 4096}};
    gert::StorageShape o0 = {{4096}, {4096}};
    gert::StorageShape o1 = {{2, 4096}, {2, 4096}};
    gert::StorageShape o2 = {{2, 4096}, {2, 4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_dy_dim_zero)
{
    gert::StorageShape s0 = {{48, 0}, {48, 0}};
    gert::StorageShape s1 = {{48, 0}, {48, 0}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{0}, {0}};
    gert::StorageShape o0 = {{48, 0}, {48, 0}};
    gert::StorageShape o1 = {{0}, {0}};
    gert::StorageShape o2 = {{0}, {0}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_gamma_shape_mismatch_last_dims)
{
    gert::StorageShape s0 = {{2, 3, 64}, {2, 3, 64}};
    gert::StorageShape s1 = {{2, 3, 64}, {2, 3, 64}};
    gert::StorageShape s2 = {{2, 1}, {2, 1}};
    gert::StorageShape s3 = {{2, 1}, {2, 1}};
    gert::StorageShape s4 = {{32}, {32}};
    gert::StorageShape o0 = {{2, 3, 64}, {2, 3, 64}};
    gert::StorageShape o1 = {{32}, {32}};
    gert::StorageShape o2 = {{32}, {32}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_output_mask_wrong_size)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_910B.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    LayerNormGradV3CompileInfo compile_info;
    std::string op_type("LayerNormGradV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO_910B.c_str()),
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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    std::vector<bool> mask = {true, true};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&s0, &s1, &s2, &s3, &s4})
                      .OutputShapes({&o0, &o1, &o2})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(mask)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_x_dy_dtype_mismatch)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_pdx_dy_dtype_mismatch)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_pdgamma_pdbeta_dtype_mismatch)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT16, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_3d_dy_1d_gamma_success)
{
    gert::StorageShape s0 = {{2, 24, 64}, {2, 24, 64}};
    gert::StorageShape s1 = {{2, 24, 64}, {2, 24, 64}};
    gert::StorageShape s2 = {{2, 24}, {2, 24}};
    gert::StorageShape s3 = {{2, 24}, {2, 24}};
    gert::StorageShape s4 = {{64}, {64}};
    gert::StorageShape o0 = {{2, 24, 64}, {2, 24, 64}};
    gert::StorageShape o1 = {{64}, {64}};
    gert::StorageShape o2 = {{64}, {64}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 412);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_3d_dy_2d_gamma_success)
{
    gert::StorageShape s0 = {{2, 24, 64}, {2, 24, 64}};
    gert::StorageShape s1 = {{2, 24, 64}, {2, 24, 64}};
    gert::StorageShape s2 = {{2, 1}, {2, 1}};
    gert::StorageShape s3 = {{2, 1}, {2, 1}};
    gert::StorageShape s4 = {{24, 64}, {24, 64}};
    gert::StorageShape o0 = {{2, 24, 64}, {2, 24, 64}};
    gert::StorageShape o1 = {{24, 64}, {24, 64}};
    gert::StorageShape o2 = {{24, 64}, {24, 64}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 112);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_bf16_bf16_dtype)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT,
                  ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 114);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_fp32_common_small_col)
{
    gert::StorageShape s0 = {{48, 100}, {48, 100}};
    gert::StorageShape s1 = {{48, 100}, {48, 100}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{100}, {100}};
    gert::StorageShape o0 = {{48, 100}, {48, 100}};
    gert::StorageShape o1 = {{100}, {100}};
    gert::StorageShape o2 = {{100}, {100}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                  ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 411);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_output_mask_partial)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_910B.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    LayerNormGradV3CompileInfo compile_info;
    std::string op_type("LayerNormGradV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO_910B.c_str()),
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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    std::vector<bool> mask = {true, false, true};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&s0, &s1, &s2, &s3, &s4})
                      .OutputShapes({&o0, &o1, &o2})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(mask)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_output_mask_all_false)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_910B.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    LayerNormGradV3CompileInfo compile_info;
    std::string op_type("LayerNormGradV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO_910B.c_str()),
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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    std::vector<bool> mask = {false, false, false};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&s0, &s1, &s2, &s3, &s4})
                      .OutputShapes({&o0, &o1, &o2})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(mask)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_ascend950_workspace_fp32)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    std::map<std::string, std::string> soc_ver = {{"Short_SoC_version", "Ascend950"}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                  ge::DT_FLOAT, COMPILE_INFO_950, ge::GRAPH_SUCCESS, -1, &soc_ver);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_ascend950_workspace_bf16)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    std::map<std::string, std::string> soc_ver = {{"Short_SoC_version", "Ascend950"}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT,
                  ge::DT_FLOAT, COMPILE_INFO_950, ge::GRAPH_SUCCESS, -1, &soc_ver);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_ascend950_workspace_fp16_float_gamma)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    std::map<std::string, std::string> soc_ver = {{"Short_SoC_version", "Ascend950"}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_950, ge::GRAPH_SUCCESS, -1, &soc_ver);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_ascend950_bigm_shape)
{
    gert::StorageShape s0 = {{8192, 256}, {8192, 256}};
    gert::StorageShape s1 = {{8192, 256}, {8192, 256}};
    gert::StorageShape s2 = {{8192, 1}, {8192, 1}};
    gert::StorageShape s3 = {{8192, 1}, {8192, 1}};
    gert::StorageShape s4 = {{256}, {256}};
    gert::StorageShape o0 = {{8192, 256}, {8192, 256}};
    gert::StorageShape o1 = {{256}, {256}};
    gert::StorageShape o2 = {{256}, {256}};
    std::map<std::string, std::string> soc_ver = {{"Short_SoC_version", "Ascend950"}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_950, ge::GRAPH_SUCCESS, -1, &soc_ver);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_ascend950_bign_shape)
{
    gert::StorageShape s0 = {{256, 16384}, {256, 16384}};
    gert::StorageShape s1 = {{256, 16384}, {256, 16384}};
    gert::StorageShape s2 = {{256, 1}, {256, 1}};
    gert::StorageShape s3 = {{256, 1}, {256, 1}};
    gert::StorageShape s4 = {{16384}, {16384}};
    gert::StorageShape o0 = {{256, 16384}, {256, 16384}};
    gert::StorageShape o1 = {{16384}, {16384}};
    gert::StorageShape o2 = {{16384}, {16384}};
    std::map<std::string, std::string> soc_ver = {{"Short_SoC_version", "Ascend950"}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_950, ge::GRAPH_SUCCESS, -1, &soc_ver);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_ascend950_recompute_shape)
{
    gert::StorageShape s0 = {{128, 1024}, {128, 1024}};
    gert::StorageShape s1 = {{128, 1024}, {128, 1024}};
    gert::StorageShape s2 = {{128, 1}, {128, 1}};
    gert::StorageShape s3 = {{128, 1}, {128, 1}};
    gert::StorageShape s4 = {{1024}, {1024}};
    gert::StorageShape o0 = {{128, 1024}, {128, 1024}};
    gert::StorageShape o1 = {{1024}, {1024}};
    gert::StorageShape o2 = {{1024}, {1024}};
    std::map<std::string, std::string> soc_ver = {{"Short_SoC_version", "Ascend950"}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_950, ge::GRAPH_SUCCESS, -1, &soc_ver);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_fp32_single_read)
{
    gert::StorageShape s0 = {{48, 8192}, {48, 8192}};
    gert::StorageShape s1 = {{48, 8192}, {48, 8192}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{8192}, {8192}};
    gert::StorageShape o0 = {{48, 8192}, {48, 8192}};
    gert::StorageShape o1 = {{8192}, {8192}};
    gert::StorageShape o2 = {{8192}, {8192}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                  ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 111);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_fp16_float_gamma_single_read)
{
    gert::StorageShape s0 = {{48, 8192}, {48, 8192}};
    gert::StorageShape s1 = {{48, 8192}, {48, 8192}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{8192}, {8192}};
    gert::StorageShape o0 = {{48, 8192}, {48, 8192}};
    gert::StorageShape o1 = {{8192}, {8192}};
    gert::StorageShape o2 = {{8192}, {8192}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 113);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_transpose_col_not_power2)
{
    gert::StorageShape s0 = {{48, 48}, {48, 48}};
    gert::StorageShape s1 = {{48, 48}, {48, 48}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{48}, {48}};
    gert::StorageShape o0 = {{48, 48}, {48, 48}};
    gert::StorageShape o1 = {{48}, {48}};
    gert::StorageShape o2 = {{48}, {48}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 412);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_transpose_power2_col)
{
    gert::StorageShape s0 = {{48, 32}, {48, 32}};
    gert::StorageShape s1 = {{48, 32}, {48, 32}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{32}, {32}};
    gert::StorageShape o0 = {{48, 32}, {48, 32}};
    gert::StorageShape o1 = {{32}, {32}};
    gert::StorageShape o2 = {{32}, {32}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 412);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_gamma_not_float_not_same_as_dy)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_pdgamma_not_float_not_same_as_gamma)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_BF16, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_pdbeta_not_float_not_same_as_gamma)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_BF16, COMPILE_INFO_910B, ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_rstd_not_float)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_910B.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    LayerNormGradV3CompileInfo compile_info;
    std::string op_type("LayerNormGradV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO_910B.c_str()),
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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&s0, &s1, &s2, &s3, &s4})
                      .OutputShapes({&o0, &o1, &o2})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_mean_not_float)
{
    gert::StorageShape s0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s1 = {{48, 4096}, {48, 4096}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{4096}, {4096}};
    gert::StorageShape o0 = {{48, 4096}, {48, 4096}};
    gert::StorageShape o1 = {{4096}, {4096}};
    gert::StorageShape o2 = {{4096}, {4096}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_910B.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    LayerNormGradV3CompileInfo compile_info;
    std::string op_type("LayerNormGradV3");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO_910B.c_str()),
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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNormGradV3")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&s0, &s1, &s2, &s3, &s4})
                      .OutputShapes({&o0, &o1, &o2})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_common_fp32_col_768)
{
    gert::StorageShape s0 = {{48, 768}, {48, 768}};
    gert::StorageShape s1 = {{48, 768}, {48, 768}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{768}, {768}};
    gert::StorageShape o0 = {{48, 768}, {48, 768}};
    gert::StorageShape o1 = {{768}, {768}};
    gert::StorageShape o2 = {{768}, {768}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                  ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 111);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_single_read_boundary_768)
{
    gert::StorageShape s0 = {{48, 769}, {48, 769}};
    gert::StorageShape s1 = {{48, 769}, {48, 769}};
    gert::StorageShape s2 = {{48, 1}, {48, 1}};
    gert::StorageShape s3 = {{48, 1}, {48, 1}};
    gert::StorageShape s4 = {{769}, {769}};
    gert::StorageShape o0 = {{48, 769}, {48, 769}};
    gert::StorageShape o1 = {{769}, {769}};
    gert::StorageShape o2 = {{769}, {769}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 112);
}

TEST_F(LayerNormGradV3Tiling, tiling_base_4d_dy_2d_gamma)
{
    gert::StorageShape s0 = {{2, 3, 4, 64}, {2, 3, 4, 64}};
    gert::StorageShape s1 = {{2, 3, 4, 64}, {2, 3, 4, 64}};
    gert::StorageShape s2 = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape s3 = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape s4 = {{4, 64}, {4, 64}};
    gert::StorageShape o0 = {{2, 3, 4, 64}, {2, 3, 4, 64}};
    gert::StorageShape o1 = {{4, 64}, {4, 64}};
    gert::StorageShape o2 = {{4, 64}, {4, 64}};
    RunTilingTest(s0, s1, s2, s3, s4, o0, o1, o2, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                  ge::DT_FLOAT, ge::DT_FLOAT, COMPILE_INFO_910B, ge::GRAPH_SUCCESS, 412);
}
